"""LLM extraction: three prompt strategies, structured output, retries, disk cache, offline mock.

Strategies
  zero_shot : minimal instructions + the JSON schema.
  few_shot  : zero-shot instructions + worked examples drawn ONLY from the dev split.
  rules     : zero-shot instructions + explicit rules (number formats, dates, credit notes,
              tax-inclusive totals, discount sign, distractors).

The model name is never hardcoded: resolve_model() takes the CLI value, else $ANTHROPIC_MODEL.
The API key is read by the SDK from $ANTHROPIC_API_KEY only.

Cache: .cache/extractions/<sha256>.json, keyed by sha256(model, strategy, document text, prompt
fingerprint). The fingerprint covers system prompt, few-shot messages, schema and generation settings,
so editing a prompt automatically invalidates stale entries. Only successful extractions are cached;
failures are retried on the next run.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic
import pdfplumber

from .schema import ALL_FIELDS, LINE_ITEM_FIELDS, json_schema

STRATEGIES = ("zero_shot", "few_shot", "rules")
PROMPT_VERSION = "v1"
DEFAULT_CACHE_DIR = Path(".cache/extractions")
MAX_TOKENS = 16000
RETRYABLE = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError,
             anthropic.OverloadedError, anthropic.ServiceUnavailableError)


class ConfigError(RuntimeError):
    pass


class MalformedOutput(Exception):
    def __init__(self, msg: str, grow_tokens: bool = False):
        super().__init__(msg)
        self.grow_tokens = grow_tokens


class Refusal(Exception):
    pass


# ----------------------------------------------------------------------------- config / text
def resolve_model(cli_value: str | None = None) -> str:
    """CLI flag wins; otherwise $ANTHROPIC_MODEL. There is deliberately no default model."""
    model = (cli_value or os.environ.get("ANTHROPIC_MODEL") or "").strip()
    if not model:
        raise ConfigError("No model given: pass --model or set ANTHROPIC_MODEL (no default is hardcoded).")
    return model


def pdf_to_text(path: str | Path) -> str:
    """Layout-preserving text of all pages (long space runs collapsed to keep tokens down)."""
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text(layout=True) or "" for p in pdf.pages]
    cleaned = []
    for page in pages:
        lines = [re.sub(r" {4,}", "   ", ln).rstrip() for ln in page.splitlines()]
        txt = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")
        cleaned.append(txt)
    if len(cleaned) == 1:
        return cleaned[0]
    return "".join(f"\n--- page {i} ---\n{t}\n" for i, t in enumerate(cleaned, 1)).strip("\n")


# ----------------------------------------------------------------------------- prompts
BASE_SYSTEM = (
    "You extract structured data from the text of a vendor invoice or credit note. "
    "Return only data that matches the provided JSON schema. Use null for a nullable field that is "
    "not present in the document."
)

RULES = """
Follow these rules exactly.

Numbers
- Output plain JSON numbers with no thousands separators or currency symbols.
- Documents use either US format (1,234.56) or European format (1.234,56). Decide from the position
  of the separators; the last separator is normally the decimal mark. Quantities can be fractional.

Dates
- Output ISO format YYYY-MM-DD.
- Dotted dates (15.03.2024) are day.month.year. For slash dates (03/04/2024) decide day-first or
  month-first from any date in the document whose first or second part exceeds 12; use the same
  convention for every date in that document.

Credit notes
- document_type is "credit_note" if the document is titled credit note / credit memo, else "invoice".
- On a credit note, subtotal, discount, tax_amount, total and every line item amount must be NEGATIVE
  numbers, even when printed without a minus sign or inside parentheses. quantity and unit_price stay positive.
- invoice_number is the credit note's own number, not the number of the original invoice it refers to.

Tax and totals
- tax_rate is a percentage (20 for 20%); null if no rate is printed. tax_amount is the tax amount.
- If prices are tax-inclusive (labels like "incl. tax", "of which tax", "includes tax"), keep line item
  amounts exactly as printed (gross), set total to the gross total, tax_amount to the tax portion,
  discount to 0, and subtotal = total - tax_amount, even if no net amount is printed.
- discount is the discount AMOUNT (not the percentage). On invoices it is a positive number even when
  printed with a minus sign; on credit notes it is negative. Use 0 when there is no discount.

Other
- currency is the ISO 4217 code printed in the document.
- Join a multi-line item description into a single string separated by single spaces.
- Include every line item, including items on later pages.
- Ignore distractors: PO numbers, customer IDs, order numbers, phone numbers, IBANs, stamps, scan
  artifacts, previous balance, deposits received and credit limits.
"""

ZERO_SHOT_SYSTEM = BASE_SYSTEM
RULES_SYSTEM = BASE_SYSTEM + "\n" + RULES.strip("\n")


def format_user(doc_text: str) -> str:
    return f"Extract the invoice data from this document.\n\n<document>\n{doc_text}\n</document>"


def select_few_shot(dev_records: list[dict], k: int = 3) -> list[dict]:
    """Deterministically pick k non-hard dev records covering distinct features (greedy, doc_id order).

    Raises if any record is not from the dev split (guards against test leakage).
    """
    if any(r["split"] != "dev" for r in dev_records):
        raise ValueError("few-shot examples may only come from the dev split")

    def tags(r):
        f = r["meta"]["features"]
        return {f"type:{f['document_type']}", f"num:{f['number_format']}", f"incl:{f['tax_inclusive']}",
                f"multi:{f['multiline_descriptions']}", f"tpl:{r['meta']['template']}"}

    pool = sorted((r for r in dev_records if not r["meta"]["hard"]), key=lambda r: r["doc_id"])
    chosen, covered = [], set()
    while pool and len(chosen) < k:
        best = max(pool, key=lambda r: (len(tags(r) - covered), -pool.index(r)))
        chosen.append(best)
        covered |= tags(best)
        pool.remove(best)
    return chosen


def load_few_shot_examples(data_dir: str | Path, k: int = 3) -> list[dict]:
    """Return [{doc_id, text, truth}] from the dev split only."""
    from .generate import load_split
    dev = load_split(Path(data_dir), "dev")
    return [{"doc_id": r["doc_id"], "text": pdf_to_text(r["pdf_path"]), "truth": r["truth"]}
            for r in select_few_shot(dev, k)]


# ----------------------------------------------------------------------------- results
@dataclass
class ExtractionResult:
    doc_id: str
    extraction: dict | None
    ok: bool
    error: str | None = None
    cached: bool = False
    attempts: int = 0
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    temperature_applied: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def check_shape(obj) -> dict:
    """Minimal structural check (no jsonschema dependency); returns obj restricted to schema fields."""
    if not isinstance(obj, dict):
        raise MalformedOutput("top-level JSON is not an object")
    missing = [f for f in ALL_FIELDS if f not in obj]
    if missing:
        raise MalformedOutput("missing fields: " + ", ".join(missing))
    items = obj["line_items"]
    if not isinstance(items, list) or not all(isinstance(i, dict) and all(k in i for k in LINE_ITEM_FIELDS)
                                              for i in items):
        raise MalformedOutput("line_items malformed")
    return {f: obj[f] for f in ALL_FIELDS}


def save_results(results: dict[str, ExtractionResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc_id in sorted(results):
            fh.write(json.dumps(results[doc_id].to_dict(), ensure_ascii=False) + "\n")


def load_results(path: str | Path) -> dict[str, ExtractionResult]:
    out = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            d = json.loads(ln)
            out[d["doc_id"]] = ExtractionResult(**d)
    return out


# ----------------------------------------------------------------------------- LLM extractor
class LLMExtractor:
    def __init__(self, model: str, strategy: str, *, client=None, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                 examples: list[dict] | None = None, max_attempts: int = 5, max_tokens: int = MAX_TOKENS,
                 sleep=time.sleep, rng: random.Random | None = None, send_temperature: bool = True):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
        if strategy == "few_shot" and not examples:
            raise ValueError("few_shot strategy needs examples (see load_few_shot_examples)")
        if strategy != "few_shot" and examples:
            raise ValueError("examples are only used by the few_shot strategy")
        self.model, self.strategy = model, strategy
        self.examples = examples or []
        self.cache_dir = Path(cache_dir)
        self.max_attempts, self.max_tokens = max_attempts, max_tokens
        self._client = client
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        self._send_temperature = send_temperature

    # -- prompt construction
    @property
    def system(self) -> str:
        return RULES_SYSTEM if self.strategy == "rules" else ZERO_SHOT_SYSTEM

    def build_messages(self, doc_text: str) -> list[dict]:
        msgs: list[dict] = []
        for ex in self.examples:
            msgs.append({"role": "user", "content": format_user(ex["text"])})
            msgs.append({"role": "assistant", "content": json.dumps(ex["truth"], ensure_ascii=False)})
        msgs.append({"role": "user", "content": format_user(doc_text)})
        return msgs

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({"v": PROMPT_VERSION, "system": self.system, "schema": json_schema(),
                           "shots": [(e["doc_id"], e["text"], e["truth"]) for e in self.examples],
                           "temperature": 0}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def cache_key(self, doc_text: str) -> str:
        blob = json.dumps({"model": self.model, "strategy": self.strategy,
                           "doc": hashlib.sha256(doc_text.encode()).hexdigest(), "prompt": self.fingerprint},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- API
    @property
    def client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ConfigError("ANTHROPIC_API_KEY is not set.")
            self._client = anthropic.Anthropic(max_retries=0)  # retries/backoff are handled here
        return self._client

    def _create(self, messages: list[dict], max_tokens: int):
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=self.system, messages=messages,
                      output_config={"format": {"type": "json_schema", "schema": json_schema()}})
        if self._send_temperature:
            try:
                # SDK 1.x has no typed `temperature` argument; some models reject it entirely.
                return self.client.messages.create(**kwargs, extra_body={"temperature": 0})
            except anthropic.BadRequestError as e:
                if "temperature" not in str(e).lower():
                    raise
                self._send_temperature = False  # model rejects sampling params; recorded in the result
        return self.client.messages.create(**kwargs)

    def _delay(self, attempt: int, err: Exception) -> float:
        headers = getattr(getattr(err, "response", None), "headers", None) or {}
        try:
            ra = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            ra = None
        backoff = min(60.0, 2.0 ** attempt) * (0.5 + self._rng.random())
        return max(ra or 0.0, backoff)

    @staticmethod
    def _parse(resp) -> dict:
        if resp.stop_reason == "refusal":
            raise Refusal("model refused the request")
        if resp.stop_reason == "max_tokens":
            raise MalformedOutput("output truncated at max_tokens", grow_tokens=True)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise MalformedOutput(f"invalid JSON: {e}") from e
        return check_shape(obj)

    def extract(self, doc_id: str, doc_text: str) -> ExtractionResult:
        path = self.cache_dir / f"{self.cache_key(doc_text)}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return ExtractionResult(doc_id=doc_id, extraction=d["extraction"], ok=True, cached=True,
                                    attempts=d["attempts"], stop_reason=d["stop_reason"], usage=d["usage"],
                                    temperature_applied=d["temperature_applied"])
        messages, max_tokens, last_err = self.build_messages(doc_text), self.max_tokens, "unknown"
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._create(messages, max_tokens)
                obj = self._parse(resp)
            except Refusal as e:
                return ExtractionResult(doc_id, None, False, str(e), attempts=attempt, stop_reason="refusal")
            except MalformedOutput as e:
                last_err = f"malformed output: {e}"
                if e.grow_tokens:
                    max_tokens *= 2
                continue
            except RETRYABLE as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_attempts:
                    self._sleep(self._delay(attempt, e))
                continue
            usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0),
                     "output_tokens": getattr(resp.usage, "output_tokens", 0)}
            res = ExtractionResult(doc_id, obj, True, attempts=attempt, stop_reason=resp.stop_reason, usage=usage,
                                   temperature_applied=self._send_temperature)
            self._write_cache(path, res)
            return res
        return ExtractionResult(doc_id, None, False, f"gave up after {self.max_attempts} attempts: {last_err}",
                                attempts=self.max_attempts)

    def _write_cache(self, path: Path, res: ExtractionResult) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"model": self.model, "strategy": self.strategy, "doc_id": res.doc_id, "extraction": res.extraction,
               "attempts": res.attempts, "stop_reason": res.stop_reason, "usage": res.usage,
               "temperature_applied": res.temperature_applied}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


# ----------------------------------------------------------------------------- offline mock
class MockExtractor:
    """Offline stand-in for tests and dry runs. NOT a model: it perturbs ground truth with seeded errors.

    Mock outputs exist only to exercise the pipeline (validation, metrics, reports). They must never
    be reported as model results.
    """
    strategy = "mock"
    model = "mock"
    ERRORS = ("total_off", "dropped_line", "swapped_day_month", "vendor_typo", "sign_flip", "wrong_tax_rate")

    def __init__(self, truths: dict[str, dict], error_rate: float = 0.3, seed: int = 0):
        self.truths, self.error_rate, self.seed = truths, error_rate, seed

    def extract(self, doc_id: str, doc_text: str = "") -> ExtractionResult:
        import copy
        rng = random.Random(f"{self.seed}-{doc_id}")
        x = copy.deepcopy(self.truths[doc_id])
        if rng.random() < self.error_rate:
            kind = rng.choice(self.ERRORS)
            if kind == "total_off":
                x["total"] = round(x["total"] + rng.choice([-1, 1]) * rng.randint(5, 90), 2)
            elif kind == "dropped_line" and len(x["line_items"]) > 1:
                x["line_items"].pop(rng.randrange(len(x["line_items"])))
            elif kind == "swapped_day_month":
                y, m, d = x["invoice_date"].split("-")
                if int(d) <= 12:
                    x["invoice_date"] = f"{y}-{d}-{m}"
                else:
                    x["vendor_name"] = x["vendor_name"].upper() + " X"
            elif kind == "vendor_typo":
                x["vendor_name"] = x["vendor_name"][:-1] + "x"
            elif kind == "sign_flip" and x["document_type"] == "credit_note":
                for k in ("subtotal", "discount", "tax_amount", "total"):
                    x[k] = abs(x[k])
            elif kind == "wrong_tax_rate" and x["tax_rate"] is not None:
                x["tax_rate"] = x["tax_rate"] + 5
            else:
                x["invoice_number"] = x["invoice_number"] + "0"
        return ExtractionResult(doc_id, x, True, attempts=1, stop_reason="end_turn",
                                usage={"input_tokens": 0, "output_tokens": 0}, temperature_applied=None)


def extract_all(extractor, docs: list[dict], progress: bool = True) -> dict[str, ExtractionResult]:
    """Run an extractor over [{doc_id, text}] (text only: ground truth is never passed)."""
    out = {}
    for i, d in enumerate(docs, 1):
        out[d["doc_id"]] = extractor.extract(d["doc_id"], d["text"])
        if progress:
            r = out[d["doc_id"]]
            print(f"[{i}/{len(docs)}] {d['doc_id']} {'cached' if r.cached else 'live/mock'} ok={r.ok}", flush=True)
    return out
