"""Pre-run cost estimate. Estimates only; the real bill comes from the API usage you actually incur.

Prices are USD per 1M tokens (input, output) copied from Anthropic's published pricing as of
2026-09-25. They may be out of date: verify, or pass --price-in/--price-out to override.
"""
from __future__ import annotations

import json
import math

from .schema import json_schema

PRICES_PER_MTOK = {
    "claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0), "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
PRICES_AS_OF = "2026-09-25"
CHARS_PER_TOKEN = 3.5      # rough for English/numeric text
SAFETY_MARGIN = 1.15       # inflate input estimate a little
DEFAULT_OUT_TOKENS = 700   # assumed output tokens per document (the JSON is ~300-500; headroom for retries)


def lookup_price(model: str, price_in: float | None = None, price_out: float | None = None):
    """(input, output) USD per 1M tokens, or None if unknown. Overrides win; then exact id; then longest prefix."""
    if price_in is not None and price_out is not None:
        return (price_in, price_out)
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    for known in sorted(PRICES_PER_MTOK, key=len, reverse=True):
        if model.startswith(known):
            return PRICES_PER_MTOK[known]
    return None


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_strategy(extractor, docs: list[dict], out_tokens_per_doc: int = DEFAULT_OUT_TOKENS) -> dict:
    """Token estimate for the documents NOT already in the cache (cached documents cost nothing)."""
    schema_tokens = estimate_tokens(json.dumps(json_schema()))
    todo = [d for d in docs if not (extractor.cache_dir / f"{extractor.cache_key(d['text'])}.json").exists()]
    in_tok = 0
    for d in todo:
        chars = len(extractor.system) + sum(len(m["content"]) for m in extractor.build_messages(d["text"]))
        in_tok += math.ceil(math.ceil(chars / CHARS_PER_TOKEN) * SAFETY_MARGIN) + schema_tokens
    return {"strategy": extractor.strategy, "n_docs": len(docs), "n_to_call": len(todo),
            "input_tokens": in_tok, "output_tokens": len(todo) * out_tokens_per_doc}


def cost_usd(est: dict, price: tuple[float, float] | None) -> float | None:
    if price is None:
        return None
    return est["input_tokens"] * price[0] / 1e6 + est["output_tokens"] * price[1] / 1e6


def format_estimate(estimates: list[dict], model: str, price, out_tokens_per_doc: int) -> str:
    lines = [f"Estimated cost for model '{model}' (documents already cached are free):",
             f"{'strategy':<10} {'docs':>5} {'to call':>8} {'est. in tok':>12} {'est. out tok':>13} {'est. cost':>10}"]
    total = 0.0
    for e in estimates:
        c = cost_usd(e, price)
        total += c or 0.0
        lines.append(f"{e['strategy']:<10} {e['n_docs']:>5} {e['n_to_call']:>8} {e['input_tokens']:>12,} "
                     f"{e['output_tokens']:>13,} {('$%.4f' % c) if c is not None else 'unknown':>10}")
    if price is not None:
        lines.append(f"{'TOTAL':<10} {'':>5} {sum(e['n_to_call'] for e in estimates):>8} "
                     f"{sum(e['input_tokens'] for e in estimates):>12,} {sum(e['output_tokens'] for e in estimates):>13,} "
                     f"{'$%.4f' % total:>10}")
        lines.append(f"Pricing assumed: ${price[0]}/M input, ${price[1]}/M output (reference table dated "
                     f"{PRICES_AS_OF}, or your override).")
    else:
        lines.append(f"No price known for '{model}': pass --price-in and --price-out (USD per 1M tokens) "
                     "for a dollar estimate.")
    lines.append(f"Assumes ~{out_tokens_per_doc} output tokens per document. If the model uses thinking, thinking "
                 "tokens are NOT included, so the real cost can be higher. Treat this as a rough lower bound.")
    return "\n".join(lines)
