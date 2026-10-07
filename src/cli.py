"""Command-line entry point.

  python -m src.cli generate
  python -m src.cli run --split dev  --strategy all            # prompt development
  python -m src.cli run --split test --strategy all            # the reported run (once per strategy)
  python -m src.cli report --split test                        # writes results/summary.md + error_analysis.md
  python -m src.cli run --mock --split test --yes              # offline pipeline check (never real results)
  python -m src.cli description-audit --split test             # POST-HOC: writes results/description_audit.md
                                                                 # (reads cached results only, no API calls)
  python -m src.cli stress-validator                            # POST-HOC: writes results/validator_stress_test.md
                                                                 # (fault-injection on ground truth; no API calls)

Live runs print a cost estimate and ask for confirmation first (skip with --yes). The model comes from
--model or $ANTHROPIC_MODEL; the API key only from $ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from . import generate as gen
from .cost import DEFAULT_OUT_TOKENS, estimate_strategy, format_estimate, lookup_price
from .extract import (STRATEGIES, ConfigError, LLMExtractor, MockExtractor, extract_all, load_few_shot_examples,
                      load_results, pdf_to_text, resolve_model, save_results)
from .description_audit import write_audit
from .generate import DEFAULT_SEED
from .report import write_reports
from .stress_validator import write_stress_report

FATAL_API_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                    anthropic.BadRequestError)


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def _result_path(out_dir: Path, label: str, split: str, strategy: str) -> Path:
    return out_dir / _slug(label) / split / f"{strategy}.jsonl"


# ----------------------------------------------------------------------------- test-run log
def _log_path(results_dir: Path) -> Path:
    return results_dir / "run_log.json"


def read_log(results_dir: Path) -> list[dict]:
    p = _log_path(results_dir)
    return json.loads(p.read_text(encoding="utf-8"))["test_runs"] if p.exists() else []


def _write_log(results_dir: Path, entries: list[dict]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    _log_path(results_dir).write_text(json.dumps({"test_runs": entries}, indent=2) + "\n", encoding="utf-8")


def changed_prompts(entries: list[dict], model: str) -> list[str]:
    """Strategies whose prompt fingerprint differs between logged test runs for this model."""
    fps: dict[str, set] = {}
    for e in entries:
        if e["model"] == model:
            fps.setdefault(e["strategy"], set()).add(e["fingerprint"])
    return sorted(s for s, v in fps.items() if len(v) > 1)


# ----------------------------------------------------------------------------- commands
def _load_docs(data_dir: Path, split: str):
    records = gen.load_split(data_dir, split)
    if not records:
        raise ConfigError(f"No {split} documents under {data_dir}. Run `python -m src.cli generate` first.")
    return records, [{"doc_id": r["doc_id"], "text": pdf_to_text(r["pdf_path"])} for r in records]


def cmd_generate(a) -> int:
    gen.main(["--out", str(a.data_dir), "--seed", str(a.seed)])
    return 0


def cmd_run(a, confirm, client) -> int:
    strategies = list(STRATEGIES) if a.strategy == "all" else [a.strategy]
    data_dir, out_dir, results_dir = Path(a.data_dir), Path(a.out_dir), Path(a.results_dir)
    records, docs = _load_docs(data_dir, a.split)

    if a.mock:
        label = "mock"
        truths = {r["doc_id"]: r["truth"] for r in records}
        extractors = {s: MockExtractor(truths, a.mock_error_rate, a.mock_seed + i) for i, s in enumerate(strategies)}
        print(f"MOCK run on {a.split}: no API calls. Outputs are pipeline checks, not model results.")
    else:
        label = resolve_model(a.model)
        if not a.dry_run and client is None and not _has_key():
            raise ConfigError("ANTHROPIC_API_KEY is not set.")
        examples = load_few_shot_examples(data_dir) if "few_shot" in strategies else None
        extractors = {s: LLMExtractor(label, s, client=client, cache_dir=a.cache_dir,
                                      examples=examples if s == "few_shot" else None) for s in strategies}
        estimates = [estimate_strategy(e, docs, a.out_tokens) for e in extractors.values()]
        print(format_estimate(estimates, label, lookup_price(label, a.price_in, a.price_out), a.out_tokens))
        if a.split == "test":
            prior = read_log(results_dir)
            clash = [s for s, e in extractors.items()
                     if any(p["model"] == label and p["strategy"] == s and p["fingerprint"] != e.fingerprint
                            for p in prior)]
            if clash and not a.allow_test_rerun:
                print(f"REFUSED: the test split was already scored for {clash} with a different prompt. Re-scoring "
                      "after changing prompts breaks the run-test-once protocol. Pass --allow-test-rerun to override "
                      "(the report will carry a warning).")
                return 3
        if a.dry_run:
            print("Dry run: nothing executed.")
            return 0
        to_call = sum(e["n_to_call"] for e in estimates)
        if to_call and not a.yes:
            warn = ("\nThis is the TEST split. Run it once per strategy and do not edit prompts afterwards.\n"
                    if a.split == "test" else "")
            ans = confirm(f"{warn}Make {to_call} live API calls? [y/N] ")
            if ans.strip().lower() not in ("y", "yes"):
                print("Aborted; no API calls made.")
                return 1
        elif not to_call:
            print("Everything is already cached: no API calls needed.")

    for s, ex in extractors.items():
        print(f"== {s} on {a.split} ==")
        try:
            results = extract_all(ex, docs)
        except FATAL_API_ERRORS as e:
            print(f"error: {type(e).__name__}: {e}\nStopping. Completed responses are cached; rerun to resume.")
            return 2
        save_results(results, _result_path(out_dir, label, a.split, s))
        ok = sum(r.ok for r in results.values())
        live = sum(1 for r in results.values() if r.ok and not r.cached)
        print(f"{s}: {ok}/{len(results)} ok, {live} live calls, "
              f"{sum(r.usage.get('input_tokens', 0) for r in results.values()):,} input / "
              f"{sum(r.usage.get('output_tokens', 0) for r in results.values()):,} output tokens recorded")
        if not a.mock and a.split == "test":
            entries = read_log(results_dir)
            entries.append({"model": label, "strategy": s, "fingerprint": ex.fingerprint, "n_docs": len(docs),
                            "live_calls": live, "rerun_allowed": bool(a.allow_test_rerun),
                            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            _write_log(results_dir, entries)
    return 0


def cmd_description_audit(a) -> int:
    """POST-HOC. Reads cached extraction results already saved by `run`; makes no API calls and touches
    neither the cache nor any extraction output."""
    data_dir, out_dir = Path(a.data_dir), Path(a.out_dir)
    records = gen.load_split(data_dir, a.split)
    if not records:
        raise ConfigError(f"No {a.split} documents under {data_dir}.")
    label = "mock" if a.mock else resolve_model(a.model)
    runs = {}
    for s in STRATEGIES:
        p = _result_path(out_dir, label, a.split, s)
        if p.exists():
            runs[s] = load_results(p)
        else:
            print(f"warning: no saved results for strategy '{s}' at {p}")
    if not runs:
        raise ConfigError("No saved results found. Run `python -m src.cli run ...` first.")
    dest = Path(a.report_dir) / "description_audit.md" if a.report_dir else Path(a.results_dir) / "description_audit.md"
    p = write_audit(dest, runs, records)
    print(f"Wrote {p} (post-hoc; read {len(runs)} cached strategy result file(s), made no API calls)")
    return 0


def cmd_stress_validator(a) -> int:
    """POST-HOC. Fault-injection test of src/validate.py against the seeded ground-truth records.
    Makes no API calls and does not read or write any cached extraction."""
    records = gen.build_records(a.seed)
    dest = Path(a.results_dir) / "validator_stress_test.md"
    p = write_stress_report(dest, records, seed=a.stress_seed)
    print(f"Wrote {p} (post-hoc; fault-injection only, made no API calls)")
    return 0


def cmd_report(a) -> int:
    data_dir, out_dir, results_dir = Path(a.data_dir), Path(a.out_dir), Path(a.results_dir)
    records = gen.load_split(data_dir, a.split)
    if not records:
        raise ConfigError(f"No {a.split} documents under {data_dir}.")
    label = "mock" if a.mock else resolve_model(a.model)
    runs = {}
    for s in STRATEGIES:
        p = _result_path(out_dir, label, a.split, s)
        if p.exists():
            runs[s] = load_results(p)
        else:
            print(f"warning: no saved results for strategy '{s}' at {p}")
    if not runs:
        raise ConfigError("No saved results found. Run `python -m src.cli run ...` first.")
    notes = []
    if not a.mock and a.split == "test":
        entries = read_log(results_dir)
        changed = changed_prompts(entries, label)
        logged = {e["strategy"] for e in entries if e["model"] == label}
        if changed:
            notes.append(f"WARNING: the prompt for {', '.join(changed)} changed after the test split was first scored; "
                         "this violates the run-test-once protocol.")
        elif set(runs) <= logged:
            notes.append("Test split was scored with one fixed prompt per strategy (see results/run_log.json).")
        else:
            notes.append("WARNING: no run-log entry for some strategies; the run-test-once protocol cannot be verified.")
    if a.report_dir:
        dest = Path(a.report_dir)
    elif a.mock:
        dest = out_dir / "mock_report"
    elif a.split != "test":
        dest = out_dir / f"{a.split}_report"
    else:
        dest = results_dir
    meta = {"model": label, "mock": a.mock, "split": a.split, "seed": a.seed, "notes": notes}
    sp, ep = write_reports(dest, records, runs, meta)
    print(f"Wrote {sp} and {ep}")
    return 0


# ----------------------------------------------------------------------------- argument parsing
def _has_key() -> bool:
    import os
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data-dir", default="data/generated")
        sp.add_argument("--out-dir", default="outputs", help="where raw extraction outputs are stored")
        sp.add_argument("--results-dir", default="results", help="where reports and the test run log go")
        sp.add_argument("--seed", type=int, default=gen.DEFAULT_SEED)

    g = sub.add_parser("generate", help="generate the synthetic dataset")
    common(g)
    r = sub.add_parser("run", help="extract with an LLM (or --mock)")
    common(r)
    r.add_argument("--split", choices=("dev", "test"), default="dev")
    r.add_argument("--strategy", choices=STRATEGIES + ("all",), default="all")
    r.add_argument("--model", help="overrides $ANTHROPIC_MODEL")
    r.add_argument("--cache-dir", default=".cache/extractions")
    r.add_argument("--mock", action="store_true", help="offline mock extractor; never produces real results")
    r.add_argument("--mock-error-rate", type=float, default=0.3)
    r.add_argument("--mock-seed", type=int, default=0)
    r.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    r.add_argument("--dry-run", action="store_true", help="print the cost estimate and exit")
    r.add_argument("--out-tokens", type=int, default=DEFAULT_OUT_TOKENS, help="assumed output tokens per document")
    r.add_argument("--price-in", type=float, help="USD per 1M input tokens (overrides the price table)")
    r.add_argument("--price-out", type=float, help="USD per 1M output tokens (overrides the price table)")
    r.add_argument("--allow-test-rerun", action="store_true",
                   help="allow re-scoring the test split after a prompt change (the report will warn)")
    rp = sub.add_parser("report", help="write summary.md and error_analysis.md from saved results")
    common(rp)
    rp.add_argument("--split", choices=("dev", "test"), default="test")
    rp.add_argument("--model", help="overrides $ANTHROPIC_MODEL")
    rp.add_argument("--mock", action="store_true")
    rp.add_argument("--report-dir", help="override the output directory")

    da = sub.add_parser("description-audit", help="POST-HOC: audit line-item description mismatches from cache")
    common(da)
    da.add_argument("--split", choices=("dev", "test"), default="test")
    da.add_argument("--model", help="overrides $ANTHROPIC_MODEL")
    da.add_argument("--mock", action="store_true")
    da.add_argument("--report-dir", help="override the output directory")

    sv = sub.add_parser("stress-validator", help="POST-HOC: fault-injection test of src/validate.py")
    common(sv)
    sv.add_argument("--stress-seed", type=int, default=0, help="seed for the corruption RNG (not the dataset seed)")
    return p


def main(argv=None, *, confirm=input, client=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "generate":
            return cmd_generate(a)
        if a.cmd == "run":
            return cmd_run(a, confirm, client)
        if a.cmd == "description-audit":
            return cmd_description_audit(a)
        if a.cmd == "stress-validator":
            return cmd_stress_validator(a)
        return cmd_report(a)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
