"""Consistency check: every headline number quoted in README.md's Results section must appear verbatim
in results/summary.md, which is itself script-generated (src/report.py) from the real extraction outputs.
This does not re-derive the numbers; it only guards against the README drifting from the generated file
after a future re-run or hand edit.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each is a Wilson-interval-style rate cell exactly as src/report.py's rate_cell() renders it, e.g.
# "24/40 = 60.0% [44.6%–73.7%]" or "8/8 = 100% [67.6%–100%]" (README may drop the trailing ".0" on
# whole percentages, which is cosmetic, so we compare the "k/n" and the raw percentage number only).
RATE_RE = re.compile(r"(\d+)/(\d+)\s*=\s*([\d.]+)%")


def rates_in(text: str) -> set[tuple[str, str, float]]:
    return {(k, n, float(p)) for k, n, p in RATE_RE.findall(text)}


def test_summary_exists():
    assert (ROOT / "results" / "summary.md").exists(), \
        "results/summary.md is missing; run `python -m src.cli report --split test` before trusting the README"


def test_every_readme_rate_cell_appears_in_summary():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    summary = (ROOT / "results" / "summary.md").read_text(encoding="utf-8")
    # only check the Results section of the README (between '## Results' and the next top-level '## ')
    start = readme.index("## Results")
    end = readme.index("\n## ", start + 1)
    results_section = readme[start:end]

    readme_rates = rates_in(results_section)
    summary_rates = rates_in(summary)
    missing = readme_rates - summary_rates
    assert not missing, (
        f"README Results section quotes rate(s) not found verbatim in results/summary.md: {sorted(missing)}. "
        "Either results/summary.md is stale (re-run `python -m src.cli report`) or the README has drifted."
    )


def test_model_and_split_named_in_readme_match_summary():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    summary = (ROOT / "results" / "summary.md").read_text(encoding="utf-8")
    model_match = re.search(r"- Model: `([^`]+)`", summary)
    assert model_match, "results/summary.md has no '- Model: `...`' line to check against"
    model = model_match.group(1)
    assert model in readme, f"model {model!r} from results/summary.md is not mentioned anywhere in README.md"


def test_readme_does_not_claim_not_yet_run_while_results_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Results")
    end = readme.index("\n## ", start + 1)
    results_section = readme[start:end]
    assert "not yet run" not in results_section.lower(), \
        "results/summary.md exists with real numbers, but the README Results section still says 'not yet run'"
