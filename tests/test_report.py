from pathlib import Path

from aepk_paging.harness.report import REPORT_PATH, write_report


def test_report_is_generated_with_literal_gate_verdict() -> None:
    write_report()
    report_path = Path(REPORT_PATH)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "## Baseline Matrix" in text
    assert "## Net-Overhead Gate" in text
    assert ("GATE VERDICT: PASS" in text) or ("GATE VERDICT: FAIL" in text)
