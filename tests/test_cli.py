from typer.testing import CliRunner

from llm_trace_inspector.cli import app


def test_eval_threshold_fails_for_risky_trace() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["eval", "examples/rag_trace_hallucinated.json", "--max-risk", "0.4"],
    )

    assert result.exit_code == 1
    assert "Failed threshold" in result.output


def test_eval_dir_writes_reports(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["eval-dir", "examples", "--output-dir", str(tmp_path), "--max-risk", "1.0"],
    )

    assert result.exit_code == 0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "report.html").exists()
