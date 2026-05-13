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
        [
            "eval-dir",
            "examples",
            "--output-dir",
            str(tmp_path),
            "--max-risk",
            "1.0",
            "--min-groundedness",
            "0",
            "--min-citation-coverage",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "report.html").exists()


def test_validate_accepts_good_trace() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["validate", "examples/rag_trace_good.json"])

    assert result.exit_code == 0
    assert "Valid trace" in result.output


def test_eval_uses_config_thresholds(tmp_path) -> None:
    config = tmp_path / "strict.toml"
    config.write_text(
        "[thresholds]\nmax_risk = 0.9\nmin_groundedness = 0.9\nmin_citation_coverage = 0.9\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["eval", "examples/rag_trace_hallucinated.json", "--config", str(config)],
    )

    assert result.exit_code == 1
    assert "Failed thresholds" in result.output
