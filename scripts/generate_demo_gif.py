from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.gif"
W, H = 1280, 720


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = font(24)
FONT_SM = font(19)
FONT_XS = font(16)
FONT_BOLD = font(28, bold=True)
FONT_HERO = font(44, bold=True)
FONT_MONO = font(19)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline, width=2 if outline else 1)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fill: str = "#1f2937", fnt=FONT) -> None:
    draw.text(xy, value, fill=fill, font=fnt)


def base(title: str, subtitle: str) -> Image.Image:
    image = Image.new("RGB", (W, H), "#f6f8fc")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 86), fill="#111827")
    text(draw, (42, 22), "LLM Trace Inspector", "#ffffff", FONT_BOLD)
    text(draw, (42, 104), title, "#111827", FONT_HERO)
    text(draw, (44, 162), subtitle, "#526070", FONT)
    return image


def terminal(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[tuple[str, str]]) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, "#101827", "#263244")
    draw.rectangle((x1, y1, x2, y1 + 42), fill="#182235")
    draw.ellipse((x1 + 18, y1 + 15, x1 + 30, y1 + 27), fill="#ef4444")
    draw.ellipse((x1 + 42, y1 + 15, x1 + 54, y1 + 27), fill="#f59e0b")
    draw.ellipse((x1 + 66, y1 + 15, x1 + 78, y1 + 27), fill="#22c55e")
    y = y1 + 66
    for value, color in lines:
        text(draw, (x1 + 26, y), value, color, FONT_MONO)
        y += 31


def metric_card(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, score: str, color: str) -> None:
    rounded(draw, (x, y, x + 230, y + 112), "#ffffff", "#d7deea")
    text(draw, (x + 22, y + 20), label, "#526070", FONT_SM)
    text(draw, (x + 22, y + 54), score, color, FONT_BOLD)


def frame_intro() -> Image.Image:
    image = base(
        "Debug RAG and agent traces",
        "Groundedness, hallucination risk, citation coverage, missing facts, and chunk usefulness.",
    )
    draw = ImageDraw.Draw(image)
    rounded(draw, (76, 240, 1204, 596), "#ffffff", "#d7deea")
    text(draw, (128, 306), "Input", "#526070", FONT_BOLD)
    text(draw, (128, 354), "Trace JSON: query + chunks + answer + optional reference", "#111827", FONT)
    text(draw, (128, 426), "Output", "#526070", FONT_BOLD)
    text(draw, (128, 474), "Human-readable diagnostics plus machine-readable JSON, CSV, JSONL, and HTML reports", "#111827", FONT)
    return image


def frame_good() -> Image.Image:
    image = base("Run a grounded trace", "The CLI gives a quick quality read without any API key.")
    draw = ImageDraw.Draw(image)
    terminal(
        draw,
        (70, 226, 1210, 610),
        [
            ("$ llm-trace-inspector eval examples/rag_trace_good.json", "#dbeafe"),
            ("Groundedness                 1.000", "#bbf7d0"),
            ("Hallucination risk           0.000", "#bbf7d0"),
            ("Relevance                    0.309", "#e0e7ff"),
            ("Citation/support coverage    1.000", "#bbf7d0"),
            ("Diagnostic: low hallucination risk; most useful chunk: chunk-2", "#f8fafc"),
        ],
    )
    return image


def frame_bad() -> Image.Image:
    image = base("Catch hallucinated claims", "Unsupported claims and missing facts are surfaced directly.")
    draw = ImageDraw.Draw(image)
    metric_card(draw, 76, 234, "Groundedness", "0.000", "#dc2626")
    metric_card(draw, 326, 234, "Risk", "0.830", "#dc2626")
    metric_card(draw, 576, 234, "Relevance", "0.167", "#d97706")
    metric_card(draw, 826, 234, "Citation coverage", "0.000", "#dc2626")
    rounded(draw, (76, 392, 1204, 620), "#ffffff", "#d7deea")
    text(draw, (112, 428), "Unsupported claims", "#991b1b", FONT_BOLD)
    text(draw, (112, 480), "• Active-active cross-region replication with automatic failover", "#1f2937", FONT)
    text(draw, (112, 524), "• One-click SOC 2 audit exports and HIPAA compliance reports", "#1f2937", FONT)
    return image


def frame_batch() -> Image.Image:
    image = base("Evaluate a directory of traces", "Batch mode creates artifacts for CI and release checks.")
    draw = ImageDraw.Draw(image)
    terminal(
        draw,
        (70, 220, 1210, 620),
        [
            ("$ llm-trace-inspector eval-dir examples --output-dir reports --max-risk 0.6", "#dbeafe"),
            ("agent-tool-error-001     risk 0.830   unsupported_claim, missing_retrieval", "#fecaca"),
            ("rag-bad-citations-001    risk 0.210   citation_mismatch", "#fde68a"),
            ("rag-good-001             risk 0.000   none", "#bbf7d0"),
            ("Wrote summary.json, results.jsonl, results.csv, report.html", "#f8fafc"),
        ],
    )
    return image


def frame_ui() -> Image.Image:
    image = base("Inspect results visually", "Streamlit view for claim support, failure modes, citations, and chunk rankings.")
    draw = ImageDraw.Draw(image)
    rounded(draw, (74, 220, 1206, 628), "#ffffff", "#d7deea")
    metric_card(draw, 110, 260, "Groundedness", "0.50", "#d97706")
    metric_card(draw, 360, 260, "Risk", "0.44", "#d97706")
    metric_card(draw, 610, 260, "Support", "0.50", "#d97706")
    metric_card(draw, 860, 260, "Relevance", "0.21", "#d97706")
    rounded(draw, (110, 414, 574, 576), "#f8fafc", "#d7deea")
    rounded(draw, (622, 414, 1086, 576), "#f8fafc", "#d7deea")
    text(draw, (142, 448), "Failure Modes", "#111827", FONT_BOLD)
    text(draw, (142, 500), "unsupported_claim, missing_retrieval", "#526070", FONT_SM)
    text(draw, (654, 448), "Chunk Ranking", "#111827", FONT_BOLD)
    text(draw, (654, 500), "chunk-1 useful • chunk-2 relevant_unused", "#526070", FONT_SM)
    return image


def frame_close() -> Image.Image:
    image = base("GitHub-ready LLM evaluation tool", "CLI, API, UI, examples, tests, reports, and deterministic fallback heuristics.")
    draw = ImageDraw.Draw(image)
    rounded(draw, (132, 258, 1148, 564), "#111827", "#263244")
    text(draw, (184, 326), "pip install -e .", "#dbeafe", FONT_MONO)
    text(draw, (184, 378), "llm-trace-inspector eval examples/rag_trace_hallucinated.json", "#bbf7d0", FONT_MONO)
    text(draw, (184, 430), "uvicorn llm_trace_inspector.api:app --reload", "#fde68a", FONT_MONO)
    text(draw, (184, 482), "streamlit run app/streamlit_app.py", "#fecaca", FONT_MONO)
    return image


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        frame_intro(),
        frame_good(),
        frame_bad(),
        frame_batch(),
        frame_ui(),
        frame_close(),
    ]
    expanded = []
    for frame in frames:
        expanded.extend([frame] * 10)
    expanded[0].save(
        OUT,
        save_all=True,
        append_images=expanded[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
