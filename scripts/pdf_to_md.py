"""Convert PDFs in holdings/ into LLM-readable markdown in data/research/.

Pipeline per PDF:
  1. pymupdf4llm extracts body text + native tables into markdown.
  2. pymupdf extracts every embedded image into data/research/images/<stem>/.
  3. Claude Opus 4.7 vision describes each image, focused on quantitative content
     (axes, price targets, peer comparisons, tables-rendered-as-image transcribed).
  4. Image descriptions are appended to the markdown as per-page sections so any
     downstream agent (research.py, multi_agent.py) reads a single .md file and
     gets both text and chart content.

Output stays under data/research/, which is gitignored — analyst content tied
to user holdings never leaves the machine.

Usage:
  python scripts/pdf_to_md.py                       # convert every analyst PDF
  python scripts/pdf_to_md.py --pattern "bofa_*.pdf"
  python scripts/pdf_to_md.py --pattern "bofa_intc_1.pdf" --no-vision
  python scripts/pdf_to_md.py --force               # re-run even if .md exists

Personal files in holdings/ that this script will SKIP by default:
  Statement*.pdf  (Fidelity statements — parsed separately by portfolio.py)
  *.png           (broker screenshots — not PDFs anyway)
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import pymupdf
import pymupdf4llm
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
HOLDINGS_DIR = ROOT / "holdings"
RESEARCH_DIR = ROOT / "data" / "research"
IMAGES_ROOT = RESEARCH_DIR / "images"

load_dotenv(ROOT / ".env")

VISION_MODEL = "claude-opus-4-7"
MIN_IMAGE_BYTES = 4 * 1024
MAX_IMAGE_BYTES = 5 * 1024 * 1024

SKIP_PREFIXES = ("statement",)

VISION_PROMPT = (
    "You are reading an image extracted from a sell-side equity research PDF "
    "(BofA, Goldman, Morgan Stanley, JPM, etc.). Describe what's in this image "
    "in 4-10 sentences, focusing on QUANTITATIVE content useful to an "
    "investment-research agent:\n"
    "  - If it's a chart: name the axes, the time range, and the trajectory. "
    "Quote any visible price targets, percentage moves, or trend annotations.\n"
    "  - If it's a table rendered as an image: transcribe the rows and columns "
    "as plain text (use simple markdown if helpful).\n"
    "  - If it's a heatmap or sector grid: describe the comparison axes and the "
    "highest / lowest cells with their values.\n"
    "  - If it's a logo, headshot, or decorative element: return the single line "
    "`(decorative — no analytical content)`.\n"
    "Return ONLY the description text. No preamble like 'This image shows...'."
)


def _get_client():
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def describe_image(client, img_bytes: bytes, media_type: str) -> str:
    b64 = base64.standard_b64encode(img_bytes).decode("ascii")
    resp = client.messages.create(
        model=VISION_MODEL,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    )
    for block in resp.content:
        if block.type == "text" and block.text.strip():
            return block.text.strip()
    return "(no description generated)"


def extract_images(pdf_path: Path, image_dir: Path) -> list[dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    doc = pymupdf.open(pdf_path)
    seen_xrefs: set[int] = set()
    try:
        for page_num, page in enumerate(doc, start=1):
            for img_idx, img_info in enumerate(page.get_images(full=True), start=1):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    base_img = doc.extract_image(xref)
                except Exception:
                    continue
                img_bytes = base_img.get("image")
                ext = base_img.get("ext", "png")
                if not img_bytes:
                    continue
                size = len(img_bytes)
                if size < MIN_IMAGE_BYTES or size > MAX_IMAGE_BYTES:
                    continue
                media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
                fname = f"page{page_num:03d}_img{img_idx}.{ext}"
                fpath = image_dir / fname
                fpath.write_bytes(img_bytes)
                results.append({
                    "page": page_num,
                    "path": fpath,
                    "bytes": img_bytes,
                    "media_type": media_type,
                    "size_kb": round(size / 1024, 1),
                })
    finally:
        doc.close()
    return results


def convert_pdf(pdf_path: Path, *, describe: bool, force: bool) -> Path | None:
    stem = pdf_path.stem
    out_md = RESEARCH_DIR / f"{stem}.md"
    img_dir = IMAGES_ROOT / stem

    if out_md.exists() and not force:
        print(f"  SKIP {pdf_path.name} (already converted — pass --force to redo)")
        return out_md

    print(f"  [1/3] Extracting text + tables...")
    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    print(f"  [2/3] Extracting embedded images...")
    images = extract_images(pdf_path, img_dir)
    print(f"        Found {len(images)} image(s) in {pdf_path.name}")

    image_section_lines: list[str] = []
    if images:
        image_section_lines.extend(["", "---", "", "## Embedded Images", ""])
        if describe:
            print(f"  [3/3] Describing {len(images)} image(s) via {VISION_MODEL} vision...")
            client = _get_client()
            for i, img in enumerate(images, start=1):
                print(f"        {i}/{len(images)} page {img['page']} ({img['size_kb']} KB)... ", end="", flush=True)
                t0 = time.time()
                try:
                    desc = describe_image(client, img["bytes"], img["media_type"])
                except Exception as exc:
                    desc = f"(vision failed: {exc})"
                rel = img["path"].relative_to(RESEARCH_DIR).as_posix()
                image_section_lines.append(f"### Page {img['page']} — `{img['path'].name}`")
                image_section_lines.append(f"![{img['path'].name}]({rel})")
                image_section_lines.append("")
                image_section_lines.append(f"**Vision description**: {desc}")
                image_section_lines.append("")
                print(f"done ({time.time() - t0:.1f}s)")
        else:
            print(f"  [3/3] Skipping vision descriptions (--no-vision)")
            for img in images:
                rel = img["path"].relative_to(RESEARCH_DIR).as_posix()
                image_section_lines.append(f"- Page {img['page']}: ![{img['path'].name}]({rel})")
            image_section_lines.append("")

    final = md_text.rstrip() + "\n" + "\n".join(image_section_lines)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(final, encoding="utf-8")
    print(f"        -> {out_md.relative_to(ROOT).as_posix()}")
    return out_md


def is_analyst_pdf(pdf: Path) -> bool:
    name = pdf.name.lower()
    return not any(name.startswith(skip) for skip in SKIP_PREFIXES)


def main():
    parser = argparse.ArgumentParser(description="Convert holdings PDFs into research markdown.")
    parser.add_argument("--pattern", default="*.pdf", help='Glob pattern in holdings/ (default "*.pdf")')
    parser.add_argument("--no-vision", action="store_true", help="Skip Claude vision image descriptions")
    parser.add_argument("--force", action="store_true", help="Re-convert even if output .md already exists")
    args = parser.parse_args()

    if not HOLDINGS_DIR.exists():
        print(f"holdings/ not found at {HOLDINGS_DIR}", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(HOLDINGS_DIR.glob(args.pattern))
    pdfs = [p for p in pdfs if is_analyst_pdf(p)]
    if not pdfs:
        print(f"No analyst PDFs matched {args.pattern} in {HOLDINGS_DIR}")
        return

    print(f"Found {len(pdfs)} analyst PDF(s) to process")
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for pdf in pdfs:
        print(f"\n>>> {pdf.name}")
        try:
            convert_pdf(pdf, describe=not args.no_vision, force=args.force)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append((pdf.name, str(exc)))

    print()
    if failures:
        print(f"Done with {len(failures)} failure(s):")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
