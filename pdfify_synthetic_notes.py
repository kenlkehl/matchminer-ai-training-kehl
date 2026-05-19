#!/usr/bin/env python3
"""
Add synthetic dates to synthetic notes and render patient-level PDF bundles.

Default paths are resolved from this repository layout:

    python pdfify_synthetic_notes.py

Inputs:
    ../data/no_phi/all_synthetic_notes.parquet

Outputs:
    ../data/no_phi/all_synthetic_notes_with_dates.parquet
    ../data/no_phi/synthetic_patient_pdfs/*.pdf

The output parquet gets a `date` column. For each pseudo_mrn, the first note date
is deterministically assigned between 2014-01-01 and 2020-12-31. Subsequent notes
for that patient advance by a deterministic pseudo-random 0 to 180 days.

Some patient PDFs are rasterized into JPEG-backed PDF pages with scan lines,
speckles, skew, and smudges so OCR workflows see fax-like documents. The rest are
text PDFs with lighter vector noise.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
import tempfile
import textwrap
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable


LETTER_WIDTH_PT = 612.0
LETTER_HEIGHT_PT = 792.0


@dataclass
class PatientDateState:
    note_count: int
    current_date: date


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def stable_u64(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big", signed=False)


def deterministic_fraction(*parts: object) -> float:
    return stable_u64(*parts) / float(2**64)


def patient_key(value: object) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float) and math.isnan(value):
        return "missing"
    return str(value)


def patient_slug(key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._-")
    return (slug or "missing")[:80]


def patient_digest(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()[:12]


def patient_pdf_path(pdf_dir: Path, key: str) -> Path:
    return pdf_dir / f"pseudo_mrn_{patient_slug(key)}_{patient_digest(key)}.pdf"


def start_date_for_patient(
    key: str,
    seed: int,
    min_start: date,
    max_start: date,
) -> date:
    span_days = (max_start - min_start).days
    if span_days < 0:
        raise ValueError("start date min must be on or before start date max")
    offset = stable_u64(seed, key, "start_date") % (span_days + 1)
    return min_start + timedelta(days=offset)


def next_note_date(
    key: str,
    states: dict[str, PatientDateState],
    seed: int,
    min_start: date,
    max_start: date,
    max_increment_days: int,
) -> date:
    state = states.get(key)
    if state is None:
        first_date = start_date_for_patient(key, seed, min_start, max_start)
        states[key] = PatientDateState(note_count=1, current_date=first_date)
        return first_date

    increment = stable_u64(seed, key, state.note_count, "increment") % (
        max_increment_days + 1
    )
    state.current_date = state.current_date + timedelta(days=increment)
    state.note_count += 1
    return state.current_date


def clean_note_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n")


class PatientSpooler:
    def __init__(self, spool_dir: Path, max_open_files: int = 128):
        self.spool_dir = spool_dir
        self.max_open_files = max_open_files
        self._open_files: OrderedDict[str, BinaryIO] = OrderedDict()
        self._keys_by_digest: dict[str, str] = {}

    def path_for_key(self, key: str) -> Path:
        digest = patient_digest(key)
        old_key = self._keys_by_digest.get(digest)
        if old_key is not None and old_key != key:
            raise RuntimeError(
                f"Patient digest collision between {old_key!r} and {key!r}"
            )
        self._keys_by_digest[digest] = key
        return self.spool_dir / f"{digest}.txt"

    def append_document(self, key: str, note_date: date, note_text: str) -> None:
        path = self.path_for_key(key)
        handle = self._open_files.get(str(path))
        if handle is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("ab")
            self._open_files[str(path)] = handle
            if len(self._open_files) > self.max_open_files:
                _, old_handle = self._open_files.popitem(last=False)
                old_handle.close()
        else:
            self._open_files.move_to_end(str(path))

        payload = (
            "\n\n"
            + "=" * 78
            + f"\nDocument Date: {note_date.isoformat()}\n"
            + f"Pseudo MRN: {key}\n"
            + "-" * 78
            + "\n"
            + note_text.strip()
            + "\n"
        )
        handle.write(payload.encode("utf-8", errors="replace"))

    def close(self) -> None:
        while self._open_files:
            _, handle = self._open_files.popitem(last=False)
            handle.close()

    def patient_items(self) -> list[tuple[str, Path]]:
        return [
            (key, self.spool_dir / f"{digest}.txt")
            for digest, key in sorted(self._keys_by_digest.items())
        ]


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing pyarrow. Install the project dependencies, or run with an "
            "environment that includes pyarrow."
        ) from exc
    return pa, pq


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def add_dates_and_spool(
    args: argparse.Namespace,
    spooler: PatientSpooler,
) -> tuple[int, int]:
    pa, pq = require_pyarrow()
    input_path = Path(args.input_parquet)
    output_path = Path(args.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("input parquet and output parquet must be different files")

    parquet_file = pq.ParquetFile(input_path)
    schema_names = parquet_file.schema_arrow.names
    for required_col in (args.patient_id_col, args.text_col):
        if required_col not in schema_names:
            raise ValueError(
                f"Column {required_col!r} not found in {input_path}. "
                f"Available columns: {schema_names}"
            )

    writer = None
    states: dict[str, PatientDateState] = {}
    rows_written = 0
    min_start = parse_iso_date(args.min_start_date)
    max_start = parse_iso_date(args.max_start_date)

    try:
        for batch in parquet_file.iter_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([batch])
            ids = table[args.patient_id_col].to_pylist()
            texts = table[args.text_col].to_pylist()
            dates: list[date] = []

            for raw_patient_id, raw_text in zip(ids, texts):
                key = patient_key(raw_patient_id)
                note_date = next_note_date(
                    key=key,
                    states=states,
                    seed=args.seed,
                    min_start=min_start,
                    max_start=max_start,
                    max_increment_days=args.max_increment_days,
                )
                dates.append(note_date)
                spooler.append_document(key, note_date, clean_note_text(raw_text))

            date_array = pa.array(dates, type=pa.date32())
            if args.date_col in table.column_names:
                col_idx = table.schema.get_field_index(args.date_col)
                table = table.set_column(col_idx, args.date_col, date_array)
            else:
                table = table.append_column(args.date_col, date_array)

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression=args.compression,
                    use_dictionary=True,
                )
            writer.write_table(table)

            rows_written += table.num_rows
            if rows_written % args.progress_rows < table.num_rows:
                print(
                    f"Processed {rows_written:,} rows across "
                    f"{len(states):,} patients...",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        spooler.close()

    return rows_written, len(states)


def pdf_escape(text: str) -> str:
    text = "".join(ch if ch >= " " or ch == "\t" else " " for ch in text)
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_pdf_text(text: str, chars_per_line: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        raw_line = raw_line.expandtabs(4).rstrip()
        if not raw_line:
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            raw_line,
            width=chars_per_line,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return lines


def paginate_lines(lines: list[str], lines_per_page: int) -> list[list[str]]:
    if not lines:
        return [[""]]
    pages = []
    for start in range(0, len(lines), lines_per_page):
        pages.append(lines[start : start + lines_per_page])
    return pages


def stream_object(data: bytes) -> bytes:
    return (
        f"<< /Length {len(data)} >>\nstream\n".encode("ascii")
        + data
        + b"\nendstream"
    )


class SimplePDF:
    def __init__(self):
        self.pages: list[dict[str, object]] = []

    def add_text_page(self, lines: list[str], rng: random.Random, noisy: bool) -> None:
        content = build_text_page_content(lines, rng, noisy)
        self.pages.append({"kind": "text", "content": content})

    def add_jpeg_page(
        self, jpeg_bytes: bytes, pixel_width: int, pixel_height: int
    ) -> None:
        self.pages.append(
            {
                "kind": "jpeg",
                "jpeg": jpeg_bytes,
                "pixel_width": pixel_width,
                "pixel_height": pixel_height,
            }
        )

    def write(self, path: Path) -> None:
        objects: dict[int, bytes] = {}
        catalog_id = 1
        pages_id = 2
        font_id = 3
        next_id = 4
        page_ids: list[int] = []

        objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

        for idx, page in enumerate(self.pages, start=1):
            page_id = next_id
            next_id += 1
            page_ids.append(page_id)

            if page["kind"] == "text":
                content_id = next_id
                next_id += 1
                objects[content_id] = stream_object(
                    page["content"]  # type: ignore[arg-type]
                )
                objects[page_id] = (
                    b"<< /Type /Page "
                    + f"/Parent {pages_id} 0 R ".encode("ascii")
                    + b"/MediaBox [0 0 612 792] "
                    + b"/Resources << /Font << /F1 "
                    + f"{font_id} 0 R".encode("ascii")
                    + b" >> >> "
                    + f"/Contents {content_id} 0 R >>".encode("ascii")
                )
                continue

            image_id = next_id
            content_id = next_id + 1
            next_id += 2
            image_name = f"Im{idx}"
            jpeg_bytes = page["jpeg"]  # type: ignore[assignment]
            assert isinstance(jpeg_bytes, bytes)
            objects[image_id] = (
                b"<< /Type /XObject /Subtype /Image "
                + f"/Width {page['pixel_width']} /Height {page['pixel_height']} ".encode(
                    "ascii"
                )
                + b"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
                + b"/Filter /DCTDecode "
                + f"/Length {len(jpeg_bytes)} >>\nstream\n".encode("ascii")
                + jpeg_bytes
                + b"\nendstream"
            )
            draw_image = (
                "q\n"
                f"{LETTER_WIDTH_PT:.2f} 0 0 {LETTER_HEIGHT_PT:.2f} 0 0 cm\n"
                f"/{image_name} Do\n"
                "Q\n"
            ).encode("ascii")
            objects[content_id] = stream_object(draw_image)
            objects[page_id] = (
                b"<< /Type /Page "
                + f"/Parent {pages_id} 0 R ".encode("ascii")
                + b"/MediaBox [0 0 612 792] "
                + b"/Resources << /XObject << /"
                + image_name.encode("ascii")
                + b" "
                + f"{image_id} 0 R".encode("ascii")
                + b" >> >> "
                + f"/Contents {content_id} 0 R >>".encode("ascii")
            )

        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        objects[pages_id] = (
            f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
        )
        objects[catalog_id] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode(
            "ascii"
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
            offsets = [0] * (max(objects) + 1)
            for obj_id in range(1, max(objects) + 1):
                offsets[obj_id] = handle.tell()
                handle.write(f"{obj_id} 0 obj\n".encode("ascii"))
                handle.write(objects[obj_id])
                handle.write(b"\nendobj\n")

            xref_start = handle.tell()
            handle.write(f"xref\n0 {len(offsets)}\n".encode("ascii"))
            handle.write(b"0000000000 65535 f \n")
            for offset in offsets[1:]:
                handle.write(f"{offset:010d} 00000 n \n".encode("ascii"))
            handle.write(
                b"trailer\n"
                + f"<< /Size {len(offsets)} /Root {catalog_id} 0 R >>\n".encode("ascii")
                + b"startxref\n"
                + f"{xref_start}\n".encode("ascii")
                + b"%%EOF\n"
            )


def build_text_page_content(
    lines: list[str], rng: random.Random, noisy: bool
) -> bytes:
    ops: list[str] = []
    if noisy:
        ops.extend(vector_noise_ops(rng))

    ops.append("BT")
    ops.append("/F1 9.5 Tf")
    ops.append("11.8 TL")
    ops.append("50 748 Td")
    for line in lines:
        ops.append(f"({pdf_escape(line[:240])}) Tj")
        ops.append("T*")
    ops.append("ET")

    if noisy and rng.random() < 0.35:
        ops.extend(vector_noise_ops(rng, foreground=True))
    return ("\n".join(ops) + "\n").encode("latin-1", errors="replace")


def vector_noise_ops(rng: random.Random, foreground: bool = False) -> list[str]:
    ops: list[str] = []
    line_count = rng.randint(4, 12) if foreground else rng.randint(8, 22)
    rect_count = rng.randint(2, 6) if foreground else rng.randint(5, 14)

    for _ in range(line_count):
        gray = rng.uniform(0.55, 0.92) if foreground else rng.uniform(0.82, 0.97)
        width = rng.uniform(0.2, 1.3)
        x1 = rng.uniform(0, LETTER_WIDTH_PT)
        x2 = rng.uniform(0, LETTER_WIDTH_PT)
        y = rng.uniform(20, LETTER_HEIGHT_PT - 20)
        y2 = y + rng.uniform(-2, 2)
        ops.append(
            f"{gray:.3f} G {width:.2f} w "
            f"{x1:.1f} {y:.1f} m {x2:.1f} {y2:.1f} l S"
        )

    for _ in range(rect_count):
        gray = rng.uniform(0.80, 0.96) if not foreground else rng.uniform(0.65, 0.88)
        x = rng.uniform(0, LETTER_WIDTH_PT - 20)
        y = rng.uniform(0, LETTER_HEIGHT_PT - 20)
        w = rng.uniform(2, 28)
        h = rng.uniform(1, 10)
        ops.append(f"{gray:.3f} g {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f")
    return ops


def load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
    except ModuleNotFoundError:
        return None
    return Image, ImageDraw, ImageFilter, ImageFont


def find_monospace_font(image_font_module, font_size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return image_font_module.truetype(candidate, font_size)
    return image_font_module.load_default()


def build_fax_page_image(
    lines: list[str],
    rng: random.Random,
    dpi: int,
    page_number: int,
):
    pillow = load_pillow()
    if pillow is None:
        raise RuntimeError("Pillow is required for image-style PDF pages")

    Image, ImageDraw, ImageFilter, ImageFont = pillow
    width = int(8.5 * dpi)
    height = int(11.0 * dpi)
    bg = rng.randint(246, 255)
    image = Image.new("RGB", (width, height), (bg, bg, bg))
    draw = ImageDraw.Draw(image)
    font = find_monospace_font(ImageFont, max(10, int(dpi * 0.085)))
    header_font = find_monospace_font(ImageFont, max(9, int(dpi * 0.075)))

    margin_x = int(dpi * rng.uniform(0.45, 0.62))
    y = int(dpi * rng.uniform(0.42, 0.58))
    line_height = int(dpi * 0.135)

    header = f"FAX TRANSMISSION  PAGE {page_number:03d}"
    if rng.random() < 0.70:
        draw.text(
            (margin_x, int(dpi * 0.18)),
            header,
            fill=(95, 95, 95),
            font=header_font,
        )

    for line in lines:
        draw.text(
            (margin_x + rng.randint(-2, 2), y),
            line[:180],
            fill=(rng.randint(20, 55),) * 3,
            font=font,
        )
        y += line_height + rng.choice([-1, 0, 0, 1])

    # Scan lines and streaks.
    for yy in range(0, height, rng.randint(7, 13)):
        gray = rng.randint(205, 240)
        draw.line((0, yy, width, yy + rng.choice([0, 0, 1])), fill=(gray, gray, gray))

    for _ in range(rng.randint(3, 9)):
        x = rng.randint(0, width - 1)
        gray = rng.randint(175, 225)
        draw.rectangle(
            (x, 0, min(width - 1, x + rng.randint(1, 5)), height),
            fill=(gray, gray, gray),
        )

    # Speckles, toner crud, and small smudges.
    for _ in range(int(width * height * rng.uniform(0.00035, 0.00085))):
        x = rng.randrange(width)
        y = rng.randrange(height)
        shade = rng.choice([0, 25, 55, 200, 225])
        if rng.random() < 0.12:
            draw.rectangle(
                (x, y, min(width - 1, x + rng.randint(1, 3)), min(height - 1, y + 1)),
                fill=(shade, shade, shade),
            )
        else:
            draw.point((x, y), fill=(shade, shade, shade))

    for _ in range(rng.randint(6, 18)):
        x = rng.randint(0, width - int(dpi * 0.2))
        y = rng.randint(0, height - int(dpi * 0.1))
        w = rng.randint(4, int(dpi * 0.45))
        h = rng.randint(2, int(dpi * 0.18))
        shade = rng.randint(130, 215)
        draw.ellipse((x, y, x + w, y + h), fill=(shade, shade, shade))

    if rng.random() < 0.85:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.45)))

    if rng.random() < 0.80:
        angle = rng.uniform(-0.55, 0.55)
        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )

    # Fax/copy quality: downsample and re-expand to soften edges.
    if rng.random() < 0.75:
        scale = rng.uniform(0.72, 0.88)
        small = image.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.BILINEAR,
        )
        image = small.resize((width, height), Image.Resampling.BILINEAR)

    bio = BytesIO()
    image.save(
        bio,
        format="JPEG",
        quality=rng.randint(42, 58),
        optimize=True,
        progressive=False,
    )
    return bio.getvalue(), width, height


def render_patient_pdf(
    patient_id: str,
    spool_path: Path,
    pdf_path: Path,
    args: argparse.Namespace,
    use_image_style: bool,
) -> None:
    text = spool_path.read_text(encoding="utf-8", errors="replace")
    rng = random.Random(stable_u64(args.seed, patient_id, "pdf"))
    pdf = SimplePDF()

    if use_image_style:
        lines = wrap_pdf_text(text, args.image_chars_per_line)
        pages = paginate_lines(lines, args.image_lines_per_page)
        for page_number, page_lines in enumerate(pages, start=1):
            jpeg, width, height = build_fax_page_image(
                page_lines, rng, args.image_dpi, page_number
            )
            pdf.add_jpeg_page(jpeg, width, height)
    else:
        lines = wrap_pdf_text(text, args.vector_chars_per_line)
        pages = paginate_lines(lines, args.vector_lines_per_page)
        for page_idx, page_lines in enumerate(pages):
            noisy = deterministic_fraction(
                args.seed, patient_id, page_idx, "vector_noise"
            ) < args.vector_noise_page_rate
            pdf.add_text_page(page_lines, rng, noisy=noisy)

    pdf.write(pdf_path)


def render_patient_pdfs(
    patient_items: Iterable[tuple[str, Path]],
    pdf_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, int]:
    pillow_available = load_pillow() is not None
    if args.image_style_patient_rate > 0 and not pillow_available:
        print(
            "Pillow is not available; falling back to text PDFs with vector noise only.",
            flush=True,
        )

    pdf_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    image_style_count = 0
    for patient_id, spool_path in patient_items:
        use_image_style = (
            pillow_available
            and deterministic_fraction(args.seed, patient_id, "image_style_patient")
            < args.image_style_patient_rate
        )
        if use_image_style:
            image_style_count += 1
        render_patient_pdf(
            patient_id=patient_id,
            spool_path=spool_path,
            pdf_path=patient_pdf_path(pdf_dir, patient_id),
            args=args,
            use_image_style=use_image_style,
        )
        rendered += 1
        if rendered % args.progress_pdfs == 0:
            print(
                f"Rendered {rendered:,} patient PDFs "
                f"({image_style_count:,} image-style)...",
                flush=True,
            )
    return rendered, image_style_count


def clean_pdf_dir(pdf_dir: Path) -> None:
    if not pdf_dir.exists():
        return
    for path in pdf_dir.glob("*.pdf"):
        path.unlink()


def build_parser() -> argparse.ArgumentParser:
    default_data_dir = repo_root() / "data" / "no_phi"
    parser = argparse.ArgumentParser(
        description=(
            "Assign synthetic note dates and create patient-level synthetic note PDFs."
        )
    )
    parser.add_argument(
        "--input_parquet",
        default=str(default_data_dir / "all_synthetic_notes.parquet"),
        help="Input parquet with one synthetic note per row.",
    )
    parser.add_argument(
        "--output_parquet",
        default=str(default_data_dir / "all_synthetic_notes_with_dates.parquet"),
        help="Output parquet with an added date column.",
    )
    parser.add_argument(
        "--pdf_dir",
        default=str(default_data_dir / "synthetic_patient_pdfs"),
        help="Output directory for patient-level PDFs.",
    )
    parser.add_argument("--patient_id_col", default="pseudo_mrn")
    parser.add_argument("--text_col", default="synthetic_note")
    parser.add_argument("--date_col", default="date")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--min_start_date", default="2014-01-01")
    parser.add_argument("--max_start_date", default="2020-12-31")
    parser.add_argument("--max_increment_days", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--progress_rows", type=int, default=250_000)
    parser.add_argument("--progress_pdfs", type=int, default=100)
    parser.add_argument(
        "--image_style_patient_rate",
        type=float,
        default=0.75,
        help="Fraction of patient PDFs rendered as raster fax-like image PDFs.",
    )
    parser.add_argument("--image_dpi", type=int, default=140)
    parser.add_argument("--image_chars_per_line", type=int, default=92)
    parser.add_argument("--image_lines_per_page", type=int, default=62)
    parser.add_argument("--vector_chars_per_line", type=int, default=96)
    parser.add_argument("--vector_lines_per_page", type=int, default=58)
    parser.add_argument(
        "--vector_noise_page_rate",
        type=float,
        default=0.35,
        help="Fraction of text-PDF pages that receive light vector scan artifacts.",
    )
    parser.add_argument(
        "--max_open_spool_files",
        type=int,
        default=128,
        help="LRU cap for temporary per-patient text spool files.",
    )
    parser.add_argument(
        "--spool_dir",
        default=None,
        help="Optional empty directory for temporary patient note bundles.",
    )
    parser.add_argument(
        "--clean_pdf_dir",
        action="store_true",
        help="Delete existing *.pdf files in pdf_dir before writing new PDFs.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_increment_days < 0:
        raise ValueError("--max_increment_days must be non-negative")
    for rate_name in ("image_style_patient_rate", "vector_noise_page_rate"):
        rate = getattr(args, rate_name)
        if not 0 <= rate <= 1:
            raise ValueError(f"--{rate_name} must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.image_dpi < 72:
        raise ValueError("--image_dpi should be at least 72")


def prepare_spool_dir(args: argparse.Namespace, pdf_dir: Path):
    if args.spool_dir is None:
        return tempfile.TemporaryDirectory(
            prefix="synthetic_pdf_spool_",
            dir=pdf_dir.parent,
        )

    spool_dir = Path(args.spool_dir)
    spool_dir.mkdir(parents=True, exist_ok=True)
    if any(spool_dir.iterdir()):
        raise ValueError(f"spool_dir must be empty: {spool_dir}")
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_pdf_dir:
        clean_pdf_dir(pdf_dir)

    temp_spool = prepare_spool_dir(args, pdf_dir)
    try:
        spool_dir = (
            Path(temp_spool.name) if temp_spool is not None else Path(args.spool_dir)
        )
        spooler = PatientSpooler(spool_dir, max_open_files=args.max_open_spool_files)

        print(f"Reading: {args.input_parquet}", flush=True)
        print(f"Writing dated parquet: {args.output_parquet}", flush=True)
        rows, patients = add_dates_and_spool(args, spooler)
        print(
            f"Wrote {rows:,} dated rows across {patients:,} patients. "
            "Rendering patient PDFs...",
            flush=True,
        )

        rendered, image_style = render_patient_pdfs(
            spooler.patient_items(),
            pdf_dir=pdf_dir,
            args=args,
        )
        print(
            f"Rendered {rendered:,} PDFs to {pdf_dir} "
            f"({image_style:,} image-style).",
            flush=True,
        )
    finally:
        if temp_spool is not None:
            temp_spool.cleanup()


if __name__ == "__main__":
    main()
