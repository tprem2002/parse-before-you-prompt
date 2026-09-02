"""Warm every enabled local model used by the Docling standard pipeline."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.docling_converter import (  # noqa: E402
    convert_docling_standard,
    docling_standard_configuration,
)


def _write_synthetic_pdf(path: Path) -> None:
    """Create a tiny local document that exercises visual and enriched stages."""

    image = Image.new("RGB", (480, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 460, 200), outline="navy", width=4)
    draw.line((60, 170, 180, 90, 300, 120, 420, 45), fill="royalblue", width=6)
    draw.text((30, 28), "Synthetic readiness chart", fill="black")
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setTitle("Local Docling model prefetch")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(54, 742, "Local Docling Model Prefetch")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(54, 720, "This synthetic PDF never leaves the local machine.")
    pdf.drawImage(ImageReader(image_bytes), 54, 430, width=480, height=220)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(54, 414, "Figure 1. Synthetic readiness trend for model initialization.")
    pdf.setFont("Courier", 10)
    pdf.drawString(54, 375, "def readiness(value: float) -> bool:")
    pdf.drawString(72, 359, "return value >= 0.92")
    pdf.setFont("Times-Italic", 14)
    pdf.drawString(54, 325, "R = successful checks / total checks")
    pdf.save()


def main() -> int:
    """Run the configured standard converter once so model caches are populated."""

    settings = get_settings()
    with tempfile.TemporaryDirectory(prefix="pbtp-docling-prefetch-") as temp_name:
        pdf_path = Path(temp_name) / "prefetch.pdf"
        _write_synthetic_pdf(pdf_path)
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        configuration = docling_standard_configuration(digest, settings)
        converter = configuration["converter_configuration"]
        print(f"Device: {converter['accelerator']['device']}")
        print(f"CPU threads: {converter['accelerator']['num_threads']}")
        print(f"HF cache: {converter['local_cache']['hf_home']}")
        print(
            "Presets: "
            f"layout={converter['layout']['preset']}, "
            f"picture_classifier={converter['picture_classification']['preset']}, "
            f"picture_description={converter['picture_description']['preset']}, "
            f"code_formula={converter['code_formula']['preset']}"
        )
        print("Remote services: disabled")
        print("External plugins: disabled")
        conversion = convert_docling_standard(pdf_path, settings)
        print(f"Completion status: {conversion.result.status.value}")
        print(f"Conversion duration: {conversion.duration_ms} ms")
        print(f"Warnings: {len(conversion.warnings)}")
        return 0 if conversion.result.status.value in {"success", "partial_success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
