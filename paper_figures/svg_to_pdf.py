"""Convert fig0_architecture_compact.svg to PDF using pycairo + svglib."""
import sys
from pathlib import Path

SVG_IN  = Path(r"E:\PycharmProjects\DocVerify\paper_figures\output\fig0_architecture_compact.svg")
PDF_OUT = Path(r"E:\PycharmProjects\DocVerify\paper\prltemplate\figures\fig0_architecture_compact.pdf")

# --- svglib + ReportLab (rlPyCairo backend used only if available) ---
try:
    try:
        import rlPyCairo  # noqa — optional Cairo backend for nicer rendering
    except Exception:
        pass
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    drawing = svg2rlg(str(SVG_IN))
    if drawing is None:
        raise RuntimeError("svg2rlg returned None")
    print(f"Drawing: {drawing.width:.1f} x {drawing.height:.1f} pt")
    renderPDF.drawToFile(drawing, str(PDF_OUT))
    print(f"Saved (svglib): {PDF_OUT}")
    sys.exit(0)
except Exception as e:
    print(f"svglib failed: {e}")

# --- Fallback: pycairo SVGSurface → PDFSurface ---
try:
    import cairo
    import xml.etree.ElementTree as ET

    # Get SVG dimensions
    tree = ET.parse(str(SVG_IN))
    root = tree.getroot()
    vb = root.get("viewBox", "0 0 386 170").split()
    W_pt, H_pt = float(vb[2]), float(vb[3])
    print(f"SVG viewBox: {W_pt:.1f} x {H_pt:.1f} pt")

    # Create PDF surface and render SVG onto it
    pdf_surface = cairo.PDFSurface(str(PDF_OUT), W_pt, H_pt)
    ctx = cairo.Context(pdf_surface)

    svg_surface = cairo.SVGSurface(str(SVG_IN), W_pt, H_pt)
    # Record the SVG surface to a recording surface, then paint it
    rec = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    ctx2 = cairo.Context(rec)
    ctx2.set_source_surface(svg_surface, 0, 0)
    ctx2.paint()

    ctx.set_source_surface(rec, 0, 0)
    ctx.paint()
    pdf_surface.finish()
    print(f"Saved (pycairo): {PDF_OUT}")
    sys.exit(0)
except Exception as e:
    print(f"pycairo fallback failed: {e}")

print("All methods failed.")
sys.exit(1)
