from pathlib import Path

try:
    import markdown
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "markdown"])
    import markdown

root = Path(__file__).resolve().parents[1] / "outputs"
md = (root / "METHODOLOGY_REPORT.md").read_text(encoding="utf-8")
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SupeRR Methodology Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:40px auto;line-height:1.5;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ccc;padding:6px 8px;text-align:left}}
th{{background:#f5f5f5}} code{{background:#f0f0f0;padding:1px 4px}}
</style></head><body>{body}</body></html>
"""
out = root / "METHODOLOGY_REPORT.html"
out.write_text(html, encoding="utf-8")
print("Wrote", out)
