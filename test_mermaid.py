import subprocess
import os

html = """<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad: true});</script>
</head>
<body>
<h1>Mermaid Test</h1>
<div class="mermaid">
flowchart LR
    A[Start] --> B[End]
</div>
</body>
</html>"""

with open("test_mermaid.html", "w", encoding="utf-8") as f:
    f.write(html)

exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
pdf = os.path.abspath("test_mermaid.pdf")
html_abs = os.path.abspath("test_mermaid.html")
cmd = [exe, "--headless", "--virtual-time-budget=5000", "--run-all-compositor-stages-before-draw", f"--print-to-pdf={pdf}", f"file:///{html_abs}"]
res = subprocess.run(cmd, capture_output=True, text=True)
print("PDF created:", os.path.exists(pdf))
if os.path.exists(pdf):
    print("Size:", os.path.getsize(pdf))
    with open(pdf, "rb") as f:
        data = f.read()
    print("Contains Start:", b"Start" in data)
    os.remove(pdf)
if os.path.exists("test_mermaid.html"):
    os.remove("test_mermaid.html")
