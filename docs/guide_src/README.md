Source of `docs/report_guide.pdf`. Edit `report_guide.html`, then render it (US Letter, page size from its CSS):

```bash
pip install playwright && playwright install chromium
python - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto("file://" + __import__("os").path.abspath("docs/guide_src/report_guide.html"))
    pg.pdf(path="docs/report_guide.pdf", print_background=True, prefer_css_page_size=True)
    b.close()
PY
```
