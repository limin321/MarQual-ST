"""
HTML report of one pipeline run.

Scans the figures folder and organizes every output into one HTML page, in pipeline order:

  Overview            key numbers + verdicts (technical QC, Moran's I, co-localization)
  1 Technical QC      summaries, depth maps, high-pass depth map
  2 Spatial structure Moran's I chart/tables; one collapsible block per marker set
  3 Niches and DE     one block per niche / region: mask, depth checks, top DE genes, dot plot
  4 Annotation        cell-type annotation map and bins per label
  5 Co-localization   all-pairs table, technical controls; one block per pair
  6 Gene pairs        figures/gene_pairs/ (``marqual-st gene-pairs``), if present
  7 Run parameters    every setting of the run (run_parameters.csv)
  8 Other files       anything not matched above - nothing is silently dropped

Figures: the pipeline writes publication-quality PDFs only. The report makes a preview of each PDF
in memory (PdfPreviewer: pypdfium2, PyMuPDF or poppler's pdftoppm) and embeds it - one file you
can share - with each figure linked to its PDF; no PNG files are written. Figures of older runs
(PNG) are still shown. Large CSVs show the first rows and are linked.
Nothing is recomputed - the report only reads what is in the folder.
"""
from __future__ import annotations

import base64
import datetime
import html
import io
import os
from importlib import resources
from pathlib import Path

import pandas as pd

from ._logging import get_logger

log = get_logger(__name__)

VERDICT_GOOD = ("PASS", "CO-LOCALIZED")
VERDICT_BAD = ("DEPTH_DRIVEN", "MASKED_BY_DEPTH", "NO_SPATIAL_SIGNAL", "SEGREGATED", "NOT_SIGNIFICANT",
               "LOW_MARKER_COVERAGE", "STOP")
NAN = float("nan")


def esc(x) -> str:
    return html.escape(str(x))


def fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "–"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(v)


def read_csv(folder, name) -> pd.DataFrame | None:
    p = os.path.join(folder, name)
    try:
        return pd.read_csv(p) if os.path.exists(p) else None
    except Exception:
        return None


def verdict_class(v: str) -> str:
    base = v.split(";")[0]
    return "good" if base in VERDICT_GOOD else ("bad" if base in VERDICT_BAD else "")


# ---------------------------------------------------------------------------
# file bookkeeping
# ---------------------------------------------------------------------------
class FileIndex:
    """The files of one folder; every file is claimed at most once, the rest go to "Other files"."""

    REPORT_SUFFIX = "_QCreport.html"

    def __init__(self, folder: str | Path, prefix: str = ""):
        self.folder = str(folder)
        self.prefix = prefix            # link prefix relative to the report, e.g. "gene_pairs/"
        self.all = sorted(f for f in os.listdir(self.folder)
                          if os.path.isfile(os.path.join(self.folder, f)) and not self.is_report(f))
        self.used: set[str] = set()

    @classmethod
    def is_report(cls, f: str) -> bool:
        return f.endswith(cls.REPORT_SUFFIX) or f in ("QCreport.html", "report.html")

    def get(self, name: str) -> str | None:
        """Claim one file by exact name (None if absent)."""
        if name in self.all:
            self.used.add(name)
            return name
        return None

    def stems(self, suffix: str) -> list[str]:
        """Names X for which X + suffix exists."""
        return [f[: -len(suffix)] for f in self.all if f.endswith(suffix) and len(f) > len(suffix)]

    def figure(self, stem: str):
        """Claim stem.png / .pdf / .svg / .jpg; returns (display file, all files)."""
        found = [f for f in (self.get(stem + ext) for ext in (".png", ".pdf", ".svg", ".jpg")) if f]
        return (found[0] if found else None), found

    def rest(self) -> list[str]:
        return [f for f in self.all if f not in self.used]

    def href(self, f: str) -> str:
        return esc(self.prefix + f)

    def path(self, f: str) -> str:
        return os.path.join(self.folder, f)


class ImageEmbedder:
    """
    Figure bytes for embedding. Large PNGs - mostly spatial maps, which compress poorly - are
    scaled to ``max_width`` and re-encoded as JPEG when that is clearly smaller (< ``accept`` x
    the original). The files on disk stay untouched and are linked.
    """

    MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}

    def __init__(self, max_width: int = 1400, jpeg_min_bytes: int = 60_000, quality: int = 85, accept: float = 0.85):
        self.max_width = max_width
        self.jpeg_min_bytes = jpeg_min_bytes
        self.quality = quality
        self.accept = accept

    def compact(self, path: str) -> tuple[bytes, str]:
        with open(path, "rb") as fh:
            raw = fh.read()
        return self.compact_bytes(raw, path.rsplit(".", 1)[1].lower())

    def compact_bytes(self, raw: bytes, ext: str) -> tuple[bytes, str]:
        mime = self.MIME[ext]
        if ext != "png" or len(raw) < self.jpeg_min_bytes:
            return raw, mime
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(raw))
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
            if im.width > self.max_width:
                im = im.resize((self.max_width, round(im.height * self.max_width / im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=self.quality, optimize=True)
            if buf.tell() < self.accept * len(raw):
                return buf.getvalue(), "image/jpeg"
        except Exception:                      # never fail the report over one image
            pass
        return raw, mime

    def data_uri(self, path: str) -> str:
        data, mime = self.compact(path)
        return f"data:{mime};base64," + base64.b64encode(data).decode()

    def bytes_uri(self, raw: bytes, ext: str = "png") -> str:
        data, mime = self.compact_bytes(raw, ext)
        return f"data:{mime};base64," + base64.b64encode(data).decode()


class PdfPreviewer:
    """
    PNG bytes of page 1 of a PDF figure, made in memory for the report (no file is written).
    Tries pypdfium2, then PyMuPDF, then poppler's ``pdftoppm``; returns None when none is available
    (``missing`` is then set, and the report links the PDFs instead).
    """

    def __init__(self, dpi: int = 200, max_width: int = 1400):
        self.dpi = dpi
        self.max_width = max_width
        self.backend: str | None = None
        self.missing = False

    def render(self, path: str) -> bytes | None:
        for backend in (self._pdfium, self._pymupdf, self._pdftoppm):
            try:
                data = backend(path)
            except ImportError:
                continue
            except Exception as e:                  # a broken file must not stop the report
                log.warning(f"Preview of {path} failed ({backend.__name__.strip('_')}): {e}")
                continue
            if data is not None:
                return data
        self.missing = True
        return None

    def _pdfium(self, path):
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(path)
        try:
            page = doc[0]
            scale = min(self.dpi / 72.0, self.max_width / max(page.get_width(), 1))
            buf = io.BytesIO()
            page.render(scale=scale).to_pil().save(buf, format="PNG")
        finally:
            doc.close()
        self.backend = "pypdfium2"
        return buf.getvalue()

    def _pymupdf(self, path):
        import fitz
        doc = fitz.open(path)
        try:
            page = doc[0]
            scale = min(self.dpi / 72.0, self.max_width / max(page.rect.width, 1))
            data = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")
        finally:
            doc.close()
        self.backend = "PyMuPDF"
        return data

    def _pdftoppm(self, path):
        import shutil
        import subprocess
        import tempfile
        if not shutil.which("pdftoppm"):
            return None
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "page")
            subprocess.run(["pdftoppm", "-png", "-r", str(self.dpi), "-f", "1", "-l", "1", "-singlefile", path, stem],
                           check=True, capture_output=True, timeout=120)
            with open(stem + ".png", "rb") as fh:
                data = fh.read()
        self.backend = "pdftoppm"
        return data


DEPTH_LABELS = [("n_region", "bins in region"), ("n_background", "bins in background"),
                ("median_total_counts_region", "median total counts, region"),
                ("median_total_counts_background", "median total counts, background"),
                ("median_n_genes_region", "median genes, region"),
                ("median_n_genes_background", "median genes, background"),
                ("depth_auroc", "depth AUROC (0.5 = same depth)"), ("depth_mwu_p", "Mann-Whitney p"),
                ("depth_biased", "depth-biased")]


def depth_stages(folder, name) -> dict:
    """{stage: row} from {name}_depth_check.csv (one row per stage); older runs wrote the
    after-matching row to a second file, {name}_depth_check_after_matching.csv."""
    out = {}
    d = read_csv(folder, f"{name}_depth_check.csv")
    if d is not None and len(d):
        if "stage" in d:
            for _, r in d.iterrows():
                out[str(r["stage"])] = r.to_dict()
        else:
            out["before_matching"] = d.iloc[0].to_dict()
    a = read_csv(folder, f"{name}_depth_check_after_matching.csv")
    if a is not None and len(a) and "after_matching" not in out:
        out["after_matching"] = a.iloc[0].to_dict()
    return out


# ---------------------------------------------------------------------------
# HTML blocks
# ---------------------------------------------------------------------------
class Html:
    """Small HTML building blocks bound to one folder (FileIndex)."""

    MAX_ROWS = 300      # rows loaded per table (tables scroll); the full CSV is linked

    def __init__(self, files: FileIndex, embedder: ImageEmbedder | None, self_contained: bool = True,
                 previewer: PdfPreviewer | None = None):
        self.files = files
        self.embedder = embedder or ImageEmbedder()
        self.self_contained = self_contained
        self.previewer = previewer or PdfPreviewer()

    @staticmethod
    def span(wide) -> str:
        """Grid width: True = full row, "span2" = two columns, False = one column."""
        return " wide" if wide is True else (" span2" if wide == "span2" else "")

    @staticmethod
    def cell(col: str, v) -> str:
        cls = ""
        if isinstance(v, str) and "verdict" in col.lower():
            cls = verdict_class(v)
            if ";" in v:
                cls += " warn"
        return f'<td class="{cls.strip()}">{esc(fmt(v))}</td>'

    @staticmethod
    def badge(v) -> str:
        if not isinstance(v, str) or not v:
            return ""
        return f'<span class="badge {verdict_class(v)}">{esc(v)}</span>'

    def table(self, name, title=None, note=None, transpose=False, wide=False, max_rows=None,
              visible_rows: int | None = None) -> str:
        """visible_rows: height of the table box in rows (the rest scrolls), e.g. 5 for long, minor tables."""
        f = self.files.get(name)
        if f is None:
            return ""
        try:
            df = pd.read_csv(self.files.path(f))
        except Exception as e:
            return f'<p class="muted">{esc(f)}: could not read ({esc(e)})</p>'
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": ""})
        if transpose and len(df) == 1:                       # one-row summaries read better vertically
            df = df.T.reset_index()
            df.columns = ["", "value"]
        shown = df.head(max_rows or self.MAX_ROWS)
        head = "".join(f"<th>{esc(c)}</th>" for c in shown.columns)
        body = "".join("<tr>" + "".join(self.cell(str(c), r[c]) for c in shown.columns) + "</tr>"
                       for _, r in shown.iterrows())
        more = f" · showing {len(shown)} of {len(df)} rows" if len(df) > len(shown) else ""
        cap = (f'<div class="cap"><b>{esc(title or f)}</b>'
               f'<span class="muted"> · <a href="{self.files.href(f)}">{esc(f)}</a>{more}</span></div>')
        nt = f'<p class="note">{note}</p>' if note else ""
        tw = f"tw rows{visible_rows}" if visible_rows else "tw"
        return (f'<div class="block{self.span(wide)}">{cap}{nt}<div class="{tw}"><table><thead><tr>{head}</tr></thead>'
                f'<tbody>{body}</tbody></table></div></div>')

    def depth_table(self, name, title="Depth check: niche vs. background") -> str:
        """One table: a row per measure, a column per stage (before / after depth matching)."""
        found = [f for f in (self.files.get(f"{name}_depth_check.csv"),
                             self.files.get(f"{name}_depth_check_after_matching.csv")) if f]
        if not found:
            return ""
        stages = depth_stages(self.files.folder, name)
        cols = [st for st in ("before_matching", "after_matching") if st in stages] + \
               [st for st in stages if st not in ("before_matching", "after_matching")]
        head = "<th></th>" + "".join(f"<th>{esc(c.replace('_', ' '))}</th>" for c in cols)
        body = "".join("<tr><td>" + esc(lab) + "</td>" + "".join(
            f"<td>{esc(fmt(stages[c].get(key, '')))}</td>" for c in cols) + "</tr>" for key, lab in DEPTH_LABELS)
        links = " · ".join(f'<a href="{self.files.href(f)}">{esc(f)}</a>' for f in found)
        return (f'<div class="block"><div class="cap"><b>{esc(title)}</b><span class="muted"> · {links}</span></div>'
                f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>')

    def figure(self, stem, title, note=None, wide=False) -> str:
        show, allf = self.files.figure(stem)
        if show is None:
            return ""
        links = " · ".join(f'<a href="{self.files.href(f)}">{esc(f.rsplit(".", 1)[1].upper())}</a>' for f in allf)
        nt = f'<p class="note">{note}</p>' if note else ""
        if show.endswith(".pdf"):                      # in-memory preview, always embedded (no PNG files)
            raw = self.previewer.render(self.files.path(show))
            src = self.embedder.bytes_uri(raw, "png") if raw is not None else None
        else:                                          # PNG / JPG / SVG of older runs
            src = self.embedder.data_uri(self.files.path(show)) if self.self_contained else self.files.href(show)
        if src is None:
            body = (f'<p class="muted">No preview - <a href="{self.files.href(show)}">open {esc(show)}</a> '
                    f'(install pypdfium2 to show PDF previews here: pip install pypdfium2).</p>')
        else:
            body = (f'<a href="{self.files.href(show)}" title="Open the publication-quality file">'
                    f'<img loading="lazy" src="{src}" alt="{esc(title)}"></a>')
        ar = self.aspect(self.files.path(show), src)
        st = f' style="--ar:{ar:.3f}"' if ar else ""
        return (f'<figure class="block{self.span(wide)}"{st}><div class="cap"><b>{esc(title)}</b>'
                f'<span class="muted"> · {links}</span></div>{nt}{body}</figure>')

    @staticmethod
    def aspect(path: str, src: str | None) -> float | None:
        """Width / height of a figure preview (sets its width in a figure row); None if unknown."""
        try:
            from PIL import Image
            if src and src.startswith("data:"):
                img = Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1])))
            else:
                img = Image.open(path)
            w, h = img.size
            return w / h if h else None
        except Exception:
            return None

    @staticmethod
    def frow(*figs: str) -> str:
        """Figures side by side in one row, each as wide as its aspect ratio, so all have the same height."""
        figs = tuple(f for f in figs if f)
        if len(figs) < 2:
            return figs[0] if figs else ""
        return f'<div class="frow">{"".join(figs)}</div>'

    @staticmethod
    def section(sid, title, intro, parts, subs=()) -> str:
        parts = [p for p in parts if p]
        subs = [s for s in subs if s]
        if not parts and not subs:
            return ""
        grid = f'<div class="grid">{"".join(parts)}</div>' if parts else ""
        return (f'<section id="{sid}"><h2>{esc(title)}</h2>'
                f'{f"<p class=intro>{intro}</p>" if intro else ""}{grid}{"".join(subs)}</section>')

    @staticmethod
    def sub(title, parts, summary="") -> str:
        """One collapsible block with a one-line summary (HTML)."""
        parts = [p for p in parts if p]
        if not parts:
            return ""
        return (f'<details class="sub"><summary><span class="st">{esc(title)}</span>'
                f'<span class="sm">{summary}</span></summary><div class="grid">{"".join(parts)}</div></details>')


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def first_row(df, col, key) -> dict:
    if df is None or col not in df:
        return {}
    hit = df[df[col].astype(str) == str(key)]
    return hit.iloc[0].to_dict() if len(hit) else {}


def n_sig_up(path) -> int | None:
    try:
        d = pd.read_csv(path, usecols=["logfoldchanges", "significant"])
        return int((d["significant"].astype(bool) & (d["logfoldchanges"] > 0)).sum())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
class ReportBuilder:
    """
    Build ``{sample}_QCreport.html`` from a figures folder.

    figdir         : the run's figures folder
    sample         : sample name (report file name and title)
    self_contained : embed PNGs in the HTML (default) - the page opens on its own
    """

    GENE_PAIR_DIR = "gene_pairs"
    GENE_PAIR_SUFFIX = "_spatial_colocalization.csv"
    GENE_PAIR_STATUS = "_gene_pair_status.csv"     # written for a pair that was STOPPED
    NAV = [("overview", "Overview"), ("qc", "Technical QC"), ("moran", "Spatial structure"),
           ("de", "Niches & DE"), ("annotation", "Annotation"), ("coloc", "Co-localization"),
           ("genepairs", "Gene pairs"), ("params", "Run parameters"), ("other", "Other files")]
    PARAM_FILE = "run_parameters.csv"

    def __init__(self, figdir: str | Path, sample: str | None = None, self_contained: bool = True,
                 title: str | None = None, embedder: ImageEmbedder | None = None,
                 previewer: PdfPreviewer | None = None):
        self.figdir = str(figdir)
        self.sample = sample
        self.self_contained = self_contained
        self.title = title
        self.embedder = embedder or ImageEmbedder()
        self.previewer = previewer or PdfPreviewer()

    @staticmethod
    def report_name(sample: str | None = None) -> str:
        return f"{sample}{FileIndex.REPORT_SUFFIX}" if sample else "QCreport.html"

    # ------------------------------------------------------------------ run facts
    def filter_summary(self) -> dict:
        fs = read_csv(self.figdir, "filter_summary.csv")
        return fs.iloc[0].to_dict() if fs is not None and len(fs) else {}

    def stopped(self) -> bool:
        """True when the run ended at the data-quality gate (STOP without force)."""
        fs = self.filter_summary()
        return fs.get("data_quality") == "STOP" and not truthy(fs.get("forced"))

    def niche_top_percent(self, celltype: str | None = None) -> str | None:
        """Top % of bins that defined the high-confidence niche(s), from {celltype}_niche_definition.csv."""
        names = ([f"{celltype}_niche_definition.csv"] if celltype else
                 sorted(f for f in os.listdir(self.figdir) if f.endswith("_niche_definition.csv")))
        vals = set()
        for f in names:
            d = read_csv(self.figdir, f)
            if d is not None and len(d) and "top_percent" in d:
                vals.add(float(d["top_percent"].iloc[0]))
        return " / ".join(f"{v:g}" for v in sorted(vals)) if vals else None

    # ------------------------------------------------------------------ overview
    def overview(self) -> str:
        figdir = self.figdir
        cards, alerts, rows, family = [], [], [], []
        fs = self.filter_summary()
        dq = fs.get("data_quality") if isinstance(fs.get("data_quality"), str) else None
        stopped = self.stopped()
        if dq:
            cls = {"PASS": "good", "CAUTION": "warn", "STOP": "bad"}.get(dq, "")
            cards.append(f'<div class="card"><div class="k">Data quality</div><div class="v {cls}">{esc(dq)}</div></div>')
            if dq != "PASS":
                what = ("The analysis was STOPPED after technical QC (maps show the unfiltered bins)." if stopped else
                        "The analysis continued with force: true - treat every result below as exploratory."
                        if dq == "STOP" else "The analysis continued.")
                alerts.append(f'<div class="alert{" stop" if dq == "STOP" else ""}">Data quality {esc(dq)}: '
                              f'{esc(fs.get("data_quality_reason", ""))} {esc(what)}</div>')
        qc = read_csv(figdir, "technical_qc_summary.csv")
        if qc is not None and len(qc):
            r = qc.iloc[0]
            pct_out = r.get("pct_bins_filtered_out")
            items = [("Bins (unfiltered; run stopped)" if stopped else "Bins (filtered)", r.get("n_bins")),
                     ("Bins filtered out", f"{pct_out:.1f}%" if pct_out is not None and pct_out == pct_out else None), ("Median total counts", r.get("median_total_counts")),
                     ("Median genes / bin", r.get("median_n_genes")), ("Tissue sections", r.get("num_tissue")),
                     ("Stray bins", r.get("n_stray_bins_after_filter", r.get("n_stray_bins"))),
                     ("Section depth max/min", r.get("section_depth_max_min_ratio"))]
            sec = read_csv(figdir, "section_depth_summary.csv")
            n_found = len(sec) if sec is not None else None
            if n_found is not None:
                items[4] = ("Tissue sections (set / found)", f"{r.get('num_tissue')} / {n_found}")
            cards += [f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{esc(fmt(v))}</div></div>'
                      for k, v in items if v is not None and v == v]
            if n_found is not None and n_found < int(r.get("num_tissue", 1)):
                alerts.append(f'<div class="alert">num_tissue = {int(r.get("num_tissue"))} but only {n_found} '
                              f'section(s) hold the filtered bins: the sections were joined (connected) in the '
                              f'unfiltered data, so section labels - and the "within section" control - are not '
                              f'informative in this run.</div>')
        idx = read_csv(figdir, "all_celltypes_moran_index_table.csv")
        if idx is not None and "qc_verdict" in idx:
            sfx = "_resid" if "empirical_p_value_resid" in idx else ""
            for _, r in idx.iterrows():
                rows.append(("Spatial structure", r["celltype"], r["qc_verdict"],
                             f"z = {fmt(r.get('z_score' + sfx, NAN))}, p = {fmt(r.get('empirical_p_value' + sfx, NAN))}, "
                             f"q = {fmt(r.get('fdr_qvalue' + sfx, NAN))} ({int(r.get('n_null', 0)):,} random gene sets), "
                             f"depth R² = {fmt(r.get('depth_R2', NAN))}"))
            family.append(f"Spatial structure: depth-corrected test; q = FDR across the "
                          f"{int(idx['n_tested'].iloc[0]) if 'n_tested' in idx else len(idx)} marker sets of this "
                          f"run ({', '.join(map(str, idx['celltype']))}).")
        co = read_csv(figdir, "all_pairs_colocalization_table.csv")
        if co is not None and len(co):
            vcol = "verdict_fdr" if "verdict_fdr" in co else ("verdict" if "verdict" in co else None)
            for _, r in co.iterrows():
                rows.append(("Co-localization", r.get("pair", ""), r.get(vcol, "") if vcol else "",
                             f"I_AB = {fmt(r.get('I_AB', NAN))}, z (swap A / B) = {fmt(r.get('z_swapA', NAN))} / "
                             f"{fmt(r.get('z_swapB', NAN))}, p = {fmt(r.get('p_colocalized', NAN))}, "
                             f"q = {fmt(r.get('q_colocalized', NAN))} ({int(r.get('n_null', 0)):,} random gene sets)"))
            family.append(f"Co-localization: q = FDR across the "
                          f"{int(co['n_tested'].iloc[0]) if 'n_tested' in co else len(co)} cell-type pair(s) of this run.")
        ctl = read_csv(figdir, "colocalization_technical_controls.csv")
        if ctl is not None and len(ctl):
            for _, r in ctl.iterrows():
                rows.append(("Control: within " + str(r.get("control", "")).replace("tissue_", ""),
                             r.get("pair", ""), r.get("verdict", ""),
                             f"I_AB = {fmt(r.get('I_AB', NAN))}, p = {fmt(r.get('p_colocalized', NAN))}"))
        ann = read_csv(figdir, "celltype_annotation_counts.csv")
        if ann is not None and len(ann):
            top = ann[~ann["label"].isin(["Structural Base"])]
            cards.append(f'<div class="card"><div class="k">Annotated bins</div><div class="v">'
                         f'{int(top["n_bins"].sum()):,}</div></div>')
        rows += self.gene_pair_overview_rows()
        table = ""
        if rows:
            body = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td>{Html.cell('verdict', c)}<td>{esc(d)}</td></tr>"
                           for a, b, c, d in rows)
            fam = "".join(f"<br>{esc(f)}" for f in family)
            table = (f'<div class="block wide"><div class="cap"><b>Verdicts</b></div>'
                     f'<p class="note">A verdict is a threshold at q &lt; 0.05. Read it with the numbers next to it: '
                     f'z (effect size), p and q (how close to 0.05), the number of random gene sets, and depth R². '
                     f'Results near q = 0.05 can change with the list of marker sets tested - see report_guide.pdf, '
                     f'"Reading a verdict".{fam}</p>'
                     f'<div class="tw full"><table>'
                     f'<thead><tr><th>Test</th><th>Marker set / pair</th><th>Verdict</th><th>Key numbers</th></tr>'
                     f'</thead><tbody>{body}</tbody></table></div></div>')
        if not cards and not table:
            return ""
        return (f'<section id="overview"><h2>Overview</h2><div class="cards">{"".join(cards)}</div>'
                f'{"".join(alerts)}<div class="grid">{table}</div></section>')

    # ------------------------------------------------------------------ 1 technical QC
    def technical_qc(self, h: Html) -> str:
        return h.section("qc", "1 · Technical QC",
                         "Marker-independent. Sections and strays were found on the unfiltered bins; maps show the "
                         + ("UNFILTERED bins (the run stopped at the data-quality gate)." if self.stopped()
                            else "filtered bins.") + " Look for straight-edged rectangles (imaging FOVs) on the depth maps - they "
                         "act through total counts, which every test below adjusts for.", [
            h.table("technical_qc_summary.csv", "Technical QC summary", transpose=True),
            h.table("filter_summary.csv", "Bin filter: bins removed and why", "A bin is removed when total counts "
                    "&lt; min_counts or mito % &ge; pct_mt; one bin can fail both.", transpose=True),
            h.table("section_depth_summary.csv", "Depth per tissue section",
                    "Only when sections are a technical variable (tissue_qc)."),
            h.table("stray_pieces.csv", "Stray pieces (label only)", visible_rows=5,
                    note="Pieces outside the main section(s); labeled, not removed. The box shows 5 rows; "
                         "scroll for the rest."),
            h.frow(h.figure("spatial_qc_depth_maps", "Depth maps (log1p total counts, genes per bin)"),
                   h.figure("spatial_qc_depth_highpass", "High-pass depth map (total counts minus regional trend)")),
        ])

    # ------------------------------------------------------------------ 2 spatial structure
    def spatial_structure(self, h: Html) -> str:
        files = h.files
        celltypes = sorted(set(files.stems("_null_test_summary.csv")) | set(files.stems("_moran_stats.csv")))
        idx = read_csv(self.figdir, "all_celltypes_moran_index_table.csv")
        if idx is not None and "celltype" in idx:          # same order as the index table
            known = set(idx["celltype"].astype(str))
            celltypes = [c for c in idx["celltype"].astype(str) if c in celltypes] + \
                        [c for c in celltypes if c not in known]
        per_ct = []
        for ct in celltypes:
            r = first_row(idx, "celltype", ct)
            sfx = "_resid" if "z_score_resid" in r else ""
            summ = (f'{h.badge(r.get("qc_verdict", ""))} z = {esc(fmt(r.get("z_score" + sfx, NAN)))}, '
                    f'q = {esc(fmt(r.get("fdr_qvalue" + sfx, NAN)))}') if r else ""
            per_ct.append(h.sub(f"Marker set: {ct}", [
                h.figure(f"spatial{ct}_smooth", f"{ct} signature score (smoothed)", wide=True),
                h.frow(h.figure(f"{ct}_null_distribution_plot", "Null distribution (raw)"),
                       h.figure(f"{ct}_null_distribution_plot_resid", "Null distribution (depth-corrected)")),
                h.table(f"{ct}_null_test_summary.csv", "Random-gene-set null test", transpose=True),
                h.table(f"{ct}_score_total_counts_correlation.csv", "Score vs total counts", transpose=True),
                h.table(f"{ct}_moran_stats.csv", "Analytic Moran's I (squidpy, reference only)"),
            ], summ))
            files.get(f"{ct}_null_moran_distribution.csv")     # raw null draws: not listed under Other files
        out = h.section("moran", "2 · Spatial structure of marker sets (Moran's I)",
                        "Is each marker set spatially structured beyond random genes of matched expression, and "
                        "beyond sequencing depth? qc_verdict: PASS / DEPTH_DRIVEN / MASKED_BY_DEPTH / "
                        "NO_SPATIAL_SIGNAL. q-values are corrected across the marker sets of this run (n_tested). "
                        "Click a marker set below to open its maps, null distributions and tables.", [
            h.figure("morans_i_summary_chart", "Moran's I across marker sets (depth-corrected)"),
            h.table("all_celltypes_moran_index_table.csv", "Index table (FDR across marker sets)"),
            h.table("morans_i_summary_table.csv", "All Moran's I results side by side"),
            h.table("total_counts_confound_baseline.csv", "Baseline: Moran's I of total counts", transpose=True),
        ], per_ct)
        for stem in ("all_celltypes_null_distribution_plot", "all_celltypes_null_distribution_plot_resid"):
            files.figure(stem)        # combined panels: per-marker-set plots are shown instead
        return out

    # ------------------------------------------------------------------ 3 niches and DE
    @staticmethod
    def _niche_key(n: str):
        for suffix, rank in (("_niche", 0), ("_enriched_region", 1)):
            if n.endswith(suffix):
                return (n[: -len(suffix)].lower(), rank)
        return (n.lower(), 2)

    def niches(self, h: Html) -> str:
        files, figdir = h.files, self.figdir
        names = sorted((n for n in files.stems("_depth_check.csv") if not n.endswith("_after_matching")),
                       key=self._niche_key)
        blocks = []
        for n in names:
            stages = depth_stages(figdir, n)
            bits = []
            if "before_matching" in stages:
                bits.append(f'{int(stages["before_matching"]["n_region"]):,} bins')
            if "after_matching" in stages:
                bits.append(f'depth AUROC after matching {fmt(float(stages["after_matching"]["depth_auroc"]))}')
            n_up = n_sig_up(os.path.join(figdir, f"{n}_de_genes.csv"))
            if n_up is not None:
                bits.append(f"{n_up:,} significant up-regulated genes")
            top = self.niche_top_percent(n[: -len("_niche")]) if n.endswith("_niche") else None
            kind = (f"high-confidence niche (top {top or '5'}% score)" if n.endswith("_niche")
                    else "enriched region (any marker)" if n.endswith("_enriched_region") else "")
            summ = esc(" · ".join(([kind] if kind else []) + bits))
            mask_stem = next((s for s in (f"spatial_{n}_mask", f"spatial_{n.replace('_region', '')}_region_mask")
                              if files.figure(s)[0]), f"spatial_{n}_mask")
            de_full = files.get(f"{n}_de_genes.csv")
            blocks.append(h.sub(f"Niche / region: {n}", [
                h.frow(h.figure(mask_stem, f"{n} mask"),
                       h.figure(f"{n}_de_top20_dotplot", "Top up-regulated DE genes (dot plot)",
                                "Up to 20 significant up-regulated genes, by log fold change; depth-matched bins. "
                                "Dot size = fraction of bins detected, color = mean log-normalized expression.")),
                h.table(f"{n}_definition.csv", "Niche definition (top % of bins by smoothed score)", transpose=True),
                h.depth_table(n),
                h.table(f"{n}_de_top20.csv", "Top DE genes (depth-matched)",
                        f'Full table: <a href="{files.href(de_full)}">{esc(de_full)}</a>' if de_full else None),
            ], summ))
        return h.section("de", "3 · Niches and differential expression",
                         "Depth AUROC ≈ 0.5 after matching means DE genes cannot be explained by sequencing depth. "
                         "Each cell type has a high-confidence niche and an enriched region; click one to open it.",
                         [], blocks)

    # ------------------------------------------------------------------ 4 annotation
    def annotation(self, h: Html) -> str:
        files = h.files
        note = ""
        ann = read_csv(self.figdir, "celltype_annotation_counts.csv")
        if ann is not None:
            mix = ann.loc[ann["label"] == "Mix", "combinations_in_mix"] if "combinations_in_mix" in ann else []
            note = ("Note: bins annotated with several cell types whose combination has fewer than the minimum "
                    "number of bins (annotation.min_bins, default 100) are labeled as the Mix cell type."
                    + (f" Here {str(mix.iloc[0]).count(';') + 1} combinations "
                       f"({int(ann.loc[ann['label'] == 'Mix', 'n_bins'].sum()):,} bins) are labeled Mix (list in the "
                       f"combinations_in_mix column of the counts table)."
                       if len(mix) and str(mix.iloc[0]) not in ("", "nan") else ""))
            if "plotted_as" in ann:
                labs = ann[~ann["label"].isin(["Structural Base", "Mix"])]
                shown = labs[labs["plotted_as"] == labs["label"]]
                hidden = labs[labs["plotted_as"] != labs["label"]]
                if len(hidden):
                    note += (f" Plots show the {len(shown)} largest labels in colour (&ge; "
                             f"{int(shown['n_bins'].min()) if len(shown) else 0} bins each); the other {len(hidden)} "
                             f"labels ({int(hidden['n_bins'].sum())} bins) are plotted as Mix. All labels are in "
                             f"obs['celltype_annotation'] of the saved h5ad.")
        counts_csv = files.get("celltype_annotation_counts.csv")       # linked in the intro
        if counts_csv:
            note += f' Bins per label: <a href="{files.href(counts_csv)}">{esc(counts_csv)}</a>.'
        return h.section("annotation", "4 · Cell-type annotation",
                         "Every bin labeled from the high-confidence niches (section 3, strategy 1: top "
                         f"{self.niche_top_percent() or '5'}% of the smoothed signature score). One niche -> the cell type; several -> the names joined "
                         "(e.g. NK_Neuron); none -> Structural Base. " + note, [
            h.frow(h.figure("spatial_celltype_annotation", "Cell-type annotation map"),
                   h.figure("celltype_annotation_bin_counts", "Bins per annotation label")),
        ])

    # ------------------------------------------------------------------ 5 co-localization
    def colocalization(self, h: Html) -> str:
        files = h.files
        allp = read_csv(self.figdir, "all_pairs_colocalization_table.csv")
        blocks = []
        for p in sorted(files.stems("_colocalization_summary.csv")):
            files.get(f"{p}_colocalization_null.csv")
            r = first_row(allp, "pair", p) or first_row(read_csv(self.figdir, f"{p}_colocalization_summary.csv"), "pair", p)
            v = r.get("verdict_fdr", r.get("verdict", ""))
            q = r.get("q_colocalized", r.get("p_colocalized", NAN))
            summ = (f'{h.badge(v)} I_AB = {esc(fmt(r.get("I_AB", NAN)))}, '
                    f'{"q" if "q_colocalized" in r else "p"} = {esc(fmt(q))}') if r else ""
            blocks.append(h.sub(f"Pair: {p}", [
                h.table(f"{p}_colocalization_summary.csv", "Co-localization test", transpose=True),
                h.table(f"{p}_colocalization_by_distance.csv", "By distance"),      # side by side with the test
                h.frow(h.figure(f"{p}_colocalization_null_plot", "Observed I_AB vs. random-gene-set nulls"),
                       h.figure(f"{p}_colocalization_by_distance", "Co-localization by distance")),
            ], summ))
        return h.section("coloc", "5 · Co-localization",
                         "Co-localization, not interaction. Depth-corrected; both nulls (swap A, swap B) must hold. "
                         "Click a pair to open its details.", [
            h.table("all_pairs_colocalization_table.csv", "All pairs (FDR across pairs)"),
            h.table("colocalization_technical_controls.csv", "Technical control: within tissue section"),
        ], blocks)

    # ------------------------------------------------------------------ 6 gene pairs
    @staticmethod
    def gene_pair_verdict(r: dict, alpha: float = 0.05) -> str:
        """Bivariate Moran's I row -> CO-LOCALIZED / SEGREGATED / NOT_SIGNIFICANT (FDR across directions)."""
        q = r.get("fdr_qvalue", r.get("p_sim", NAN))
        if not (q == q) or q >= alpha:
            return "NOT_SIGNIFICANT"
        return "CO-LOCALIZED" if r.get("bv_moran_I", 0) > 0 else "SEGREGATED"

    def gene_pair_overview_rows(self) -> list[tuple]:
        gp = os.path.join(self.figdir, self.GENE_PAIR_DIR)
        rows = []
        if not os.path.isdir(gp):
            return rows
        for f in sorted(os.listdir(gp)):
            if not f.endswith(self.GENE_PAIR_SUFFIX):
                continue
            name = f[: -len(self.GENE_PAIR_SUFFIX)]
            bv = read_csv(gp, f)
            if bv is None or not len(bv):
                continue
            corr = read_csv(gp, f"{name}_coexpression_correlation.csv")
            rho = ""
            if corr is not None and len(corr) == 1 and "spearman_rho" in corr:
                rho = f", Spearman ρ = {fmt(float(corr['spearman_rho'].iloc[0]))}"
            for _, r in bv.iterrows():
                rows.append(("Gene pair (supporting)", f"{r.get('reference_gene', '')} → {r.get('lag_gene', '')}",
                             self.gene_pair_verdict(r),
                             f"bivariate I = {fmt(r.get('bv_moran_I', NAN))}, q = {fmt(r.get('fdr_qvalue', NAN))}{rho}"))
        return rows

    def gene_pairs(self) -> str:
        gp = os.path.join(self.figdir, self.GENE_PAIR_DIR)
        if not os.path.isdir(gp) or not os.listdir(gp):
            return ""
        files = FileIndex(gp, prefix=f"{self.GENE_PAIR_DIR}/")
        h = Html(files, self.embedder, self.self_contained, self.previewer)
        blocks = []
        stopped = set(files.stems(self.GENE_PAIR_STATUS))
        for p in sorted(set(files.stems(self.GENE_PAIR_SUFFIX)) | stopped):
            if p in stopped:                        # pair stopped: badge + reason, no results
                st = first_row(read_csv(gp, f"{p}{self.GENE_PAIR_STATUS}"), "pair", p)
                files.get(f"{p}{self.GENE_PAIR_STATUS}")
                reason = st.get("reason", "") if st else ""
                blocks.append(h.sub(f"Gene pair: {p.replace('_', ' + ')}", [
                    f'<div class="block wide"><p><b>{h.badge("STOP")} Not analysed.</b> {esc(reason)}</p>'
                    f'<p class="note">The co-expression niche of this pair cannot be compared with the rest of '
                    f'the tissue on depth-matched bins, so no results are reported for it. Status: '
                    f'<a href="{files.href(p + self.GENE_PAIR_STATUS)}">{esc(p + self.GENE_PAIR_STATUS)}</a>.</p>'
                    f'</div>'], f'{h.badge("STOP")} too few bins for a depth-fair comparison'))
                continue
            de_full = files.get(f"{p}_coexpr_de_genes.csv")
            bv = read_csv(gp, f"{p}{self.GENE_PAIR_SUFFIX}")
            summ = ""
            if bv is not None and len(bv):
                r = bv.sort_values("p_sim").iloc[0].to_dict()
                summ = (f'{h.badge(self.gene_pair_verdict(r))} bivariate I = {esc(fmt(r.get("bv_moran_I", NAN)))}, '
                        f'q = {esc(fmt(r.get("fdr_qvalue", NAN)))}')
            blocks.append(h.sub(f"Gene pair: {p.replace('_', ' + ')}", [
                h.table(f"{p}{self.GENE_PAIR_SUFFIX}", "Bivariate Moran's I (depth-residualized, permutation null)",
                        "Both directions: reference gene vs. the neighborhood average of the other gene. "
                        "FDR across directions."),
                h.table(f"{p}_coexpression_correlation.csv",
                        "Same-bin correlation (Spearman primary, Pearson for comparison)", transpose=True),
                h.frow(h.figure(f"spatial_{p}_coexpr_mask", "Co-expression niche (bins with both genes detected)"),
                       h.figure(f"{p}_coexpr_de_top20_dotplot", "Top up-regulated DE genes (dot plot)",
                                "Up to 20 significant up-regulated genes, by log fold change; depth-matched bins.")),
                h.depth_table(f"{p}_coexpr", "Depth check: co-expression niche vs. rest of tissue"),
                h.table(f"{p}_coexpr_de_top20.csv", "Top DE genes: co-expression niche vs. rest of tissue (depth-matched)",
                        f'Full table: <a href="{files.href(de_full)}">{esc(de_full)}</a>' if de_full else None),
            ], summ))
        rest = files.rest()
        other = ""
        if rest:
            links = "".join(f'<li><a href="{files.href(f)}">{esc(f)}</a></li>' for f in rest)
            other = f'<h3>Other gene-pair files</h3><ul class="files">{links}</ul>'
        if not blocks and not other:
            return ""
        return h.section("genepairs", "6 · Gene-pair co-localization (optional)",
                         "Outputs of <code>marqual-st gene-pairs</code>. The bivariate Moran's I null shuffles "
                         "locations, which is weaker than the matched random-gene-set null of the cell-type test "
                         "(section 5) - supporting evidence only.", [], blocks + [other])

    # ------------------------------------------------------------------ build
    def parameters(self, h: Html) -> str:
        """Marker sets and software versions; the full settings list stays in run_parameters.csv (linked)."""
        f = h.files.get(self.PARAM_FILE)
        if f is None:
            return ""
        try:
            df = pd.read_csv(h.files.path(f), dtype=str, keep_default_na=False)
        except Exception:
            return ""
        if df.empty:
            return ""
        blocks = []
        ms = df[(df["step"] == "marker_sets") & (df["parameter"] == "marker_list")]
        if len(ms):
            body = "".join(f"<tr><td>{esc(r.call)}</td><td>{len(r.value.split(', '))}</td>"
                           f'<td style="white-space:normal">{esc(r.value)}</td></tr>' for r in ms.itertuples())
            blocks.append(f'<div class="block wide"><div class="cap"><b>Marker sets</b></div><div class="tw"><table>'
                          f'<thead><tr><th>Cell type</th><th>Markers</th><th>Genes</th></tr></thead>'
                          f'<tbody>{body}</tbody></table></div></div>')
        # software versions; every other setting stays in run_parameters.csv (linked below)
        sw = df[df["step"] == "software"]
        if len(sw):
            body = "".join(f'<tr><td>{esc(r.parameter)}</td><td style="white-space:normal">{esc(r.value)}</td></tr>'
                           for r in sw.drop_duplicates("parameter").itertuples())
            blocks.append('<div class="block"><div class="cap"><b>Software versions</b></div><div class="tw"><table>'
                          f'<thead><tr><th>Package</th><th>Version</th></tr></thead><tbody>{body}</tbody></table>'
                          '</div></div>')
        if not blocks:
            return ""
        return h.section("params", "7 · Run parameters",
                         "The marker sets and the software versions of this run. Every other setting - the values "
                         "set in params.csv and the defaults it left unchanged, then each step as it ran - is in "
                         f'<a href="{h.files.href(f)}">{esc(f)}</a> (also uns["run_parameters"] of the saved h5ad). '
                         "To repeat the run, use params_used.csv and marker_sets_used.csv next to the results.",
                         blocks)

    @staticmethod
    def other_files(rest: list[str]) -> str:
        if not rest:
            return ""
        links = "".join(f'<li><a href="{esc(f)}">{esc(f)}</a></li>' for f in rest)
        return (f'<section id="other"><h2>8 · Other files</h2><p class="intro">Files in the folder not placed '
                f'above (raw null draws, full tables, extra plots).</p><ul class="files">{links}</ul></section>')

    @staticmethod
    def template() -> str:
        return resources.files("marqual_st").joinpath("templates/report.html").read_text(encoding="utf-8")

    def build(self, out_html: str | Path | None = None) -> Path:
        """Write the report (default {figdir}/{sample}_QCreport.html; links are relative to figdir)."""
        from . import __version__
        if not os.path.isdir(self.figdir):
            raise FileNotFoundError(f"{self.figdir} is not a folder.")
        files = FileIndex(self.figdir)
        h = Html(files, self.embedder, self.self_contained, self.previewer)
        out_html = Path(out_html or os.path.join(self.figdir, self.report_name(self.sample)))

        # order matters: each section claims its files; "Other files" gets the rest
        s_qc, s_moran, s_de = self.technical_qc(h), self.spatial_structure(h), self.niches(h)
        s_coloc = self.colocalization(h)
        s_gp = self.gene_pairs()
        s_ann = self.annotation(h)
        s_par = self.parameters(h)
        rest = files.rest()
        body = "".join([self.overview(), s_qc, s_moran, s_de, s_ann, s_coloc, s_gp, s_par, self.other_files(rest)])

        nav = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in self.NAV if f'id="{i}"' in body)
        if 'class="sub"' in body:
            nav += ('<span class="navbtns"><button type="button" onclick="setAll(true)">Expand all</button>'
                    '<button type="button" onclick="setAll(false)">Collapse all</button></span>')
        title = self.title or f"Marker-based spatial QC pipeline report{f' · {self.sample}' if self.sample else ''}"
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        page = self.template().format(
            title=esc(title), nav=nav, body=body, stamp=stamp, n=len(files.all), version=esc(__version__),
            figdir=esc(os.sep.join(os.path.abspath(self.figdir).split(os.sep)[-2:])))
        out_html.write_text(page, encoding="utf-8")
        log.info(f"Report: {out_html} ({out_html.stat().st_size / 1e6:.1f} MB; {len(files.all)} files, "
                 f"{len(rest)} under 'Other files')")
        if self.previewer.missing:
            log.warning("Figure previews were not embedded - no PDF renderer found. Install one with "
                        "'pip install pypdfium2' (or PyMuPDF, or poppler's pdftoppm) and rebuild the report; "
                        "the PDF figures are linked in the meantime.")
        return out_html
