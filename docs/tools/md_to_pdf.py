#!/usr/bin/env python3
"""Convert a simple Markdown note (headings, paragraphs, **bold**, bullet lists, tables, images, rules) to an A4 PDF.

What: a small layout engine on matplotlib's PDF backend (vector, selectable text, DejaVu Sans, which covers ≥ → – ·
× …). Tables get column widths from their content, wrap inside cells, repeat the header row on a new page and get
a grey header background; images are embedded at full resolution and scaled to the text width; every page has a
page number. Supported Markdown: '#'/'##'/'###' headings, paragraphs, '- ' bullets, '| a | b |' tables, '![alt](path)'
images (paths relative to the Markdown file), '---' rules, **bold**; `code` marks are dropped.

Why: no pandoc / LaTeX / ReportLab on the cluster; matplotlib is available in envs/totalseg.

Usage: <envs/totalseg>/bin/python -I docs/tools/md_to_pdf.py <note.md> <out.pdf> [--title "PDF title"]
"""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

try:  # unhinted outlines, so measured widths match what the PDF draws
    from matplotlib.ft2font import LoadFlags  # noqa: E402
    NO_HINT = LoadFlags.NO_HINTING
except ImportError:  # matplotlib < 3.10
    from matplotlib.ft2font import LOAD_NO_HINTING as NO_HINT  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

PAGE_W, PAGE_H = 595.28, 841.89          # A4 in points
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 48, 48, 50, 50
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R
FONTS = {False: font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")),
         True: font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))}
_FT = {}
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def text_width(s, size, bold=False):
    f = _FT.setdefault(bold, FT2Font(FONTS[bold]))
    f.set_size(size, 720)            # measure at 10x resolution without hinting, then scale back to points
    f.set_text(s, 0.0, flags=NO_HINT)
    return f.get_width_height()[0] / 640.0


def runs(text):
    """'a **b** c' -> [('a ', False), ('b', True), (' c', False)]; drops `code` marks."""
    text = text.replace("`", "")
    out, bold = [], False
    for k, part in enumerate(text.split("**")):
        if part:
            out.append((part, bold))
        bold = not bold if k < len(text.split("**")) - 1 else bold
    return out


def wrap(text, size, width):
    """Lines of [(word_text, bold, x_offset)] for rich text within width (points)."""
    words = []
    for t, b in runs(text):
        for tok in re.findall(r"\S+|\s+", t):
            words.append((tok, b))
    lines, cur, x = [], [], 0.0
    space = text_width(" ", size)
    for tok, b in words:
        if tok.isspace():
            if cur:
                x += space
            continue
        w = text_width(tok, size, b)
        if w > width:  # very long token (URL): break by characters
            piece = ""
            for ch in tok:
                if text_width(piece + ch, size, b) + (x if cur else 0) > width and piece:
                    cur.append((piece, b, x))
                    lines.append(cur)
                    cur, x, piece = [], 0.0, ch
                else:
                    piece += ch
            tok, w = piece, text_width(piece, size, b)
        if cur and x + w > width:
            lines.append(cur)
            cur, x = [], 0.0
        cur.append((tok, b, x))
        x += w
    if cur:
        lines.append(cur)
    return lines or [[]]


class Doc:
    def __init__(self, path, title):
        self.pdf = PdfPages(path, metadata={"Title": title})
        self.page = 0
        self.fig = None
        self.new_page()

    def new_page(self):
        if self.fig is not None:
            self.finish_page()
        self.fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
        self.page += 1
        self.y = PAGE_H - MARGIN_T

    def finish_page(self):
        self.fig.text(0.5, 22 / PAGE_H, "Page %d" % self.page, ha="center", fontsize=8, color="#777777",
                      parse_math=False)
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def need(self, h):
        if self.y - h < MARGIN_B:
            self.new_page()
            return True
        return False

    def put(self, x, y, s, size, bold=False, color="black"):
        self.fig.text(x / PAGE_W, y / PAGE_H, s, fontsize=size, fontweight="bold" if bold else "normal",
                      color=color, va="baseline", ha="left", parse_math=False)

    def line(self, x0, y0, x1, y1, color="#999999", lw=0.6):
        self.fig.add_artist(plt.Line2D([x0 / PAGE_W, x1 / PAGE_W], [y0 / PAGE_H, y1 / PAGE_H],
                                       transform=self.fig.transFigure, color=color, linewidth=lw))

    def rect(self, x, y, w, h, color):
        self.fig.add_artist(Rectangle((x / PAGE_W, y / PAGE_H), w / PAGE_W, h / PAGE_H,
                                      transform=self.fig.transFigure, facecolor=color, edgecolor="none", zorder=0))

    def paragraph(self, text, size=10, lead=13.2, indent=0.0, bullet=False, color="black", space_after=5):
        lines = wrap(text, size, CONTENT_W - indent - (12 if bullet else 0))
        for k, ln in enumerate(lines):
            self.need(lead)
            self.y -= lead
            x0 = MARGIN_L + indent + (12 if bullet else 0)
            if bullet and k == 0:
                self.put(MARGIN_L + indent + 2, self.y, "•", size)
            for tok, b, dx in ln:
                self.put(x0 + dx, self.y, tok, size, b, color)
        self.y -= space_after

    def heading(self, text, level):
        size = {1: 19, 2: 13.5, 3: 11.5}[level]
        before = {1: 0, 2: 14, 3: 9}[level]
        if self.y < PAGE_H - MARGIN_T - 1:
            self.y -= before
        self.need(size * 1.4 + 40)  # keep a heading with what follows
        self.paragraph("**%s**" % text.replace("**", ""), size=size, lead=size * 1.3,
                       color="#1F3A5F" if level > 1 else "black", space_after=4 if level > 1 else 6)

    def rule(self):
        self.need(10)
        self.y -= 4
        self.line(MARGIN_L, self.y, PAGE_W - MARGIN_R, self.y, color="#BBBBBB", lw=0.8)
        self.y -= 8

    def table(self, rows):
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        show_header = any(c.strip() for c in rows[0])
        if not show_header and len(rows) > 1:   # '| | |' header: no header row at all
            rows = rows[1:]
        size = 8.5 if ncol <= 4 else 7.8 if ncol <= 6 else 7.0
        lead, pad = size * 1.28, 3.0
        nat = [max(sum(text_width(t, size, b or (k == 0 and show_header)) for t, b in runs(r[c]))
                   for k, r in enumerate(rows)) + 2 * pad + 1
               for c in range(ncol)]
        mn = [max(max((text_width(w, size, True) for w in re.findall(r"\S+", r[c].replace("**", "")) or [""]),
                      default=0) for r in rows) + 2 * pad + 1 for c in range(ncol)]
        mn = [min(m, CONTENT_W * 0.45) for m in mn]
        if sum(nat) <= CONTENT_W:
            widths = nat
        else:
            extra = CONTENT_W - sum(mn)
            grow = [max(n - m, 0) for n, m in zip(nat, mn)]
            widths = [m + (extra * g / sum(grow) if sum(grow) and extra > 0 else 0) for m, g in zip(mn, grow)]
            if sum(widths) > CONTENT_W:
                f = CONTENT_W / sum(widths)
                widths = [w * f for w in widths]
        total_w = sum(widths)
        if show_header:
            rows = [["**%s**" % c.replace("**", "") if c else "" for c in rows[0]]] + rows[1:]   # header in bold
        wrapped = [[wrap(r[c], size, widths[c] - 2 * pad) for c in range(ncol)] for r in rows]
        heights = [max(len(cell) for cell in wr) * lead + 2 * pad for wr in wrapped]

        def draw_row(i, header):
            h = heights[i]
            top = self.y
            if header:
                self.rect(MARGIN_L, top - h, total_w, h, "#E8EDF3")
            x = MARGIN_L
            for c in range(ncol):
                yy = top - pad
                for ln in wrapped[i][c]:
                    yy -= lead
                    for tok, b, dx in ln:
                        self.put(x + pad + dx, yy + lead * 0.22, tok, size, b)
                x += widths[c]
            self.line(MARGIN_L, top - h, MARGIN_L + total_w, top - h, lw=0.4)
            x = MARGIN_L
            for c in range(ncol + 1):
                self.line(x, top, x, top - h, lw=0.4)
                if c < ncol:
                    x += widths[c]
            self.y -= h

        self.need(heights[0] + (heights[1] if len(rows) > 1 else 0))
        self.line(MARGIN_L, self.y, MARGIN_L + total_w, self.y, lw=0.4)
        draw_row(0, show_header)
        for i in range(1, len(rows)):
            if self.need(heights[i]):
                self.line(MARGIN_L, self.y, MARGIN_L + total_w, self.y, lw=0.4)
                if show_header:
                    draw_row(0, True)
            draw_row(i, False)
        self.y -= 8

    @staticmethod
    def image_size(path):
        with Image.open(path) as im:
            w = CONTENT_W
            h = w * im.height / im.width
            if h > PAGE_H - MARGIN_T - MARGIN_B:
                h = PAGE_H - MARGIN_T - MARGIN_B
                w = h * im.width / im.height
        return w, h

    def image(self, path):
        im = Image.open(path).convert("RGB")
        w, h = self.image_size(path)
        self.need(h + 6)
        self.y -= h
        ax = self.fig.add_axes([MARGIN_L / PAGE_W, self.y / PAGE_H, w / PAGE_W, h / PAGE_H])
        ax.imshow(im, interpolation="none")
        ax.set_axis_off()
        self.y -= 10

    def close(self):
        self.finish_page()
        self.pdf.close()


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("markdown")
    ap.add_argument("pdf")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(args.markdown))
    lines = open(args.markdown, encoding="utf-8").read().split("\n")
    title = args.title or next((l[2:].strip() for l in lines if l.startswith("# ")), "")
    doc = Doc(args.pdf + ".part.pdf", title)
    i, para = 0, []

    def next_image(j):
        """Path of an image that follows line j after blank lines only, else None."""
        while j < len(lines) and not lines[j].strip():
            j += 1
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", lines[j].rstrip()) if j < len(lines) else None
        return os.path.join(base, m.group(1)) if m else None

    def flush():
        if para:
            text = " ".join(para)
            img = next_image(i)
            if img:  # caption + picture on the same page
                n_lines = len(wrap(text, 10, CONTENT_W))
                doc.need(n_lines * 13.2 + 5 + doc.image_size(img)[1] + 6)
            doc.paragraph(text)
            para.clear()

    while i < len(lines):
        ln = lines[i].rstrip()
        m_img = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", ln)
        if not ln.strip():
            flush()
        elif ln.startswith("#"):
            flush()
            level = len(ln) - len(ln.lstrip("#"))
            doc.heading(ln[level:].strip(), min(level, 3))
        elif re.match(r"^-{3,}$", ln.strip()):
            flush()
            doc.rule()
        elif ln.lstrip().startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                if not (re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i]) and "-" in lines[i]):
                    block.append(cells(lines[i]))
                i += 1
            doc.table(block)
            continue
        elif m_img:
            flush()
            doc.image(os.path.join(base, m_img.group(1)))
        elif re.match(r"^\s*[-*] ", ln):
            flush()
            indent = (len(ln) - len(ln.lstrip())) * 4.0
            item = re.sub(r"^\s*[-*] ", "", ln)
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip() and \
                    not re.match(r"^\s*[-*] ", lines[i + 1]):
                i += 1
                item += " " + lines[i].strip()
            doc.paragraph(item, indent=indent, bullet=True, space_after=2)
        elif re.match(r"^\s*\d+\. ", ln):
            flush()
            num, item = re.match(r"^\s*(\d+)\. (.*)$", ln).groups()
            doc.paragraph("%s. %s" % (num, item), indent=4, space_after=2)
        else:
            para.append(ln.strip())
        i += 1
    flush()
    doc.close()
    os.replace(args.pdf + ".part.pdf", args.pdf)
    print("wrote %s (%d pages)" % (args.pdf, doc.page))


if __name__ == "__main__":
    main()
