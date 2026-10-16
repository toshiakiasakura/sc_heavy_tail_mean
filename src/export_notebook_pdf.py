#!/usr/bin/env python3
"""
export_notebook_pdf.py — turn an EXECUTED notebook into a readable PDF.

    python3 src/export_notebook_pdf.py <executed.ipynb> <out.pdf> [--title "..."]

`jupyter nbconvert --to pdf` works out of the box but produces an unreadable PDF for the notebooks
in this project, because all of them print wide DataFrames. Three things go wrong, and each needs
its own fix:

1. **nbconvert hard-wraps every output line at 80 characters** (`charlim` in
   `templates/latex/style_jupyter.tex.j2`, applied as `output.text | wrap_text(charlim)`). The
   wrap uses `textwrap`, which collapses leading whitespace on continuation lines, so a DataFrame
   splits into fragments whose columns no longer line up — the table is not merely ugly, it is
   unreadable. ⚠ Overriding `charlim` via a copied template does NOT work: the value is set under
   `if charlim is not defined`, and a custom template directory did not take effect here even when
   installed in `~/.local/share/jupyter/nbconvert/templates` and selected with `--template`. So
   this script registers a no-op `wrap_text` on the exporter instead, which is the one place the
   filter is actually resolved from.
2. **DataFrames.jl emits ANSI colour codes** (bold headers, coloured eltypes) because IJulia's
   stdout reports colour support. nbconvert's `ansi2latex` turns each run into `\\textbf{...}` /
   `\\textcolor{...}{...}`, which inflates the .tex and breaks lines further. The colour carries no
   information in a PDF, so it is stripped first.
3. **Portrait A4 cannot hold the result.** The widest line in 15j is 144 monospace characters.
   DejaVu Sans Mono has a 0.602 em advance and the nbconvert preamble applies `Scale=0.86`, so at
   `\\footnotesize` (9pt of an 11pt base) a character is 9*0.86*0.602 = 4.66pt and 144 of them need
   671pt — against the 773pt that landscape A4 at 0.5in margins leaves. Hence landscape + `\\fvset`.

The script reports the widest line it found and the width the page can hold, and FAILS if the
second is smaller than the first rather than silently emitting a wrapped table.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# landscape A4, 0.5in margins -> 773pt of text width; see the module docstring for the arithmetic.
TEXT_WIDTH_PT = 773.0
CHAR_PT = 9 * 0.86 * 0.602


def strip_ansi(nb):
    n = 0
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                t = "".join(out["text"])
                u = ANSI.sub("", t)
                n += t != u
                out["text"] = u
            elif out.get("output_type") in ("execute_result", "display_data"):
                d = out.get("data", {})
                if "text/plain" in d:
                    t = "".join(d["text/plain"])
                    u = ANSI.sub("", t)
                    n += t != u
                    d["text/plain"] = u
    return n


def widest_output_line(nb):
    w = 0
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                t = "".join(out["text"])
            elif out.get("output_type") in ("execute_result", "display_data"):
                t = "".join(out.get("data", {}).get("text/plain", ""))
            else:
                continue
            for line in ANSI.sub("", t).split("\n"):
                w = max(w, len(line))
    return w


def patch_preamble(tex, title=None):
    tex = tex.replace(
        r"\geometry{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}",
        r"\geometry{a4paper,landscape,verbose,tmargin=0.5in,bmargin=0.5in,"
        r"lmargin=0.5in,rmargin=0.5in}")
    i = tex.index(r"\begin{document}")
    # `\newcounter{none}`: pandoc renders an UNCAPTIONED markdown table as a longtable guarded by
    # `\def\LTcaptype{none}`, and longtable then calls `\refstepcounter{none}` on a counter nobody
    # defined — "LaTeX Error: No counter 'none' defined" plus a knock-on "Missing number". The PDF
    # still builds, which is exactly why it is worth fixing rather than ignoring.
    tex = tex[:i] + "\n% --- export_notebook_pdf.py: fit wide DataFrame output ---\n" \
                    r"\fvset{fontsize=\footnotesize}" + "\n" \
                    r"\newcounter{none}" + "\n\n" + tex[i:]
    if title:
        tex = re.sub(r"\\title\{[^}]*\}", r"\\title{%s}" % title, tex, count=1)
    return tex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("output")
    ap.add_argument("--title", default=None)
    ap.add_argument("--keep-tex", action="store_true")
    a = ap.parse_args()

    nb = json.load(open(a.notebook))
    n_exec = sum(1 for c in nb["cells"] if c["cell_type"] == "code" and c.get("outputs"))
    if n_exec == 0:
        sys.exit("export_notebook_pdf: %s has no cell outputs — run it first "
                 "(jupyter nbconvert --to notebook --execute ..., never --inplace)" % a.notebook)
    n_ansi = strip_ansi(nb)
    widest = widest_output_line(nb)
    capacity = int(TEXT_WIDTH_PT / CHAR_PT)
    print("cells with output : %d" % n_exec)
    print("ANSI stripped from: %d outputs" % n_ansi)
    print("widest output line: %d chars   page holds %d at \\footnotesize landscape A4"
          % (widest, capacity))
    if widest > capacity:
        sys.exit("export_notebook_pdf: the widest line (%d) exceeds what the page can hold (%d). "
                 "Narrow the printed tables or drop the font a step further; emitting a silently "
                 "wrapped table is worse than failing here." % (widest, capacity))

    # Register a no-op `wrap_text`. This is the fix that the template route could not deliver.
    import nbformat
    from nbconvert import LatexExporter
    from traitlets.config import Config

    # Disable the SVG->PDF preprocessor: it resolves an Inkscape path eagerly and raises
    # FileNotFoundError on images that are not installed here. Every figure in this project is a
    # PNG written by Plots/GR, so nothing is lost.
    cfg = Config()
    cfg.SVG2PDFPreprocessor.enabled = False
    exporter = LatexExporter(config=cfg)
    exporter.register_filter("wrap_text", lambda text, width=100: text)

    node = nbformat.reads(json.dumps(nb), as_version=4)
    body, resources = exporter.from_notebook_node(node)
    body = patch_preamble(body, a.title)

    with tempfile.TemporaryDirectory() as d:
        stem = "nb"
        open(os.path.join(d, stem + ".tex"), "w").write(body)
        for fn, data in (resources.get("outputs") or {}).items():
            open(os.path.join(d, fn), "wb").write(data)
        for i in (1, 2):
            r = subprocess.run(["xelatex", "-interaction=nonstopmode", stem + ".tex"],
                               cwd=d, capture_output=True, text=True)
        log = open(os.path.join(d, stem + ".log"), errors="ignore").read()
        pdf = os.path.join(d, stem + ".pdf")
        if not os.path.exists(pdf):
            print("\n".join(l for l in log.split("\n") if l.startswith("! "))[:2000])
            sys.exit("export_notebook_pdf: xelatex produced no PDF")
        nover = log.count("Overfull \\hbox")
        errlines = [l for l in log.split("\n") if l.startswith("! ")]
        nerr = len(errlines)
        os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
        shutil.copy(pdf, a.output)
        if a.keep_tex:
            shutil.copy(os.path.join(d, stem + ".tex"), a.output + ".tex")
    # Print them. A PDF that built despite errors is the "blank but plausible" failure mode: it
    # looks finished and is quietly missing something.
    print("latex errors %d, overfull hboxes %d" % (nerr, nover))
    for l in errlines[:10]:
        print("   " + l)
    print("wrote %s (%.2f MB)" % (a.output, os.path.getsize(a.output) / 1e6))


if __name__ == "__main__":
    main()
