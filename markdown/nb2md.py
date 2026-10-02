"""Convert the course notebooks to standalone Markdown editions.

Markdown cells are copied (with their maths rewritten so it renders on GitHub and in VS Code,
see fix_markdown), code cells become ```python blocks, text outputs become ```text blocks,
pandas HTML tables become Markdown tables, and PNG figures are written to figures/<notebook>/
next to this script and linked. Re-run after editing a notebook, from the course folder:

    python markdown/nb2md.py                                    # every notebook
    python markdown/nb2md.py notebooks/09_decision_trees.ipynb  # just the ones named
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent              # the Markdown editions are written here
NOTEBOOKS = HERE.parent / "notebooks"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
NOISE = [re.compile(p) for p in (r"^findfont: ", r"^<Figure size .*>$", r"^<Axes: .*>$")]


def is_noise(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return not lines or all(any(p.search(l) for p in NOISE) for l in lines)


# --- Markdown cells: maths and stray tags --------------------------------------------------

KATEX_OPERATOR_SUB = re.compile(
    r"([_^])\\(min|max|sup|inf|lim|log|ln|exp|det|arg|dim|ker|Pr|sin|cos|tan)(?![A-Za-z])")
MARKDOWN_HAZARD = re.compile(r"\\[^A-Za-z]|[_*<>&\[\]~]")   # what GitHub's parser would rewrite
SEP = "<span></span>"            # an empty element: a boundary that both renderers respect
SAFE_BEFORE = " \t\n("           # GitHub only opens maths after these
SAFE_AFTER = " \t\n.,;:)-–/"     # ... and only closes it before these (VS Code: before no letter)
RAW_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9-]*)(?:\s[^<>]*)?/?>")
HTML_TAGS = set("""a abbr b bdo blockquote br caption center cite code dd del details dfn div dl
    dt em figcaption figure font h1 h2 h3 h4 h5 h6 hr i img ins kbd li mark ol p picture pre q rp
    rt ruby s samp small source span strike strong sub summary sup table tbody td tfoot th thead
    time tr tt u ul var wbr""".split())


def fix_markdown(src):
    """Make a Markdown cell render on GitHub and in VS Code the way it does in Jupyter.

    Jupyter lifts the maths out before it parses the Markdown. GitHub parses first: inside
    $...$ and $$...$$ it applies backslash escapes (\\{ -> {, \\, -> ,) and emphasis (the _ of
    }_i), and it accepts a $ only next to a space or a bracket. VS Code keeps the maths intact
    but rejects a $ touching a letter, as in $k$NN. So, without changing what renders:
      - inline maths goes on one line and, when it contains characters Markdown would rewrite,
        is written $`...`$ (GitHub passes it verbatim, VS Code drops the backticks); inside a
        table row, | and \\| are spelled \\vert and \\Vert;
      - an empty <span></span> parts a $ from a neighbouring letter or punctuation mark;
      - display maths on lines of its own becomes a ```math block, elsewhere \\displaystyle;
      - a $ that is not maths becomes <span>\\$</span>, GitHub's documented form;
      - x_\\min becomes x_{\\min}, which KaTeX requires;
      - a tag that is not HTML (the "e</w>" of a tokeniser) is escaped so that it shows.
    Returns the fixed text and a list of (before, after) pairs.
    """
    changes = []
    # fenced code blocks are copied untouched; maths never crosses one (it ends the paragraph)
    chunks = re.split(r"(^[ ]{0,3}(?:```|~~~).*?^[ ]{0,3}(?:```|~~~)[^\n]*$)", src,
                      flags=re.M | re.S)
    return "".join(c if k % 2 else _fix_text(c, changes) for k, c in enumerate(chunks)), changes


def _table_bars(tex):
    """Bars inside maths in a table row: \\| -> \\Vert, | -> \\vert (a \\\\ pair is left alone)."""
    return re.sub(r"\\\\|\\\||\|",
                  lambda m: {"\\\\": "\\\\", "\\|": "\\Vert ", "|": "\\vert "}[m.group()], tex)


def _quote_depth(line):
    return re.match(r"[ \t]*((?:>[ \t]?)*)", line).group(1).count(">")


def _one_line(tex, line):
    """Join a formula's lines, dropping the blockquote markers of its continuation lines."""
    quote = _quote_depth(line)
    join = r"[ \t]*\n[ \t]*" + (r"(?:>[ \t]?){0,%d}[ \t]*" % quote if quote else "")
    return re.sub(join, " ", tex)


def _closing_dollar(src, i, quote):
    """Index of the $ that closes the one at i, or len(src) if the paragraph ends first."""
    n, j = len(src), i + 1
    while j < n:
        if src[j] == "\\":
            j += 2
            continue
        if src[j] == "\n":
            eol = src.find("\n", j + 1)
            if not src[j + 1:n if eol < 0 else eol].strip(" \t>" if quote else " \t"):
                return n                              # a blank line ends the paragraph
        if src[j] == "$" and not (j + 1 < n and src[j + 1].isdigit()):
            return j
        j += 1
    return n


def _fix_text(src, changes):
    out, i, n = [], 0, len(src)

    def math(tex, start, end, line):
        """Write the formula that occupied src[start:end], delimiters included."""
        if line.lstrip().startswith("|"):             # inside a table row
            tex = _table_bars(tex)
        tex = re.sub(r"(?<!\\)\s+$", "", KATEX_OPERATOR_SUB.sub(r"\1{\\\2}", tex))
        new = f"$`{tex}`$" if MARKDOWN_HAZARD.search(tex) and "`" not in tex else f"${tex}$"
        if start > 0 and src[start - 1] not in SAFE_BEFORE:
            new = SEP + new
        if end < n and src[end] not in SAFE_AFTER:
            new += SEP
        if new != src[start:end]:
            changes.append((src[start:end], new))
        out.append(new)

    while i < n:
        ch = src[i]
        if ch == "\\":                                # an escaped character, e.g. \$
            out.append(src[i:i + 2]); i += 2
            continue
        if ch == "`":                                 # inline code span: copy as is
            run = i
            while run < n and src[run] == "`":
                run += 1
            end = src.find(src[i:run], run)
            end = run if end < 0 else end + run - i
            out.append(src[i:end]); i = end
            continue
        line = src[src.rfind("\n", 0, i) + 1:i]       # the line so far, up to here
        if ch == "<":
            m = RAW_TAG.match(src, i)
            if m and m.group(1).lower() not in HTML_TAGS:
                changes.append((m.group(), "&lt;" + m.group()[1:]))
                out.append("&lt;"); i += 1
                continue
        if ch != "$":
            out.append(ch); i += 1
            continue
        if src.startswith("$$", i):                   # display maths
            end = src.find("$$", i + 2)
            if end < 0:
                out.append(src[i:]); break
            eol = src.find("\n", end + 2)
            if line == "" and not src[end + 2:n if eol < 0 else eol].strip():
                tex = KATEX_OPERATOR_SUB.sub(r"\1{\\\2}", src[i + 2:end].strip("\n"))
                new = f"```math\n{tex}\n```"         # a block of its own: kept verbatim
                changes.append((src[i:end + 2], new))
                out.append(new)
            else:                                     # inside a list item or a sentence
                math("\\displaystyle " + _one_line(src[i + 2:end], line).strip(), i, end + 2, line)
            i = end + 2
            continue
        j = (n if i + 1 >= n or src[i + 1].isspace()
             else _closing_dollar(src, i, _quote_depth(line)))
        if j >= n:                                    # "100 $ billed", or no closing $: a dollar
            literal = "$" if line.lstrip().startswith("<") else "<span>\\$</span>"
            if literal != "$":
                changes.append(("$", literal))
            out.append(literal); i += 1
            continue
        math(_one_line(src[i + 1:j], line), i, j + 1, line)
        i = j + 1
    return "".join(out)


# --- DataFrame tables ------------------------------------------------------------------------

def cell_text(t):
    """Escape a table cell so Markdown shows it literally (no emphasis, HTML, maths or splits)."""
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"([|*`$~\\])", r"\\\1", t)
    return re.sub(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])", r"\\_", t)   # word-boundary underscores


def html_table_to_md(html):
    """Render a pandas DataFrame HTML table as a Markdown table.

    The column-label rows merge into one header ("MASE / mean"). pandas' row of index names
    becomes the corner cell, written "index \\ columns" when the column axis has a name too,
    as in a crosstab of contract (rows) against internet_service (columns).
    """
    table = BeautifulSoup(html, "html.parser").find("table")
    if table is None:
        return None

    def grid(rows, header=False):
        out, carry = [], {}                      # carry: col -> remaining rowspan count
        for tr in rows:
            row, col = [], 0
            it = iter(tr.find_all(["th", "td"], recursive=False))
            while True:
                if carry.get(col, 0) > 0:        # sparsified MultiIndex level: leave blank
                    carry[col] -= 1
                    row.append("")
                    col += 1
                    continue
                cell = next(it, None)
                if cell is None:
                    break
                span = int(cell.get("colspan", 1))
                rspan = int(cell.get("rowspan", 1))
                for k in range(span):            # a spanning header labels every column below
                    row.append(cell_text(cell.get_text()) if k == 0 or header else "")
                    if rspan > 1:
                        carry[col + k] = rspan - 1
                col += span
            out.append(row)
        return out

    thead, tbody = table.find("thead"), table.find("tbody")
    body_rows = tbody.find_all("tr") if tbody else table.find_all("tr")
    head = grid(thead.find_all("tr"), header=True) if thead else []
    body = grid(body_rows)
    width = max(len(r) for r in head + body)
    pad = lambda r: r + [""] * (width - len(r))
    head, body = [pad(r) for r in head], [pad(r) for r in body]
    n_index = max((len(tr.find_all("th", recursive=False)) for tr in body_rows), default=0)

    names = [""] * width                         # pandas' last header row: the index names
    if head and n_index and any(head[-1][:n_index]) and not any(head[-1][n_index:]):
        names = head.pop()
    header = []
    for j in range(width):
        labels = []
        for r in head:                           # column labels, top level first
            if r[j] and r[j] not in labels:
                labels.append(r[j])
        if j < n_index:                          # corner: index name \ column-axis name
            axis = " / ".join(labels)
            header.append(f"{names[j]} \\\\ {axis}".strip() if axis else names[j])
        else:
            header.append(" / ".join(labels))
    if not head and names == [""] * width:
        header, body = body[0], body[1:]

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# --- figure alt text ------------------------------------------------------------------------

LITERAL = r"""\s*([rRfF]{0,2})(?:"((?:[^"\\\n]|\\.)*)"|'((?:[^'\\\n]|\\.)*)')"""
TEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Omega": "Ω", "Rightarrow": "⇒", "rightarrow": "→",
    "to": "→", "leftarrow": "←", "uparrow": "↑", "downarrow": "↓", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈", "sim": "~", "propto": "∝",
    "times": "×", "cdot": "·", "infty": "∞", "pm": "±", "star": "*", "top": "ᵀ", "mid": "|",
    "sum": "Σ"}
SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
SUBSCRIPT = dict(zip("0123456789aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))


def tex_to_text(tex):
    """A plain-text reading of a short formula, for alt text: \\lambda -> λ, \\log_{10} -> log₁₀."""
    t = tex.replace("\\{", "").replace("\\}", "").replace("\\|", "‖")   # literal braces
    t = re.sub(r"\\[,;:! ]", " ", t)
    t = re.sub(r"\\(?:mathbf|mathrm|mathit|mathcal|mathbb|boldsymbol|text|operatorname)\{([^{}]*)\}",
               r"\1", t)
    t = re.sub(r"\\hat\{([^{}]*)\}", "\\1\u0302", t)
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", t)
    t = re.sub(r"\\([A-Za-z]+)", lambda m: TEX_SYMBOLS.get(m.group(1), m.group(1)), t)
    t = re.sub(r"\^\{?([0-9+-]+)\}?", lambda m: m.group(1).translate(SUPERSCRIPT), t)
    t = re.sub(r"\^\{?([ᵀ*])\}?", r"\1", t)
    t = re.sub(r"_\{?([0-9a-z]+)\}?",
               lambda m: ("".join(SUBSCRIPT[c] for c in m.group(1))
                          if all(c in SUBSCRIPT for c in m.group(1)) else "_" + m.group(1)), t)
    return t.replace("{", "").replace("}", "").replace("", "{").replace("", "}")


def _literal_text(pieces):
    """The value of a run of (implicitly concatenated) Python string literals, or None for an
    f-string with placeholders."""
    text = []
    for prefix, dq, sq in pieces:
        s = dq or sq
        if "f" in prefix.lower() and re.search(r"(?<!\{)\{(?!\{)", s):
            return None
        if "r" not in prefix.lower():
            s = re.sub(r"\\(.)", lambda m: {"n": " ", "t": " ", "\\": "\\", "'": "'", '"': '"'}
                       .get(m.group(1), "\\" + m.group(1)), s)
        text.append(s)
    return "".join(text)


def _titles(code):
    """(kind, position, text or None) for each figure or axes title set in the code."""
    found = []
    for m in re.finditer(rf"\.(suptitle|set_title)\(((?:{LITERAL})+)", code):
        found.append((m.group(1), m.start(), _literal_text(re.findall(LITERAL, m.group(2)))))
    for m in re.finditer(rf"\btitle\s*=\s*((?:{LITERAL})+)", code):
        depth, k = 0, m.start()                  # the call this keyword belongs to
        while k > 0:
            k -= 1
            depth += {")": 1, "(": -1}.get(code[k], 0)
            if depth < 0:
                break
        call = re.search(r"(def\s+)?([\w.]+)\s*$", code[:k])
        if call and not call.group(1) and not call.group(2).endswith("legend"):
            found.append(("set_title", m.start(), _literal_text(re.findall(LITERAL, m.group(1)))))
    return sorted(found, key=lambda t: t[1])


def alt_text(code, n):
    """'Figure n: <title>', with the title read from the code that drew the figure (its
    suptitle, else its first axes title), or plain 'Figure n' when that is not a fixed string or
    when alternative branches (if/else) would give different titles."""
    titles = _titles(code)
    branch = re.search(r"^\s*(else\s*:|elif\b|except\b)", code, re.M)
    if branch and any(p < branch.start() for _, p, _ in titles) and any(
            p > branch.start() for _, p, _ in titles):
        return f"Figure {n}"
    sups = [t for kind, _, t in titles if kind == "suptitle"]
    if len(set(sups)) > 1:                       # two different figure titles: do not guess
        return f"Figure {n}"
    chosen = sups[0] if sups and sups[0] else next(
        (t for kind, _, t in titles if kind != "suptitle" and t), None)
    if not chosen:
        return f"Figure {n}"
    parts = re.split(r"(\$[^$]*\$)", chosen)
    title = "".join(tex_to_text(p[1:-1]) if p.startswith("$") and p.endswith("$") and len(p) > 1
                    else p for p in parts)
    title = re.sub(r"\s+", " ", title).strip(" :;,.–—-")
    if title.count("[") != title.count("]"):      # an unbalanced bracket would end the alt text
        title = title.replace("[", "(").replace("]", ")")
    return f"Figure {n}: {title}" if title else f"Figure {n}"


def figure_sources(src, n_png):
    """The code behind each of a cell's figures: the cell split after every plt.show(), or the
    whole cell for one figure. None when the split does not match the number of figures."""
    parts, cur = [], []
    for line in src.splitlines(keepends=True):
        cur.append(line)
        if re.search(r"\bplt\.show\(\)", line):
            parts.append("".join(cur)); cur = []
    if "".join(cur).strip() and (not parts or len(parts) < n_png):
        parts.append("".join(cur))
    if n_png == 1:
        return [src]
    return parts if len(parts) == n_png else None


# --- notebooks ------------------------------------------------------------------------------

def fence(lang, text):
    text = text.rstrip("\n")
    ticks = "````" if "```" in text else "```"
    return f"{ticks}{lang}\n{text}\n{ticks}"


def title_of(nb_path):
    """The notebook's '# N. Title' heading, or its file name."""
    for cell in json.loads(nb_path.read_text())["cells"]:
        if cell["cell_type"] == "markdown":
            for line in "".join(cell["source"]).splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
            break
    return nb_path.stem


def course_notebooks(folder):
    return sorted(folder.glob("[0-9][0-9]_*.ipynb"))


def convert(nb_path, out_dir):
    nb = json.loads(nb_path.read_text())
    stem = nb_path.stem
    fig_dir = out_dir / "figures" / stem
    for old in fig_dir.glob("fig-*.png"):        # a re-run must not leave stale figures behind
        old.unlink()
    parts, n_fig, fixes = [], 0, []

    for cell in nb["cells"]:
        src = "".join(cell["source"]).rstrip()
        if cell["cell_type"] == "markdown":
            if src:
                src, cell_fixes = fix_markdown(src)
                fixes += cell_fixes
                parts.append(src)
            continue
        if cell["cell_type"] != "code" or not src:
            continue
        parts.append(fence("python", src))
        n_png = sum("image/png" in o.get("data", {}) for o in cell.get("outputs", []))
        sources = figure_sources(src, n_png) if n_png else []
        k_png = 0

        stream_buf = []
        def flush():
            if stream_buf:
                text = ANSI.sub("", "".join(stream_buf))
                if not is_noise(text):
                    parts.append(fence("text", text))
                stream_buf.clear()

        for o in cell.get("outputs", []):
            kind = o["output_type"]
            if kind == "stream":
                text = "".join(o["text"])
                if o["name"] == "stderr" and is_noise(text):
                    continue
                stream_buf.append(text)
                continue
            flush()
            if kind == "error":
                parts.append(fence("text", ANSI.sub("", "\n".join(o.get("traceback", [])))))
                continue
            data = o.get("data", {})
            if "image/png" in data:
                n_fig += 1
                fig_dir.mkdir(parents=True, exist_ok=True)
                name = f"fig-{n_fig:02d}.png"
                (fig_dir / name).write_bytes(base64.b64decode("".join(data["image/png"])))
                alt = alt_text(sources[k_png], n_fig) if sources else f"Figure {n_fig}"
                parts.append(f"![{alt}](figures/{stem}/{name})")
                k_png += 1
                continue
            html = "".join(data.get("text/html", ""))
            if 'class="dataframe"' in html:
                md = html_table_to_md(html)
                if md:
                    parts.append(md)
                    continue
            plain = ANSI.sub("", "".join(data.get("text/plain", "")))
            if plain and not is_noise(plain):
                parts.append(fence("text", plain))
        flush()
    if fig_dir.is_dir() and not any(fig_dir.iterdir()):
        fig_dir.rmdir()

    siblings = course_notebooks(nb_path.parent)
    i = siblings.index(nb_path)
    prev = siblings[i - 1] if i > 0 else None
    nxt = siblings[i + 1] if i + 1 < len(siblings) else None
    nav = " · ".join(x for x in (
        f"← [{title_of(prev)}]({prev.stem}.md)" if prev else "",
        "[all notebooks](README.md)",
        f"[{title_of(nxt)}]({nxt.stem}.md) →" if nxt else "",
    ) if x)
    rel = Path(os.path.relpath(nb_path, out_dir)).as_posix()
    banner = (f"> Markdown edition of [`notebooks/{nb_path.name}`]({rel}). Code and outputs are "
              "from the notebook's last saved run; to experiment, run the notebook itself.\n>\n> "
              + nav)
    body = "\n\n".join(parts)
    first, _, rest = body.partition("\n")       # put the banner right after the H1 title
    if first.startswith("# "):
        text = f"{first}\n\n{banner}\n\n{rest.lstrip(chr(10))}"
    else:
        text = f"{banner}\n\n{body}"
    text = text.rstrip() + f"\n\n---\n\n{nav}\n"
    out = out_dir / f"{stem}.md"
    out.write_text(text)
    return out, n_fig, fixes


if __name__ == "__main__":
    targets = [Path(p).resolve() for p in sys.argv[1:]] or course_notebooks(NOTEBOOKS)
    for nb in targets:
        out, n, fixes = convert(nb, HERE)
        print(f"{out.relative_to(HERE.parent)}  ({out.stat().st_size // 1024} KB, {n} figures, "
              f"{len(fixes)} Markdown rewrites)")
