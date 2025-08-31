import os
import re
import shutil
import hashlib
import subprocess

standalone_tex = """
\\documentclass[border=2pt]{standalone}
\\input{$PREAMBLE$}
\\begin{document}
\\begin{minipage}{15cm}
\\begin{figure}[H]
$CONTENT$
\\end{figure}
\\end{minipage}
\\end{document}
"""


def figure_handler(content: str, converter) -> str:
    """
    Convert a LaTeX figure environment into an embedded SVG.
    """

    tex_dir = os.path.dirname(converter.tex_path)
    fig_tex = os.path.join(tex_dir, "fig.tex")
    fig_pdf = os.path.join(tex_dir, "fig.pdf")
    fig_svg = os.path.join(tex_dir, "fig.svg")

    # Preamble located in same folder as this file.
    preamble = os.path.join(os.path.dirname(__file__), "preamble.tex")

    # Remove first line of content
    content = "\n".join(content.split("\n")[1:])

    # Remove any captions
    content = re.sub(r"\\caption\{[^\n]*\}", "", content)

    tex = standalone_tex.replace("$PREAMBLE$", preamble)[1:]
    tex = tex.replace("$CONTENT$", content)

    with open(fig_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    # Run LaTeX -> PDF
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-synctex=0", os.path.basename(fig_tex)],
            cwd=tex_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"**[Error rendering figure: {e}]**")

    # Convert PDF -> SVG
    try:
        subprocess.run(
            ["pdf2svg", fig_pdf, fig_svg],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"**[Error converting figure to SVG: {e}]**")
        return

    # Copy the SVG to the output directory
    out_dir = os.path.dirname(converter.out_path)
    out_dir = os.path.join(out_dir, "99 - Attachments")

    # Use hashing to ensure unique figure names
    tex_name = os.path.basename(converter.tex_path).replace(".tex", "")
    hashed = hashlib.sha256(content.encode()).hexdigest()
    out_name = tex_name + '_' + hashed[:6] + ".svg"

    try:
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(fig_svg, os.path.join(out_dir, out_name))
    except Exception as e:
        print(f"**[Error copying SVG to output directory: {e}]**")
        return

    # Remove any temporary files
    try:
        # List all files in tex_dir
        for f in os.listdir(tex_dir):
            if f.startswith("fig."):
                if "busy" in f: continue
                os.remove(os.path.join(tex_dir, f))
    except Exception as e:
        print(f"**[Error removing temporary files: {e}]**")

    fig_path = os.path.join(out_name)

    return f"![Figure]({fig_path})"

ENV_HANDLERS = {
    "figure": figure_handler,
}
