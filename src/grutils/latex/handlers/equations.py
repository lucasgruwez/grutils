import re


def _convert_SI_units(content, s="$"):
    """
    Convert LaTeX SI unit commands to Markdown-friendly format.
    """

    # SI units package is not supported in Markdown, so we will replace it
    # \SI{1.38e-23}{\J \per \K} -> 1.38 \times 10^{-23} \mathrm{J / K}
    # \SI{0}{\K} -> 0 \mathrm{K}
    # \SI{9.81}{\m \per \s\squared} -> 9.81 \mathrm{m / s^2}
    for match in re.finditer(r"\\SI\{(.*?)\}\{(.*?)\}", content):
        value = match.group(1).strip()
        unit = match.group(2).strip()
        # Replace units syntax
        unit = unit.replace(r"\per", "/")
        unit = unit.replace(r"\squared", "^2")
        unit = unit.replace(r"\cubed", "^3")
        # Remove backslashes from units
        unit = re.sub(r"\\([a-zA-Z]+)", r"\1", unit)
        # Replace LaTeX scientific notation with Markdown
        if "e" in value or "E" in value:
            value = (
                value.replace("e", r"\times 10^{").replace("E", r"\times 10^{") + "}"
            )
        else:
            value = value.strip()
        # Replace the SI command with Markdown format
        content = content.replace(match.group(0), f"{s}{value} \\mathrm{{{unit}}}{s}")

    # SI units without value, e.g. \si{K} -> \\mathrm{K}
    for match in re.finditer(r"\\si\{(.*?)\}", content):
        unit = match.group(1).strip()
        # Replace units syntax
        unit = unit.replace(r"\per", "/")
        unit = unit.replace(r"\squared", "^2")
        unit = unit.replace(r"\cubed", "^3")
        # Remove backslashes from units
        unit = re.sub(r"\\([a-zA-Z]+)", r"\1", unit)
        # Replace the SI command with Markdown format
        content = content.replace(match.group(0), f"{s}\\mathrm{{{unit}}}{s}")

    # Values without units, e.g. \num{1.38e-23} -> 1.38 \times 10^{-23}
    for match in re.finditer(r"\\num\{(.*?)\}", content):
        value = match.group(1).strip()
        # Replace scientific notation with Markdown format
        if "e" in value or "E" in value:
            value = (
                value.replace("e", r"\times 10^{").replace("E", r"\times 10^{") + "}"
            )
        else:
            value = value.strip()
        # Replace the num command with Markdown format
        content = content.replace(match.group(0), f"{s}{value}{s}")

    return content


def equation_handler(content: str, converter) -> str:
    """Block math (MathJax style)."""
    content = _convert_SI_units(content, s="")
    return f"$$\n{content.strip()}\n$$\n"


def align_handler(content: str, converter) -> str:
    """Multiline math block (align)."""
    # In Markdown/MathJax, align is usually just block math with \\.
    start = "$$\\begin{aligned}"
    end = "\\end{aligned}$$"

    content = _convert_SI_units(content, s="")

    return f"{start}\n{content.strip()}\n{end}\n"

def multiline_handler(content: str, converter) -> str:
    """Multiline math block (align)."""
    content = _convert_SI_units(content, s="")
    return f"$$\n{content.strip()}\n$$\n"

def inline_math_postfix(content: str) -> str:
    """Inline math (MathJax style)."""
    content = _convert_SI_units(content, s="")
    return f"${content.strip()}$"


ENV_HANDLERS = {
    "equation": equation_handler,
    "align": align_handler,
}
