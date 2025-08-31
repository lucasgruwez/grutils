import re

from .equations import _convert_SI_units

def dash_postfix(content: str) -> str:
    """
    Handles latex --- and converts to an emdash.
    """
    return content.replace("---", "—")


def collapsible_proofs(content: str) -> str:
    """
    Turns proof callouts into collapsible callouts
    """
    return content.replace("[!proof]", "[!proof]-")


def math_postfix(content: str) -> str:
    """
    Some post-fixes for math
    """
    # Fix \qty|...| not working. Replace with \left|...\right|
    content = re.sub(r"\\qty\|(.+?)\|", r"\\left|\1\\right|", content)

    # Fix \eval_ not working. Replace with \bigg|
    content = re.sub(r"\\eval", r"\\bigg|", content)

    # Replace multiple oints with a single oint (e.g. \\oiint -> \\oint)
    content = re.sub(r"\\oi+nt", r"\\oint", content)

    # Trailing and leading whitespace on inline math
    matches = re.finditer(r"\$([^\n]+?)\$", content)

    for match in matches:
        inline_math = match.group(1)
        inline_math = inline_math.strip().lstrip()

        # SI units inside the inline math
        inline_math = _convert_SI_units(inline_math, s='')

        content = content.replace(match.group(0), f"${inline_math}$")

    return content


def si_units_postfix(content: str) -> str:
    """
    Fix latex si units
    """
    content = _convert_SI_units(content, s='$')
    return content


def quotes_postfix(content: str) -> str:
    """
    Fix latex quotes
    """
    return re.sub(r"``(.+?)''", r"“\1”", content)

def command_postfixes(content: str) -> str:
    """
    Remove any \\newcommand and \\renewcommands
    """
    # Remove custom latex command definitions
    content = re.sub(r'\\newcommand\{.*?\}\{.*?\}', '', content)
    content = re.sub(r'\\renewcommand\{.*?\}\{.*?\}', '', content)
    content = re.sub(r'\\definecolor\{.*?\}\{.*?\}\{.*?\}', '', content)
    return content

def misc_postfixes(content: str) -> str:
    """
    Apply miscellaneous post-processing to the content.
    """
    content = content.replace("\\newpage", "")
    return content

def lecture_handler(content: str) -> str:
    """
    Handles lecture notes title.
    """

    # Extract lecture title
    title_match = re.search(r"\\lecture{\d+}{(.+?)}", content)

    if title_match:
        lecture_tag = title_match.group(0)
        title = title_match.group(1).strip()

        content = content.replace(lecture_tag, f"# {title}")

        # Add cssclasses frontmatter
        content = "---\ncssclasses: lecture\n---\n" + content

    return content

def label_handler(content: str) -> str:
    """
    Handles \label{...} by removing them and the corresponding references.
    """
    content = re.sub(r"\\label\{(.+?)\}", "", content)
    content = re.sub(r"\\ref\{(.+?)\}", "", content)
    return content


TEX_HANDLERS = [
    dash_postfix,
    math_postfix,
    quotes_postfix,
    misc_postfixes,
    command_postfixes,
    collapsible_proofs,
    lecture_handler,
    label_handler,
    # SI units needs to happen last
    si_units_postfix,
]
