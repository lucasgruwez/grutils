import re
from functools import partial

def _callout_handler(kind: str, content: str, converter) -> str:
    """
    Shared handler for Obsidian flavored callouts (define, theorem, etc).
    """

    # Search for title in the first line, greedily matching {...}
    first_line, *rest = content.split("\n", 1)
    title_match = re.match(r"\s*{(.*)}\s*$", first_line)
    title = title_match.group(1).strip() if title_match else ""
    body = rest[0] if rest else ""

    converted = converter.convert(body).strip()

    result = f"> [!{kind}] {title}\n" if title else f"> [!{kind}] \u200D\n"
    result += "\n".join(f"> {line.strip()}" for line in converted.splitlines())

    return result + "\n"


_environments = [
    "define",
    "theorem",
    "lemma",
    "corollary",
    "law",
    "prop",
    "remark",
    "example",
    "proof",
]

ENV_HANDLERS = {env: partial(_callout_handler, env) for env in _environments}
