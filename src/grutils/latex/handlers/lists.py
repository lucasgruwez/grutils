import re


def itemize_handler(content: str, converter) -> str:
    """Convert LaTeX itemize to Markdown bullets."""
    # Convert items: \item Something -> - Something
    items = content.split(r"\item ")[1:]  # Skip the first empty split
    items = [item for item in items if item.strip()]  # Remove empty items

    for i, item in enumerate(items):
        items[i] = "- " + item.strip().replace("\n\t", "\n")
        items[i] = converter.convert(items[i])
        items[i] = re.sub(r"\n", "\n\t", items[i])  # indent sub-lines

    return "\n".join(items)

def enumerate_handler(content: str, converter) -> str:
    """Convert LaTeX enumerate to Markdown numbered list."""
    # Convert items: \item Something -> - Something
    items = content.split(r"\item ")[1:]  # Skip the first empty split
    items = [item for item in items if item.strip()]  # Remove empty items

    for i, item in enumerate(items):
        items[i] = f"{i+1}. " + item.strip().replace("\n\t", "\n")
        items[i] = converter.convert(items[i])
        items[i] = re.sub(r"\n", "\n\t", items[i])  # indent sub-lines
        
    return "\n".join(items)


ENV_HANDLERS = {
    "itemize": itemize_handler,
    "enumerate": enumerate_handler,
}
