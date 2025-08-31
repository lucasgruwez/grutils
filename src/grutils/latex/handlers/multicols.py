import re

def multicols_handler(content: str, converter) -> str:

    # Remove first line
    content = "\n".join(content.split("\n")[1:])

    # Remove any custom column tags
    content = content.replace("\\raggedcolumns", "")
    content = content.replace("\\columnbreak", "")

    # Recursion
    content = converter.convert(content)

    return content

def nopagebreak_handler(content: str, converter) -> str:
    return converter.convert(content)


ENV_HANDLERS = {
    "multicols": multicols_handler,
    "absolutelynopagebreak": nopagebreak_handler
}