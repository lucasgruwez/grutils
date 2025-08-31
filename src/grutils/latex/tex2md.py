import re
import os
from typing import Callable, Dict

from handlers import ENV_HANDLERS, TEX_HANDLERS

# Define the handler type
Converter = Callable[[str], str]
EnvHandler = Callable[[str, Converter], str]


class TeX2Markdown:

    def __init__(self):
        """
        Initializes the instance with an empty dictionary to store environment
        handlers.

        Attributes
        ----------
        env_handlers (Dict[str, Callable]):
            A dictionary mapping environment names (as strings) to handler
            functions. Each handler function takes a string as input and
            returns a string.
        """
        self.env_handlers: Dict[str, EnvHandler] = ENV_HANDLERS.copy()
        self.tex_handlers: Dict[str, Converter] = TEX_HANDLERS.copy()

    def add_environment_handler(self, env_name: str, handler: EnvHandler):
        """
        Registers a custom handler function for a specific LaTeX environment.

        Parameters
        ----------
        env_name (str):
            Name of the LaTeX environment to handle
        handler (Callable[[str], str]):
            Function that takes the environment content as a string and returns
            a processed string.
        """

        # Add to list of handlers
        self.env_handlers[env_name] = handler

    def convert(self, tex: str) -> str:
        """
        Converts a LaTeX string to Markdown format.

        Parameters
        ----------
            tex (str): The input LaTeX string to be converted.

        Returns
        -------
            str: The converted Markdown string.
        """

        # Step 1: Handle environments
        def replace_env(match):
            env_name = match.group(1)
            content = match.group(2)

            # Fix indentation
            content = re.sub(r"\n\t", "\n", content)

            # Handle starred environments
            if env_name.endswith("*"):
                env_name = env_name[:-1]

            if env_name in self.env_handlers:
                return self.env_handlers[env_name](content, self)
            else:
                # default: just return the content
                return f"\n```{env_name}\n{content.strip()}\n```\n"
            
        # Step 1: resolve any inputs
        matches = re.finditer(r"\\input\{(.*?)\}", tex)
        for match in matches:
            input_file = match.group(1).strip()
            input_dir = os.path.dirname(self.tex_path)
            with open(os.path.join(input_dir, input_file), "r", encoding="utf-8") as f:
                input_content = f.read()
            tex = tex.replace(match.group(0), input_content)

        env_pattern = re.compile(r"\\begin\{(.*?)\}(.*?)\\end\{\1\}", re.DOTALL)
        tex = env_pattern.sub(replace_env, tex)

        # Step 2: Strip comments, but ignore escaped % (i.e., \%)
        tex = re.sub(r"(?<!\\)%.*", "", tex)

        # Step 3: Simple replacements for inline macros
        tex = re.sub(r'\\textbf{(.+?)}', r'**\1**', tex)
        tex = re.sub(r'\\textit{(.+?)}', r'*\1*', tex)
        tex = re.sub(r'\\texttt{(.+?)}', r'`\1`', tex)

        # Step 4: Remove remaining LaTeX commands like \section{Title}
        tex = re.sub(r'\\paragraph{}', '', tex)
        tex = re.sub(r'\\paragraph{(.+?)}', r'**\1**\n', tex)

        tex = re.sub(r"\\section\*?\{(.*?)\}", r"# \1", tex)
        tex = re.sub(r"\\subsection\*?\{(.*?)\}", r"## \1", tex)
        tex = re.sub(r"\\subsubsection\*?\{(.*?)\}", r"### \1", tex)

        # Replace multiple newlines with a single newline
        tex = re.sub(r'\n{2,}', '\n\n', tex)

        # Any additional handling that needs to be done
        for handler in self.tex_handlers:
            tex = handler(tex)

        return tex.strip()

    def convert_file(self, tex_path: str, out_path: str):
        """
        Convert a .tex file into a .md file.

        Parameters
        ----------
        tex_path (str):
            Path to the input .tex file
        out_path (str):
            Path to the output .md file
        """

        self.tex_path = tex_path
        self.out_path = out_path

        with open(tex_path, "r", encoding="utf-8") as f:
            tex = f.read()
        
        md = self.convert(tex)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    
    import os

    os.chdir(os.path.dirname(__file__))

    converter = TeX2Markdown()
    
    converter.convert_file("example.tex", "example.md")