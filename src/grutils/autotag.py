import os
import logging
import re
import sys
import json
import yaml
import ollama
import hashlib

from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Suppress common HTTP-related loggers that can be noisy
for _name in (
    "requests",
    "urllib3",
    "httpx",
    "requests.packages.urllib3",
    "ollama",
    "http.client",
):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.WARNING)
    _logger.addHandler(logging.NullHandler())


HEADING_RE = re.compile(r"^(#{1,6})\s*(.+)$", re.MULTILINE)
YAML_FRONT_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TAGS_RE = re.compile(r"\[(.*?)\]")

# Tags that are too broad or not useful
USELESS_WORDS = [
    "classification",
    "introduction",
    "theorem",
    "definition",
    "example",
    "calculation",
    "equation",
    "formula",
    "figure",
    "table",
    "proof",
    "supports",
    "diagram",
    "graph",
    "plot",
]


# ------------------------------------------------------------------------------
# File handling and parsing logic
# ------------------------------------------------------------------------------


def find_markdown_files(folder: str) -> List[str]:
    """
    Recursively find all markdown files in the given folder.
    """
    md_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".md"):
                md_files.append(os.path.join(root, f))
    return sorted(md_files)


def extract_toc_and_text(path: str) -> Tuple[List[str], str]:
    """
    Extract the table of contents (headings) and full text from a markdown file.
    """

    with open(path, "r", encoding="utf-8") as fh:
        txt = fh.read()
    headings = [m.group(2).strip() for m in HEADING_RE.finditer(txt)]
    return headings, txt


# ------------------------------------------------------------------------------
# Embedding logic
# ------------------------------------------------------------------------------


def embed_text(text: str, model: str = "nomic-embed-text:latest") -> List[float]:
    """
    Generate an embedding for the given text using the specified model.
    """
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


def file_hash(path: str) -> str:
    """
    Compute the SHA256 hash of the file content.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_cache(file: str, cache: Dict[str, Dict]) -> bool:
    """
    Check if the file is in the cache and if its hash matches the cached hash.
    """
    rel = os.path.relpath(file)
    hsh = file_hash(file)
    cached = cache.get(rel, {})
    return cached.get("hash") == hsh


def update_embeddings(
    folder: str,
    md_files: List[str],
    cache_file: str = "./file_cache.json",
    force: bool = False,
) -> Dict[str, List[float]]:
    """
    Update or create embeddings for markdown files in the specified folder.
    If a cache file exists, load it and update only missing or changed embeddings.
    Otherwise, create a new cache.
    """
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as fh:
            cache = json.load(fh)

    # Determine which files need embedding, based on cache and force flag
    if not force:
        unembedded_files = [f for f in md_files if check_cache(f, cache)]
    else:
        unembedded_files = md_files

    if len(unembedded_files) == 0:
        logging.info("All files are already embedded.")
        return cache

    # Embed files using ollama
    for f in tqdm(unembedded_files, desc="Embedding files"):
        rel = os.path.relpath(f, folder)
        hsh = file_hash(f)
        cached = cache.get(rel, {})
        if force or (not cached) or (cached.get("hash") != hsh):
            headings, txt = extract_toc_and_text(f)
            # emb = embed_text(txt) # Original, full text embedding
            emb = embed_text("\n".join(headings))  # Faster and often sufficient
            cache[rel] = {"hash": hsh, "embedding": emb, "headings": headings}

    # Save updated cache
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
        tqdm.write(f"Updated cache file: {os.path.abspath(cache_file)}")

    return cache


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ------------------------------------------------------------------------------
# Prompt generation logic
# ------------------------------------------------------------------------------


def generate_prompt(file: str, cache: Dict[str, List[float]]) -> str:
    """
    Generate a prompt for the given file.
    """
    if file not in cache:
        raise ValueError(f"File {file} not found in cache")

    target_headings = cache[file]["headings"]

    logging.debug(f"Headings for '{file}': {target_headings}")

    # Headings in the main input file
    prompt = f"The following are the headings from the file '{file}':\n"
    for h in target_headings:
        prompt += f"- {h}\n"

    # Instructions for tag generation
    prompt += (
        "\nBased on the above headings, suggest relevant tags for the original file."
        "\nThe goal of these tags is to link different notes with similar topics."
        "\n\nRules:"
        "\nProvide the tags as a json fromat, just one array with strings."
        "\nUse only lowercase letters, no spaces (use hyphens instead)."
        "\nGenerate up to 10 tags."
        "\nAvoid broad tags like 'physics' or 'math'."
        '\nExample output: ["tag1", "tag2", "tag3"]'
    )

    return prompt


def get_candidate_tags(
    file: str, cache: Dict[str, List[float]], model: str = "llama3.2:3b"
) -> List[str]:
    """
    Extract candidate tags from the folder name and its parent directories.
    """
    rel = os.path.normpath(file)
    prompt = generate_prompt(rel, cache)

    # Prompt ollama model to generate candidate tags
    response = ollama.generate(model=model, prompt=prompt)
    response = response["response"]

    # Search for json match in the response
    match = TAGS_RE.search(response)
    if match:
        tags_str = match.group(1)
        tags = json.loads(f"[{tags_str}]")
        return [clean_tag(tag) for tag in tags]
    else:
        return []


def clean_tag(tag: str) -> str:
    """
    Clean and format a tag string.
    """
    tag = tag.strip().lower()
    tag = re.sub(r"\s+", "-", tag)  # Replace spaces with hyphens
    tag = re.sub(
        r"[^a-z0-9\-]", "", tag
    )  # Remove non-alphanumeric characters except hyphens
    return tag


def remove_duplicates(tags: List[str]) -> List[str]:
    """
    Remove duplicate tags from the list while preserving order.
    """
    seen = set()
    unique_tags = []

    for tag in tags:
        if tag not in seen:
            # Also remove duplicates ignoring hyphens, i.e. linear-algebra and
            # linearealgebra are really the same
            seen.add(tag.replace("-", ""))
            unique_tags.append(tag)

    return unique_tags


# ------------------------------------------------------------------------------
# Find tags based on embeddings
# ------------------------------------------------------------------------------


def embed_tags(
    tags: List[str],
    model: str = "nomic-embed-text:latest",
    cache: str = "./tag_cache.txt",
    threshold: float = 0.9,
) -> Dict[str, List[float]]:
    """
    Generate embeddings for a list of tags.
    Tags saved in a text file. Re-embedding is quick, so for readability, we
    store just the tags and not their embeddings.
    """

    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as fh:
            cached_tags = fh.read().splitlines()
        tags.extend(cached_tags)
        logging.info(f"Loaded {len(cached_tags)} cached tags from {cache}")

    tag_embeddings = {}

    for tag in tqdm(tags, desc="Embedding tags"):
        if tag not in tag_embeddings:
            emb = embed_text(tag, model=model)
            tag_embeddings[tag] = emb

    # Remove tags that are too similar to each other
    logging.info("Removing similar tags...")
    unique_tags = {}
    # Sort by length, so that shorter tags are preferred
    sorted_tags = sorted(tag_embeddings.items(), key=lambda x: len(x))
    for tag, emb in sorted_tags:
        similar_tags = {
            existing_tag: cosine_similarity(emb, existing_emb)
            for existing_tag, existing_emb in unique_tags.items()
            if cosine_similarity(emb, existing_emb) >= threshold
        }
        if all(sim < threshold for sim in similar_tags.values()):
            unique_tags[tag] = emb
        else:
            # Find similar existing tag
            most_similar_tag = max(similar_tags, key=similar_tags.get)
            logging.warning(
                f"Tag '{tag}' is too similar to '{most_similar_tag}' and will be ignored."
            )

    # Remove useless words
    unique_tags = {
        tag: emb for tag, emb in unique_tags.items() if tag not in USELESS_WORDS
    }

    # Remove long tags
    unique_tags = {tag: emb for tag, emb in unique_tags.items() if len(tag) <= 24}

    with open(cache, "w", encoding="utf-8") as fh:
        for tag in sorted(unique_tags.keys()):
            fh.write(f"{tag}\n")

    logging.info(f"Updated cache file: {os.path.abspath(cache)}")

    return unique_tags


def find_tags(
    file: str,
    folder: str,
    tags: Dict[str, List[float]],
    cache: Dict[str, List[float]],
    top_n: int = 5,
    threshold: float = 0.6,
) -> List[Tuple[str, float]]:
    """
    Find the top N most relevant tags for a given file based on cosine similarity.
    """
    if file not in cache:
        raise ValueError(f"File {file} not found in cache")
    file_emb = cache[file]["embedding"]

    similarities = [
        (tag, cosine_similarity(file_emb, tag_emb)) for tag, tag_emb in tags.items()
    ]
    # Remove tags below the threshold
    similarities = [(tag, sim) for tag, sim in similarities]
    similarities.sort(key=lambda x: x[1], reverse=True)

    similarities = [(tag, sim) for tag, sim in similarities if sim >= threshold]

    return similarities[:top_n]


# ------------------------------------------------------------------------------
# File tagging
# ------------------------------------------------------------------------------


def add_tags(file: str, tags: List[str]):
    """
    Add the given tags to the markdown file's front matter.
    If no front matter exists, create one.
    """
    with open(file, "r", encoding="utf-8") as fh:
        content = fh.read()

    # Check for existing YAML front matter
    match = YAML_FRONT_RE.match(content)
    if match:
        front_matter = match.group(1)
        try:
            fm_data = yaml.safe_load(front_matter) or {}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML front matter in {file}: {e}")
            fm_data = {}

        fm_data["tags"] = tags
        new_front_matter = yaml.dump(fm_data, default_flow_style=False)

        front_matter = f"---\n{new_front_matter}---\n"
    else:
        # No front matter, create one
        new_front_matter = f"tags: [{', '.join(tags)}]"
        front_matter = f"---\n{new_front_matter}\n---\n{content}"

    if match:
        content = front_matter + content[match.end() :]
    else:
        content = front_matter + content

    with open(file, "w", encoding="utf-8") as fh:
        fh.write(content)

    logging.info(f"Added tags to {file}: {tags}")


if __name__ == "__main__":
    regenerate_candidates = False
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "/Users/lucasgruwez/Documents/Notes/05 - UNI"

    # Embed files and update cache
    md_files = find_markdown_files(folder)
    file_embeddings = update_embeddings(folder, md_files)

    Nfiles = len(md_files)

    tags = []

    # Generate a set of candidate tags. Takes about 5s per file with a 3B model.
    if regenerate_candidates:
        for f in tqdm(md_files[:Nfiles], desc="Generating candidate tags"):
            rel = os.path.relpath(f, folder)
            tags.extend(get_candidate_tags(rel, file_embeddings, model="llama3.2:3b"))

        tags = remove_duplicates(tags)

        logging.info(f"Generated {len(tags)} candidate tags.")

    # Embed tags. If no candidates were generated, it will still embed the
    # previously generated tags from the cache file.
    tag_embeddings = embed_tags(tags)

    # Find and add relevant tags to each file
    for f in md_files[:Nfiles]:
        rel = os.path.relpath(f, folder)
        relevant_tags = find_tags(
            rel, folder, tag_embeddings, file_embeddings, top_n=5, threshold=0.55
        )

        add_tags(f, [tag for tag, _ in relevant_tags])
