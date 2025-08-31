import os
import re
import sys
import shutil

import tex2md

# ------------------------------------------------------------------------------
# Converter parameters
# ------------------------------------------------------------------------------

IN_PATH = rf"/Users/lucasgruwez/Documents/UNI"
OUT_PATH = rf"/Users/lucasgruwez/Documents/Notes/05 - Uni"

# Clear notes before converting
CLEAR = False

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def clean(title):
    """
    Clean the title by removing LaTeX commands and special characters.
    """
    # Swap : for -
    title = re.sub(r"\s*:\s*", " - ", title)
    # Remove LaTeX commands
    title = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", title)
    # Remove special characters
    title = re.sub(r"[^\w\s-]", "", title)
    # Strip whitespace
    return title.strip()


# ------------------------------------------------------------------------------
# Find all units.
# ------------------------------------------------------------------------------

# File structure is as follows:
# | Y1S1                               # Semester folder
# +--| ENG1005                         # Unit folder
# |  +--| lec_01.tex                   # Lecture file
# |  +--| lec_02.tex
# |  +--|...
# +--| PHS1011
# +--|...
# | Y1S2
# +--| ENG1001
# +--| ENG1002
# +--|...
# | Y2S1
# ...

sem_regex = re.compile(r"Y(\d)S(\d)")
unit_regex = re.compile(r"([A-Z]{3}\d{4})")
lecture_regex = re.compile(r"lec_(\d+)\.tex")

# Get all units
lectures = []

for semester in filter(lambda d: sem_regex.match(d), os.listdir(IN_PATH)):
    semester_path = os.path.join(IN_PATH, semester)

    for unit in filter(lambda d: unit_regex.match(d), os.listdir(semester_path)):
        unit_path = os.path.join(semester_path, unit)
    
        for lecture in filter(lambda f: lecture_regex.match(f), os.listdir(unit_path)):
            lectures.append(os.path.join(semester_path, unit, lecture))

# Make sure the notes directory exists and is empty
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
elif CLEAR:
    # Clear the notes directory
    for item in os.listdir(OUT_PATH):
        item_path = os.path.join(OUT_PATH, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

lectures.sort()  # Sort lectures to ensure consistent order


# ------------------------------------------------------------------------------
# Convert units into markdown
# ------------------------------------------------------------------------------

converter = tex2md.TeX2Markdown()

# For each unit, copy the lecture to the notes directory and change to markdown
for lecture in lectures:  # Limit to first 12 lectures for testing

    # Find unit code
    unit_code_match = re.search(unit_regex, lecture)
    unit_code = unit_code_match.group(1)

    # Find unit name
    master_file = os.path.join(os.path.dirname(lecture), "master.tex")
    with open(master_file, "r", encoding="utf-8") as file:
        master_content = file.read()
        unit_name_match = re.search(r"\\unit{.+?}{(.+?)}", master_content)
        unit_name = clean(unit_name_match.group(1))

    # Read lecture file
    with open(lecture, "r", encoding="utf-8") as file:
        content = file.read()

    # Find title
    title_match = re.search(r"\\lecture{\d+}{(.+?)}", content)
    title = clean(title_match.group(1))

    print(f"Unit: {unit_code}, Title: {title}")

    lecture_number = re.search(lecture_regex, lecture).group(1)

    # Ensure the unit directory exists
    unit_dir = os.path.join(OUT_PATH, f"{unit_code} - {unit_name}")
    if not os.path.exists(unit_dir):
        os.makedirs(unit_dir)

    # Convert to markdown
    target_path = os.path.join(unit_dir, f"{lecture_number} - {title}.md")

    if not os.path.exists(target_path):
        print(f"Converting {lecture}...")
        converter.convert_file(lecture, target_path)
    else:
        print(f"Skipping {lecture}, already converted.")
