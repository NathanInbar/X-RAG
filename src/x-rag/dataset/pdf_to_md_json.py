# Utility to create a json dataset from a folder of pdf files
import argparse
from pathlib import Path
import json

import fitz, pymupdf4llm
import re

def extract_text(pdf_path: Path) -> str:
    
    with fitz.open(pdf_path) as doc:
        md_text = pymupdf4llm.to_markdown(doc, ignore_images=True, ignore_graphics=True, ignore_code=True)

        md_text = re.sub(r"(?<!\n)\n(?!\n)", " ", md_text) # a single line break -> 1 whitespace
        md_text = re.sub(r"\n{2,}", "\n", md_text) # 2+ line breaks -> 1 line break
        md_text = re.sub(r"\s{2,}", " ", md_text) # 2+ whitespaces -> 1 whitespace
        md_text = re.sub(r"\n^[0-9]$\n", " ", md_text, flags=re.MULTILINE) # any lines that are just a single digit (figure breaks)
    
    return md_text


def process(input_folder: str, out_filename: str) -> None:
    input_folder = Path(input_folder)

    if not input_folder.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_folder}")

    out_path = Path(f"{out_filename}.json")

    with out_path.open("w", encoding="utf-8") as out_file:
        out_file.write("[\n")

        first = True
        for pdf in sorted(input_folder.glob("*.pdf")):
            obj = {
                "source": pdf.name,
                "text": extract_text(pdf),
            }

            if not first:
                out_file.write(",\n")
            first = False
                
            json.dump(obj, out_file, ensure_ascii=False)
            out_file.flush()

        out_file.write("\n]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a JSON dataset from a folder of PDF files."
    )
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_file", default="output_md.json")

    args = parser.parse_args()
    process(args.input_folder, args.output_file)


if __name__ == "__main__":
    main()