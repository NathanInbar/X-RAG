# Utility to create a json dataset from a folder of pdf files
import argparse
from pathlib import Path
import json

import fitz

def extract_text(pdf_path: Path) -> str:
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)


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
            json.dump(obj, out_file, ensure_ascii=False)
            out_file.flush()
            first = False

        out_file.write("\n]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a JSON dataset from a folder of PDF files."
    )
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--out_filename", default="output")

    args = parser.parse_args()
    process(args.input_folder, args.out_filename)


if __name__ == "__main__":
    main()