from typing import TypedDict
import argparse
import json
import re
import shutil
from pathlib import Path

import fitz
import pymupdf4llm
from tqdm import tqdm

def extract_text(pdf_path: Path) -> str:
    
    with fitz.open(pdf_path) as doc:
        md_text = pymupdf4llm.to_markdown(doc, ignore_images=True, ignore_graphics=True, ignore_code=True)

        md_text = re.sub(r"(?<!\n)\n(?!\n)", " ", md_text) # a single line break -> 1 whitespace
        md_text = re.sub(r"\n{2,}", "\n", md_text) # 2+ line breaks -> 1 line break
        md_text = re.sub(r"\s{2,}", " ", md_text) # 2+ whitespaces -> 1 whitespace
        md_text = re.sub(r"\n^[0-9]$\n", " ", md_text, flags=re.MULTILINE) # any lines that are just a single digit (page number)
    
    return md_text

class MINELike(TypedDict):
    essay:str
    answers:list[str]

def extract_pdf(source_pdf:Path) -> MINELike:
    extracted_pdf:MINELike = { "essay":f"```TITLE: {source_pdf.name}```\n\n{extract_text(source_pdf)}", "answers": []}
    return extracted_pdf

if __name__ == "__main__":
    # base directory for relative paths
    CWD = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("dataset_pdfs"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_output"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir: Path = (CWD / args.source_dir).resolve()
    output_dir: Path = (CWD / args.output_dir).resolve()
    should_overwrite: bool = args.overwrite

    if not source_dir.is_dir():
        raise RuntimeError(f"supplied directory path '{source_dir}' is not a valid directory")
    
    if output_dir.is_dir():
        if not should_overwrite:
            raise RuntimeError(f"supplied directory path '{output_dir}' already exists. Either remove it or use --overwrite = True")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(source_dir.iterdir(),desc="Dataset PDF extraction"):
        extracted_source:MINELike = extract_pdf(source_pdf=file_path)
        # write out to json in output_dir
        with open(output_dir/f"{file_path.stem}.json", "w") as fp:
            json.dump(extracted_source, fp, ensure_ascii=False, indent=4)
