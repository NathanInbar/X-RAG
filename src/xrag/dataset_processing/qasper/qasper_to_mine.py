import argparse
import json
from datasets import load_dataset
from pprint import pprint
from paths import DATASETS_DIR
from enum import Enum
from pydantic import BaseModel, field_validator
from tqdm import tqdm

OUTPUT_DIR = DATASETS_DIR / "QASPER"
if not OUTPUT_DIR.is_dir():
    OUTPUT_DIR.mkdir()


class FullText(BaseModel):
    section_name: list[str]
    paragraphs: list[list[str]]

    @field_validator("section_name", mode="before")
    @classmethod
    def fill_missing_section_names(cls, v):
        if v is None:
            return []
        return ["" if name is None else name for name in v]

class NLPBackground(str,Enum):
    ZERO = "zero"
    TWO = "two"
    FIVE = "five"
    INF = "infinity"

class TopicBackground(str,Enum):
    UNFAMILIAR = "unfamiliar"
    FAMILIAR = "familiar"
    RESEARCH = "research"

class PaperRead(str,Enum):
    YES = "yes"
    NO = "no"
    SOMEWHAT = "somewhat"

def empty_string_to_none(v):
    return None if v == "" else v

class Answer(BaseModel):
    unanswerable: bool

    # exactly one of these three will be non-empty (if answerable)
    extractive_spans: list[str]
    yes_no: bool | None
    free_form_answer: str | None

    evidence: list[str]  # full paragraph
    highlighted_evidence: list[str]  # highlights from those paragraphs

    @field_validator("yes_no", "free_form_answer", mode="before")

    @classmethod
    def normalize_empty(cls, v):
        return empty_string_to_none(v)

class AnswerContainer(BaseModel):
    annotation_id: list[str]
    answer: list[Answer]
    worker_id: list[str]

class QAs(BaseModel):
    question:list[str]
    question_id:list[str]
    nlp_background:list[NLPBackground|None]
    topic_background:list[TopicBackground|None]
    paper_read:list[PaperRead|None]
    search_query:list[str | None]
    question_writer:list[str]
    answers:list[AnswerContainer]

    @field_validator("nlp_background", "topic_background", "paper_read", "search_query", mode="before")
    @classmethod
    def normalize_empty(cls, v):
        if v is None:
            return None
        return [empty_string_to_none(item) for item in v]


class FiguresAndTables(BaseModel):
    caption: list[str]
    file: list[str]

class QasperRow(BaseModel):
    id:str
    title:str
    abstract:str
    full_text:FullText
    qas: QAs
    figures_and_tables:FiguresAndTables

# - - - - - - -  - - - - - - -  - - - - - - - 

class AnswerType(str, Enum):
    UNANSWERABLE = "unanswerable"
    FREE_FORM = "free_form"
    EXTRACTIVE_SPANS = "extractive_spans"
    YES_NO = "yes_no"


class SimplifiedAnswer(BaseModel):
    answer_type: AnswerType
    text: str | None
    evidence: list[str]
    highlighted_evidence: list[str]


class QasperMineLike(BaseModel):
    essay:str
    queries:list[str]
    answers:list[SimplifiedAnswer]


def answer_container_to_simple(container: AnswerContainer) -> SimplifiedAnswer:
    """Convert all annotations for a question into a single simplified answer."""

    for answer in container.answer:
        if answer.unanswerable:
            return SimplifiedAnswer(
                answer_type=AnswerType.UNANSWERABLE,
                text=None,
                evidence=answer.evidence,
                highlighted_evidence=answer.highlighted_evidence,
            )
        if answer.free_form_answer:
            return SimplifiedAnswer(
                answer_type=AnswerType.FREE_FORM,
                text=answer.free_form_answer,
                evidence=answer.evidence,
                highlighted_evidence=answer.highlighted_evidence,
            )
        if answer.extractive_spans:
            return SimplifiedAnswer(
                answer_type=AnswerType.EXTRACTIVE_SPANS,
                text="\n".join(answer.extractive_spans),
                evidence=answer.evidence,
                highlighted_evidence=answer.highlighted_evidence,
            )
        if answer.yes_no is not None:
            normalized = "yes" if answer.yes_no else "no"
            return SimplifiedAnswer(
                answer_type=AnswerType.YES_NO,
                text=normalized,
                evidence=answer.evidence,
                highlighted_evidence=answer.highlighted_evidence,
            )
    return SimplifiedAnswer(
        answer_type=AnswerType.UNANSWERABLE,
        text=None,
        evidence=[],
        highlighted_evidence=[],
    )

def flatten_article_content(full_text:FullText) -> str:
    """ flatten a QASPER article entry into a single string """

    content = ""
    for section_name, paragraph in zip(full_text.section_name, full_text.paragraphs):
        # section header:
        # count number of ':::' -> how many hashes to prepend
        sep_count = section_name.count(':::')
        content += (sep_count+1) * '#'
        content += " " + section_name.split(':::')[-1] + "\n"

        content += "".join(paragraph)
        content += "\n"*2

    return content

def qasper_row_to_mine(row:QasperRow) -> QasperMineLike:

    essay_content = flatten_article_content(row.full_text)
    
    # NOTE: possible to filter questions by background experience , paper read, etc
    queries:list[str] = row.qas.question 
    answers:list[SimplifiedAnswer] = [answer_container_to_simple(x) for x in row.qas.answers]

    if len(queries) != len(answers):
        raise RuntimeError(f"Number of queries != number of answers for row: {row.id}")

    out = QasperMineLike(
        essay=essay_content,
        queries=queries,
        answers=answers,
    )

    return out



def main(ignore_cache: bool = False):
    ds = load_dataset("allenai/qasper", split="validation")
    for raw in tqdm(ds):
        row = QasperRow.model_validate(raw)

        normalized_id = row.id.replace(".", "_")
        output_path = OUTPUT_DIR / f"{normalized_id}.json"
        if output_path.exists() and not ignore_cache:
            continue

        minelike_article = qasper_row_to_mine(row)

        with output_path.open("w") as f:
            json.dump(minelike_article.model_dump(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert QASPER rows into simplified JSON files")
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Reprocess all rows even if cached JSON already exists",
    )
    args = parser.parse_args()
    main(ignore_cache=args.ignore_cache)
