from datasets import load_dataset
from pprint import pprint
from paths import DATASETS_DIR
from enum import Enum
from pydantic import BaseModel, field_validator

OUTPUT_DIR = DATASETS_DIR / "QASPER"
if not OUTPUT_DIR.is_dir():
    OUTPUT_DIR.mkdir()


class FullText(BaseModel):
    section_name: list[str]
    paragraphs: list[list[str]]

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


def flatten_article_content(full_text:FullText) -> str:
    """ flatten a QASPER article entry into a single string """

    content = ""
    for section_name, paragraph in zip(full_text['section_name'], full_text['paragraphs']):
        # section header:
        # count number of ':::' -> how many hashes to prepend
        sep_count = section_name.count(':::')
        content += (sep_count+1) * '#'
        content += " " + section_name.split(':::')[-1] + "\n"

        content += "".join(paragraph)
        content += "\n"*2

    return content

def qasper_row_to_mine(row):
    ...


ds = load_dataset("allenai/qasper", split="validation")
for raw in ds:

    #qas.answers.0.unanswerable -> missing
    #qas.answers.0.extractive_spans -> missing
    #... yes_no, free_form_answer, evidence, highlighted evidence

    # print(raw['qas']['answers'][0])

    # break 

    row = QasperRow.model_validate(raw)
    print(row)

    break #TEMP: stop at ds0

# TODO: 
# - row -> QasperRow
# - flatten article content and create mine-like json object
# - save it to the output directory
# -- caching: (only for id.json not in dataset dir)


# pprint(ds0['full_text'])
# print(len(ds0['full_text']['section_name']))
# print(len(ds0['full_text']['paragraphs']))

# ds0 = ds[0]
