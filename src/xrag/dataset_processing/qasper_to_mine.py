from datasets import load_dataset
from pprint import pprint
from paths import DATASETS_DIR
from enum import Enum

OUTPUT_DIR = DATASETS_DIR / "QASPER"
if not OUTPUT_DIR.is_dir():
    OUTPUT_DIR.mkdir()

ds = load_dataset("allenai/qasper", split="validation")

ds0 = ds[0]

from pydantic import BaseModel

class FullText(BaseModel):
    section_name: list[str]
    paragraphs: list[str]

class NLPBackground(str,Enum):
    ZERO = "zero"
    TWO = "two"
    INF = "infinity"

class TopicBackground(str,Enum):
    UNFAMILIAR = "unfamiliar"
    FAMILIAR = "familiar"
    RESEARCH = "research"

class PaperRead(str,Enum):
    YES = "yes"
    NO = "no"

class Answer(BaseModel):
    unanswerable:bool

    # exactly one of these three will be non-empty (if answerable)
    extractive_spans:list ### spans in the paper which serve as the answer
    yes_no: bool|None
    free_form_answer: str
    
    evidence: list[str] # full paragraph
    highlighted_evidence: list[str] # highlights from those paragraphs

class QAs(BaseModel):
    question:list[str]
    question_id:list[str]
    nlp_background:list[NLPBackground]
    topic_background:list[TopicBackground]
    paper_read:list[PaperRead]
    search_query:list[str]
    question_writer:list[str]
    answers:list[Answer]


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

    
# TODO: 
# - row -> QasperRow
# - flatten article content and create mine-like json object
# - save it to the output directory
# -- caching: (only for id.json not in dataset dir)



# pprint(ds0['full_text'])
# print(len(ds0['full_text']['section_name']))
# print(len(ds0['full_text']['paragraphs']))
