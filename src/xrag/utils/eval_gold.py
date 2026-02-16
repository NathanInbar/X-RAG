from xrag.paths import DATASETS_DIR, RESULTS_DIR, CACHE_DIR
import json
import time
import dspy
from aiolimiter import AsyncLimiter
from prettytable import PrettyTable
from xrag.utils import mg_driver
from xrag.config import config
from xrag.utils.eval import MINER
from xrag.utils import Tokenizer
from xrag.dataset_processing.preprocess import process_dataset_file
import litellm
import traceback
from statistics import mean

EMBED_MODEL = config.models["embed"]
EVAL_JUDGE_LM = dspy.LM(config.models["eval_judge"])

dspy.configure(lm=EVAL_JUDGE_LM)
### INSERT DSPY SIGNATURES HERE :

class _EvalSignature(dspy.Signature):
    """
    You are evaluating whether a retrieved context sufficiently supports a gold answer to a query.

    Given:
    - A query.
    - A gold_answer (this is the correct answer).
    - A retrieved context (this is the only information allowed).

    Determine whether the context contains enough information to support the gold_answer.

    Rules:
    1. Use ONLY the provided context. Do not use external knowledge.
    2. If the gold_answer represents "unanswerable", return True only if the context clearly lacks sufficient information to answer the query.
    3. For yes/no answers, return True only if the context clearly entails the gold yes/no decision.
    4. For free-form answers, return True only if the essential claims of the gold_answer are directly supported by the context.
    5. If the context is ambiguous, incomplete, or only partially supports the gold_answer, return False.
    6. If the gold_answer contains information not present in the context, return False.

    Output:
    - is_supported: True or False
    """

    query:str = dspy.InputField(desc="")
    gold_answer:str = dspy.InputField(desc="")
    context:str = dspy.InputField(desc="")

    is_supported:bool = dspy.OutputField(desc="")
    # unsupported_claims_count:int = dspy.OutputField()
###
pred_answer_eval = dspy.Predict(_EvalSignature)

async def miner_evaluate_with_gold_answers(name: str, miner: "MINER", dataset: str):
    dataset_dir = DATASETS_DIR / dataset
    paths = list(dataset_dir.iterdir())
    result_file = RESULTS_DIR / f"{name}.json"
    tmp_file = result_file.with_suffix(result_file.suffix + ".tmp")

    print(f"Writing results file to '{result_file}'")

    # ---- load existing results (cache) ----
    results_obj = {"name": name, "result": []}
    if result_file.is_file():
        try:
            with open(result_file, "r") as fp:
                loaded = json.load(fp)
            # accept either full object {"name":..., "result":[...]} or a bare list (older attempts)
            if isinstance(loaded, dict) and isinstance(loaded.get("result"), list):
                results_obj = loaded
                results_obj["name"] = name  # keep current run name
            elif isinstance(loaded, list):
                results_obj["result"] = loaded
        except Exception:
            pass

    # consider a file "done" if we already have an entry for it that is not an error
    done_files = {
        r.get("filename")
        for r in results_obj["result"]
        if isinstance(r, dict) and r.get("filename") and ("error" not in r)
    }

    def _atomic_save():
        # atomic-ish save (write tmp then replace)
        with open(tmp_file, "w") as fp:
            json.dump(results_obj, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
        tmp_file.replace(result_file)

    _atomic_save()

    # ---- evaluation loop ----
    for i, p in enumerate(paths):

        # caching skip
        if p.name in done_files:
            print(f"SKIP (cached): {p.name} ({i+1}/{len(paths)})")
            continue

        try:
            print(f"START EVAL: {p.name} ({i+1}/{len(paths)})")

            # Load data
            with open(p, "r") as fp:
                mine_data = json.load(fp)

            # Preprocess (only if missing)
            preprocessed_chunks = CACHE_DIR / f"{p.stem}__chunks.json"
            preprocessed_descs = CACHE_DIR / f"{p.stem}__g0_descriptions.json"
            if (not preprocessed_chunks.is_file()) or (not preprocessed_descs.is_file()):
                await process_dataset_file(p)

            # Ingest
            print("Ingesting...")
            ingest_st = time.time()
            await miner.ingest(preprocessed_chunks, preprocessed_descs)
            await miner.pre_retrieve(p.name)
            ingest_en = time.time()

            # Query + evaluate
            print("Evaluating...")
            query_results = []

            queries = mine_data.get("queries", [])
            gold_answers = mine_data.get("answers", [])

            for j, (query, gold_answer) in enumerate(zip(queries, gold_answers)):
                print(f"\rQuery {j+1}/{len(queries)}", end="")
                q_st = time.time()
                context = await miner.retrieve(query, preprocessed_chunks)
                q_en = time.time()
            
                ### INSERT QUERY EVALUATION ALGORITHM HERE:
                ### (NOTE:) TO TOKENIZE: CALL Tokenizer.encode(my_string)
                ### (NOTE:) TO EMBED: CALL resp = await litellm.aembedding(model=EMBED_MODEL, input = my_string or for batch [str1,str2,...]) THEN USE resp['data]

                eval = await pred_answer_eval.acall(query=query, context=context, gold_answer=gold_answer['text'])

                query_results.append(
                    {
                        "query": query,
                        "context": context,
                        "duration": q_en - q_st,
                        "contained": eval.is_supported # should be 'supported' but this is temp for compatibility with old evaluate()
                    }
                )
                print("")

            result = {
                "filename": p.name,
                "ingest_duration": ingest_en - ingest_st,
                "queries": query_results,
            }

            await miner.reset()

        except Exception as e:
            tb = e.__traceback__
            last = traceback.extract_tb(tb)[-1]
            result = {"filename": p.name, "error": f"{last.filename}:{last.lineno} | {type(e).__name__}: {e}"}
            print(f"ERROR: {str(e)}")
            try:
                await miner.reset()
            except Exception:
                pass

        # persist + update cache
        results_obj["result"].append(result)
        if "error" not in result:
            done_files.add(p.name)
        _atomic_save()
        print(f"wrote result to {result_file}")



############## EVAL IDEA:
    # class GenerateAnswerWithContext(dspy.Signature):
    #     """  """
    #     query:str = dspy.InputField()
    #     context:str = dspy.InputField()

    #     is_unanswerable:bool = dspy.OutputField()
    #     answer_type:str = dspy.OutputField()
    #     yes_no:bool|None = dspy.OutputField()
    #     free_form:str = dspy.OutputField()
    #     rationale_quote:str = dspy.OutputField()

    # dspy_generate_answer = dspy.Predict(GenerateAnswerWithContext)
    # TAU = ... #similarity thresh

# . . . . . . . . . . . .

    # ctx_toks = Tokenizer.encode(context)
    # len_ctx_chars, ctx_toks = len(context), len(ctx_toks)

    # # Evidence coverage metrics:
    # ctx_segments: list[str] = ... # split on blank lines or sliding token window
    # evidence: list[str] = gold_answer['evidence']

    # ## embed context segments and evidence paragraphs
    # resp = await litellm.aembedding(model=EMBED_MODEL, input=ctx_segments)
    # ctx_segment_embeddings = resp['data']

    # resp = await litellm.aembedding(model=EMBED_MODEL, input=evidence)
    # evidence_embeddings = resp['data']

    # ## max similarity per evidence paragraph
    # sim_ej = ...
    # n_hits:int = ...
    # ###hit if sim_ej >= tau

    # ## metrics calculated:
    # evidence_recall_sem = n_hits / len(evidence)
    # evidence_sim_mean = mean(sim_ej)
    # evidence_sim_min = min(sim_ej)



    # # Generate Predicted Answer
    # pred_answer = await dspy_generate_answer.acall(query=query, context=context)

    # ## type-specific answer scoring
    # gold_ans_type = gold_answer['answer_type']
    # if gold_ans_type == "unanswerable":
    #     ...
    # elif gold_ans_type == "yes_no":
    #     ...
    # elif gold_ans_type == "free_form":
    #     ...
