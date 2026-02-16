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

EMBED_MODEL = config.models["embed"]


### INSERT DSPY SIGNATURES HERE :

class ExampleSignature(dspy.Signature):
    ...


###

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

            with dspy.context(lm=config.models["eval_judge"]):
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



                    ###

                    query_results.append(
                        {
                            "query": query,
                            "context": context,
                            "duration": q_en - q_st,

                            "...":..., # LOG MORE METRICS HERE IF NEEDED
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
            result = {"filename": p.name, "error": str(e)}
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
