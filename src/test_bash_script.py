from time import sleep
import json
from xrag.paths import RESULTS_DIR
from pathlib import Path
if __name__ == "__main__":

    sleep(5)

    outfile = RESULTS_DIR / "test_bash.json"
    result = {"It...": "...Worked!"}
    with open(outfile, "w") as f:
        json.dump(result, f)
