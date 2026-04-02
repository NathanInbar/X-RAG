from time import sleep
from xrag.paths import RESULTS_DIR
from pathlib import Path
if __name__ == "__main__":

    sleep(5)

    outfile = RESULTS_DIR / "test_bash.json"
    with open(outfile, "w") as f:
        f.write("Testing...")
