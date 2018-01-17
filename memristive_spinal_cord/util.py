import definitions
import pathlib
import shutil


def clean_previous_results():
    pathlib.Path(definitions.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(definitions.RESULTS_DIR + '/img', ignore_errors=True)
    from pathlib import Path
    for p in Path(definitions.RESULTS_DIR).glob("*.dat"):
        p.unlink()
    for p in Path(definitions.RESULTS_DIR).glob("*.gdf"):
        p.unlink()
