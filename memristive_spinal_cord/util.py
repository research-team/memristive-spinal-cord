import definitions

def clean_previous_results():
    import shutil
    shutil.rmtree(definitions.RESULTS_DIR + '/img', ignore_errors=True)
    from pathlib import Path
    for p in Path(definitions.RESULTS_DIR).glob("*.dat"):
        p.unlink()
    for p in Path(definitions.RESULTS_DIR).glob("*.gdf"):
        p.unlink()