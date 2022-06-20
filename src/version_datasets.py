import os
import glob
import subprocess
import shutil

UPLOAD = False

DATASET_DIR = "/workspaces/us-patent-phrase-to-phrase-matching/datasets"
# DATASET_DIR = "/workspaces/us-patent-phrase-to-phrase-matching/datasets/uspppm-public"

EXPERIMENT_ID = "2"
MLFLOW_DIR = f"/workspaces/us-patent-phrase-to-phrase-matching/mlruns/{EXPERIMENT_ID}"

RUN_IDS = [
    "018420c68d63460f9ad625d51bb3903f",
]

# make zips
for run_id in RUN_IDS:
    print("#" * 30, run_id, "#" * 30)

    if os.path.exists(os.path.join(DATASET_DIR, run_id + ".zip")):
        continue

    shutil.make_archive(
        os.path.join(DATASET_DIR, run_id),
        format="zip",
        root_dir=os.path.join(MLFLOW_DIR, run_id, "artifacts"),
    )

# kaggle api
if UPLOAD:
    os.chdir(DATASET_DIR)
    subprocess.run("kaggle datasets version -r skip -m 'default'", shell=True)
