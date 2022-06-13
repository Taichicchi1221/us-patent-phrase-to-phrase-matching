import os
import glob
import subprocess
import shutil

DATASET_DIR = "/workspaces/us-patent-phrase-to-phrase-matching/datasets"

EXPERIMENT_ID = "2"
MLFLOW_DIR = f"/workspaces/us-patent-phrase-to-phrase-matching/mlruns/{EXPERIMENT_ID}"
RUN_IDS = [
    "7923146adde64b6a9643eb163d8d223f",
    "bc8145deac9d4376b708e8b1499da970",
    "b977b33e7f8547bcb1e62e42e021e627",
]

# clear dataset direcroty
for path in glob.glob(os.path.join(DATASET_DIR, "*")):
    if os.path.isdir(path):
        shutil.rmtree(path)

# copy results
for run_id in RUN_IDS:
    shutil.copytree(os.path.join(MLFLOW_DIR, run_id), os.path.join(DATASET_DIR, run_id))

os.chdir(DATASET_DIR)
subprocess.run("kaggle datasets version --dir-mode zip -m 'default'", shell=True)
