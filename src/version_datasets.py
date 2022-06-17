import os
import glob
import subprocess
import shutil

DATASET_DIR = "/workspaces/us-patent-phrase-to-phrase-matching/datasets"

EXPERIMENT_ID = "2"
MLFLOW_DIR = f"/workspaces/us-patent-phrase-to-phrase-matching/mlruns/{EXPERIMENT_ID}"
RUN_IDS = [
    "dc95c61f15bf4ac0b9e1de3ac1299f45",
    "9aa398aea22c4048b4e904c36bb3605b",
    "7923146adde64b6a9643eb163d8d223f",
    "bc8145deac9d4376b708e8b1499da970",
    "b977b33e7f8547bcb1e62e42e021e627",
    "e38625b57d3e4f00992c4f191a786c5d",
]

# make zips
for run_id in RUN_IDS:
    print("#" * 30, run_id, "#" * 30)

    if os.path.exists(os.path.join(DATASET_DIR, run_id + ".zip")):
        continue

    shutil.make_archive(
        os.path.join(DATASET_DIR, run_id),
        format="gztar",
        root_dir=os.path.join(MLFLOW_DIR, run_id, "artifacts"),
    )

# kaggle api
os.chdir(DATASET_DIR)
subprocess.run("kaggle datasets version -m 'default'", shell=True)
