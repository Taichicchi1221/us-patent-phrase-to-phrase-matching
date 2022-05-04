import os
import shutil
import subprocess

DIR = "/workspaces/us-patent-phrase-to-phrase-matching/datasets"

os.chdir(DIR)
subprocess.run("kaggle datasets version --dir-mode zip -m 'default'", shell=True)
