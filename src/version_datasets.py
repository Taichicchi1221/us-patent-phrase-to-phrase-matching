import os
import shutil
import subprocess


os.chdir("/workspaces/us-patent-phrase-to-phrase-matching/datasets")

subprocess.run("kaggle datasets version --dir-mode tar -m 'default'", shell=True)
