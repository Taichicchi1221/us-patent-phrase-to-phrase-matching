import os
import sys
import subprocess
import transformers

sys.path.append("../input/COCO-LM/huggingface")
from cocolm.modeling_cocolm import COCOLMModel
from cocolm.configuration_cocolm import COCOLMConfig
from cocolm.tokenization_cocolm import COCOLMTokenizer


# https://huggingface.co/

MODEL_NAME = "microsoft/cocolm-large"

if "microsoft/cocolm" in MODEL_NAME:
    OUTPUT_DIR = os.path.join(
        "/workspaces/us-patent-phrase-to-phrase-matching/input/huggingface-models",
        MODEL_NAME,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)
    path = os.path.join("https://huggingface.co", MODEL_NAME, "resolve/main")
    for file in ("config.json", "dict.txt", "pytorch_model.bin", "sp.model"):
        filepath = os.path.join(path, file)
        print(filepath)
        subprocess.run(f"curl -OL {filepath}", shell=True)
    exit()
else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModel.from_pretrained(MODEL_NAME)


OUTPUT_DIR = os.path.join(
    "/workspaces/us-patent-phrase-to-phrase-matching/input/huggingface-models",
    MODEL_NAME,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
