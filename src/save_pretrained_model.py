import os
import transformers

MODEL_NAME = "microsoft/deberta-base"

OUTPUT_DIR = os.path.join(
    "/workspaces/us-patent-phrase-to-phrase-matching/input/huggingface-models",
    MODEL_NAME,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_DIR)

model = transformers.AutoModel.from_pretrained(MODEL_NAME)
model.save_pretrained(OUTPUT_DIR)
