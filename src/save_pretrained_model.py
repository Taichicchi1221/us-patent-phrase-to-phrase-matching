import os
import transformers

# https://huggingface.co/

MODEL_NAME = "gpt2-large"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModel.from_pretrained(MODEL_NAME)


OUTPUT_DIR = os.path.join(
    "/workspaces/us-patent-phrase-to-phrase-matching/input/huggingface-models",
    MODEL_NAME,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
