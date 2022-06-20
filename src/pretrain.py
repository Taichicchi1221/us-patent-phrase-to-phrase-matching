import os
import random
import warnings
import shutil
import pandas as pd

import numpy as np
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


### settings
model_name = "microsoft/deberta-v3-small"


### premain
def clear_work(directory):
    shutil.rmtree(directory)
    os.makedirs(directory)


def premain(directory):
    clear_work(directory)
    os.chdir(directory)
    if os.path.exists(str(globals().get("__file__"))):
        shutil.copy(__file__, "pretrain.py")


premain("/workspaces/us-patent-phrase-to-phrase-matching/work")


###

warnings.filterwarnings("ignore")


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(42)


df = pd.read_csv("../input/pppm-abstract/pppm_abstract.csv")
df = df.dropna().reset_index(drop=True)

with open("corpus.txt", "w", encoding="utf-8") as f:
    for ab in df["abstract"]:
        f.write(ab + "\n")


model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_name)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./corpus.txt",  # mention train text file here
    block_size=256,
)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./corpus.txt",  # mention valid text file here
    block_size=256,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=model_name + "-pretrained",  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    save_total_limit=2,
    eval_steps=5000,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=False,
    prediction_loss_only=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()
trainer.save_model(model_name)
