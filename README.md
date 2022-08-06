\*\*\* Same contents are also in kaggle discussion (https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332584) \*\*\*

My main solution is ensemble/stacking several experiments' results. Details below,

## Environments
- All experiments by my local machine(Corei7 12700 / 64GB RAM / RTX 3090)
- PyTorch(with pytorch-pfn-extras) and huggingface transformers
- Manage experiments by MLFlow
<br>
<br>


## Summary of My Experiments
| exp    | RUN\_ID                          | seed | n\_fold | encoder                    | head                   | loss              | preprocess         | optimizer/training | valid\_score | public\_score | private\_score |
| ------ | -------------------------------- | ---- | ------- | -------------------------- | ---------------------- | ----------------- | ------------------ | ------------------ | ------------ | ------------- | -------------- |
| exp018 | b02bfabced5345b689e6ac36e25a478c | 42   | 8       | microsoft/deberta-v3-large | AttentionPoolHead      | MSEWithLogitsLoss | None               | AdamW              | 0.83         | 0.839         | 0.8542         |
| exp021 | 187e25eeeed943f08f6b450e47455542 | 42   | 5       | microsoft/deberta-v3-large | SimpleHead             | MSEWithLogitsLoss | None               | AdamW              | 0.836        | 0.832         | 0.856          |
| exp041 | dc95c61f15bf4ac0b9e1de3ac1299f45 | 1221 | 10      | microsoft/deberta-v3-large | AttentionPoolHead      | MSELoss           | None               | AdamW              | 0.8325       | 0.8369        | 0.8553         |
| exp050 | 7923146adde64b6a9643eb163d8d223f | 1221 | 10      | microsoft/deberta-v3-base  | MultiSampleDropoutHead | MSELoss           | None               | AdamW              | 0.82         | 0.8213        | 0.8403         |
| exp051 | bc8145deac9d4376b708e8b1499da970 | 1221 | 10      | anferico/bert-for-patents  | MultiSampleDropoutHead | MSELoss           | None               | AdamW              | 0.825        | 0.8295        | 0.8396         |
| exp052 | b977b33e7f8547bcb1e62e42e021e627 | 1221 | 10      | roberta-large              | MultiSampleDropoutHead | MSELoss           | None               | AdamW              | 0.807        | 0.8262        | 0.8318         |
| exp053 | 9aa398aea22c4048b4e904c36bb3605b | 1221 | 10      | microsoft/deberta-v3-large | MultiSampleDropoutHead | MSELoss           | None               | AdamW              | 0.8307       | 0.8369        | 0.8513         |
| exp054 | e38625b57d3e4f00992c4f191a786c5d | 1221 | 10      | microsoft/deberta-v3-large | AttentionPoolHead      | BCEWithLogitsLoss | lowercase, “;”→”,” | AdamW              | 0.8308       | 0.8347        | 0.8526         |
| exp060 | c05c9ec899474ff58fb3552fe4a084ed | 1    | 5       | microsoft/deberta-v3-base  | AttentionPoolHead      | BCEWithLogitsLoss | None               | SiFT/Adabelief     | 0.8168       | 0.8196        | 0.8344         |
| exp061 | f87754b75cb54060b19527b551b3e6fe | 2    | 5       | anferico/bert-for-patents  | AttentionPoolHead      | BCEWithLogitsLoss | None               | SiFT/Adabelief     | 0.8271       | 0.8333        | 0.8447         |
| exp064 | c54349cdef094923b1003ba22b394ce9 | 5    | 5       | microsoft/deberta-v3-large | AttentionPoolHead      | MSELoss           | None               | SiFT/Adabelief     | 0.8343       | 0.8347        | 0.8552         |
| exp065 | 56a4472423e54bd39ff7b02c0daa08ac | 6    | 5       | microsoft/deberta-v3-base  | AttentionPoolHead      | MSELoss           | lowercase          | SiFT/Adabelief     | 0.819        | 0.8243        | 0.8424         |
| exp066 | 018420c68d63460f9ad625d51bb3903f | 7    | 5       | anferico/bert-for-patents  | AttentionPoolHead      | MSELoss           | None               | SiFT/Adabelief     | 0.8256       | 0.8283        | 0.8387         |

<br>

### common settings
#### batch_size, lr, etc...
- for large model, I used batch_size=16 and accumulate_grad_batches=2 and lr=5.0e-06(OneCycleLR) and small clip_grad_norm value(2.0~100.0)
- for small model, I used batch_size=32 and accumulate_grad_batches=1 and lr=5.0e-06(OneCycleLR) and large clip_grad_norm value(1000.0)
<br>

#### additional special tokens
I make input texts as this form,
text = f"{cpc_section} {sep} {anchor} {sep} {target} {sep} {cpc_context}."
- where cpc_section = "[A]", "[B]", ..., or "[H]" as special tokens
- where sep = "[s]" as a special token
<br>
<br>

## Ensemble
I used all results in the table above for blending/stacking.

1. Blending by Nelder-Mead weight optimization based on oof predictions.
2. BayesianRidge Stacking by oof predictions.

| No | valid_score | public_score | private_score |
|----|-------------|--------------|---------------|
| 1  | 0.8525      | 0.8493       | 0.8638        |
| 2  | 0.8522      | 0.8493       | 0.8638        |

<br>
<br>


## Code
training code: [https://github.com/Taichicchi1221/us-patent-phrase-to-phrase-matching](https://github.com/Taichicchi1221/us-patent-phrase-to-phrase-matching)
<br>
infernce notebook: [https://www.kaggle.com/code/hutch1221/uspppm-inference](https://www.kaggle.com/code/hutch1221/uspppm-inference)

