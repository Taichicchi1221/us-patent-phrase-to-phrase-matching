
cd "/workspaces/us-patent-phrase-to-phrase-matching/src"

# exp064
# python -u "/workspaces/us-patent-phrase-to-phrase-matching/src/work.py" \
#  RUN_NAME="exp064" \
#  RUN_DESC="deberta-v3-large with AdaBelief/SiFT for stacking" \
#  model_encoder_name="microsoft/deberta-v3-large" \
#  globals.seed=5 \
#  dataloader.train.batch_size=16 \
#  training.accumulate_gradient_batchs=2 \
#  training.max_grad_norm=1000.0 \
#  training.max_epochs=5

# exp065
# python -u "/workspaces/us-patent-phrase-to-phrase-matching/src/work.py" \
#  RUN_NAME="exp065" \
#  RUN_DESC="deberta-v3-base with AdaBelief/SiFT/lowercase for stacking" \
#  model_encoder_name="microsoft/deberta-v3-base" \
#  globals.seed=6 \
#  training.max_epochs=10


python -u "/workspaces/us-patent-phrase-to-phrase-matching/src/work.py" \
 RUN_NAME="exp066" \
 RUN_DESC="bert-for-patents with AdaBelief/SiFT for stacking" \
 model_encoder_name="anferico/bert-for-patents" \
 globals.seed=7 \
 training.max_epochs=5
