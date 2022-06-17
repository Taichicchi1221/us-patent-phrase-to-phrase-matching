
cd "/workspaces/us-patent-phrase-to-phrase-matching/src"

# exp056
# python -u "/workspaces/us-patent-phrase-to-phrase-matching/src/work.py" \
#  RUN_NAME="exp056" \
#  RUN_DESC="gpt2-large for stacking" \
#  model_encoder_name="gpt2-large" \
#  dataloader.train.batch_size=4 \
#  training.accumulate_gradient_batchs=1 \
#  training.max_epochs=5

# exp054
# python -u "/workspaces/us-patent-phrase-to-phrase-matching/src/work.py" \
#  RUN_NAME="exp054" \
#  RUN_DESC="deberta-v3-large lowercase/bce/attentionpoolhead for stacking" \
#  model_encoder_name="microsoft/deberta-v3-large" \
#  dataloader.train.batch_size=16 \
#  training.accumulate_gradient_batchs=2 \
#  training.max_grad_norm=25.0 \
#  training.max_epochs=8

# exp055
##########################
## settings変更！！！
## head=MultiSampledropout
## preprocess=None
##########################

python -u "/workspaces/us-patent-phrase-to-phrase-matching/src/work.py" \
 RUN_NAME="exp055" \
 RUN_DESC="deberta-v2-xlarge for stacking" \
 model_encoder_name="microsoft/deberta-v2-xlarge" \
 dataloader.train.batch_size=4 \
 training.accumulate_gradient_batchs=4 \
 training.max_grad_norm=10.0 \
 training.max_epochs=5
