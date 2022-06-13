
cd "/workspaces/us-patent-phrase-to-phrase-matching/src"

python -u "/workspaces/us-patent-phrase-to-phrase-matching/src/work.py" \
 RUN_NAME="exp053" \
 RUN_DESC="deberta-v3-large for stacking" \
 model_encoder_name="microsoft/deberta-v3-large" \
 dataloader.train.batch_size=16 \
 training.accumulate_gradient_batchs=2 \
 training.max_grad_norm=100.0