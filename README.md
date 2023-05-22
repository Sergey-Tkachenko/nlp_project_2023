# Paraphrase Detection on PAWS dataset

This project proposes a SOTA solution to the problem of paraphrase identification on PAWS$_{Wiki}$ test set. We used Concatenate Pooler with DeBERTa backbone trained on PAWS$_{Wiki}$ and PAWS$_{QQP}$ train sets to achieve $F1=0.95$: improve from $0.943$ ([previous SOTA](https://huggingface.co/domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector)).Also, we investigate the effects of unlabeled part of PAWS$_{Wiki}$.

SOTA run can be found here:
- https://api.wandb.ai/links/crendelyok_team/utm8mtl3
- https://www.kaggle.com/code/nikitapankov/concatenate-pooler-with-qqp?scriptVersionId=130458817


## Setup 
See ./notebooks to find out how to use this repo. 