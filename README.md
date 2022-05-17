# FactPEGASUS: Factuality-Aware Pre-training and Fine-tuning for Abstractive Summarization (NAACL 2022)

This repository contains PyTorch code and pre-trained models for FactPEGASUS.

- Authors: [David Wan](https://meetdavidwan.github.io/) and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/) (UNC Chapel Hill)

- [Paper](https://arxiv.org/abs/2205.07830)

## Intallation and Dependencies
- Python 3.8
- PyTorch 1.10.2
- datasets 2.0.0
- transformers 4.17.0
- deepspeed 0.6.4
- nltk 3.7
- rouge_score 0.0.4

## Fine-tuning
### Data
The prepared dataset can be downloaded at: SOON

Dataset can be loaded with `datasets.load_from_disk(dir)`, containing three Dataset object: `train`,`validation`, and `test`. Each Dataset contains `[document, document_ents,summary,summary_ents]`.


To create the dataset on your own:
1. Create DatasetDict from the original dataset
```
python scripts/create_dataset.py [xsum,gigaword,xsum]
```
For Wikihow, please download the dataset from https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358 and save the file under `<path/to/folder>/wikihowAll.csv`, according the the official instruction. You may need to modify line 7 of the script to point to the right file.

2. Use SpaCy to extract entities and depency parse information
```
python scripts/run_spacy.py xsum
```
3. Run corrector
```
python scripts/corrector.py --data_dir data/xsum_tokens --save_dir data/xsum_comb --correction_type [all,remove,replace]
```
Note that for gigaword `--lowercase` is needed to avoid we captialize the sentences.

### Training
```
GPU_IDX=$1
PORT="29501"

deepspeed --master_port=$PORT --include=localhost:$GPU_IDX src/main.py --fp16 \
--deepspeed src/ds_config.json \
--data_dir data/xsum_comb --do_finetune \
--do_train --model_name models/factpegasus \
--evaluation_strategy no \
--per_device_train_batch_size 32 --per_device_eval_batch_size 8 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-05 --weight_decay 0.01 --label_smoothing 0.1 \
--max_source_length 512 --max_target_length 64 \
--logging_step 500 --max_steps 15000 \
--warmup_steps 500 --save_steps 1500 \
--output_dir factpegasus_ft_xsum_comb \
--contrastive_learning --pertubation_type intrinsic --num_negatives 5 --contrastive_weight 5
```
This should work without deepspeed as well by removing the deepspeed commands and arguments and the `--deepspeed` argument.

## Pretraining
Pretrained model can be downloaded here: SOON

### Data
We show how to create 1000 training data with `realnewslike`, but this can be also applied to the full `C4` dataset.
1. Run `scripts/pretraining_create_data.py`. This will create the data that contains the top 5 sentences according to ROUGE-1.
2. Run FactCC for each example. Note that the prediction from FactCC is actually reverse (0 is for factual and 1 for non-factual). We account that in our script. We provide a dummy prediction file under `scripts/factcc_dummy.json`
3. Combine the scores and create the pre-training data with `scripts/pretraining_combine_scores.py`

### Training

```
# effective batch size should be 256

GPU_IDX=$1

deepspeed --include=localhost:"$GPU_IDX" src/main.py --fp16 \
--data_dir data/c4_rouge1_factcc \
--do_train --do_pretrain --model_name facebook/bart-base \
--deepspeed src/ds_config.json \
--per_device_train_batch_size 64 --gradient_accumulation_steps 2 \
--learning_rate 1e-4 --weight_decay 0.01 \
--logging_step 100  --max_steps 750000 \
--warmup_steps 20000 --save_steps 5000 \
--max_source_length 512 --max_target_length 256 \
--output_dir factpegasus --pretrain_model_type bart_base --tokenize_on_fly
```

# Acknowledgements
The code is primarily adapted from the summarization example from [transformers](https://github.com/huggingface/transformers). We also borrowed code [FactCC](https://github.com/salesforce/factCC), and [info-nce-pytorch](https://github.com/RElbers/info-nce-pytorch).

# Reference
```BibTex
@inproceedings{wan2022factpegasus,
      title={FactPEGASUS: Factuality-Aware Pre-training and Fine-tuning for Abstractive Summarization}, 
      author={Wan, David and Bansal, Mohit},
      booktitle={NAACL 2022},
      year={2022}
}
```