"""
Script mainly baised on run_mlm
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForPreTraining,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    AutoModelForSeq2SeqLM,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from data import Data
from trainer import MyTrainer
from entity_pertubation import EntityPertubation
from metric import Metric


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(default="facebook/bart-base", metadata={"help": "model"})

    do_pretrain: bool = False
    do_finetune: bool = False

    pretrain_model_type: str = "bart_base"

    contrastive_learning: bool = False
    contrastive_weight: float = 1.0

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str

    max_source_length: int = field(default=512)
    max_target_length: int = field(default=256)

    tokenize_on_fly: bool = field(default=False)
    preprocessing_num_workers: int = field(default=None)

    # Contrastor
    num_negatives: int = field(default=0)
    pertubation_type: Optional[str] = field(default="extrinsic")

    # Connector
    add_mask_token: bool = field(default=False)
    mask_token_position: int = field(default=0)

    # early stopping
    early_stopping: bool = field(default=False)
    patience: int = field(default=3)

    # evaluate on validation
    predict_validation: bool = field(default=False)

def main():
    parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
            )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                    )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                    )

            # Set seed before initializing model.
    set_seed(training_args.seed)

    if model_args.do_pretrain or model_args.contrastive_learning:
        training_args.remove_unused_columns = False

    if model_args.do_pretrain:
        tokenizer = AutoTokenizer.from_pretrained(
                "google/pegasus-large",
                cache_dir=".cache",
                use_fast=True,
                mask_token="<mask>",
                )

        # for config, we need to change the vocab size to pegasus
        d_model = 768 if "_base" in model_args.pretrain_model_type else 1024
        ffn_dim = 3072 if "_base" in model_args.pretrain_model_type else 4096
        attention_heads = 12 if "_base" in model_args.pretrain_model_type else 16

        layers_dict = {
                "bart_base": 6,
                "bart_large": 12,
                "pegasus_base": 12,
                "pegasus_large": 16,
                }
        layers = layers_dict[model_args.pretrain_model_type]

        config = AutoConfig.from_pretrained(
                "facebook/bart-base",
                cache_dir=".cache",
                vocab_size=len(tokenizer),
                d_model=d_model,
                encoder_ffn_dim=ffn_dim,
                decoder_ffn_dim=ffn_dim,
                encoder_layers=layers,
                decoder_layers=layers,
                encoder_attention_heads=attention_heads,
                decoder_attention_heads=attention_heads,
                pad_token_id=0,
                bos_token_id=0,
                eos_token_id=1,
                forced_eos_token_id=1,
                decoder_start_token_id=0,
                )

        model = AutoModelForSeq2SeqLM.from_config(config)
        print(model)
        
    elif model_args.do_finetune or training_args.do_predict:
        config = AutoConfig.from_pretrained(
                model_args.model_name,
                cache_dir=".cache",
                )
        tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name,
                cache_dir=".cache",
                use_fast=True,
                )
        model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name,
                config=config,
                cache_dir=".cache",
                )
    else:
        raise ValueError(
                "Need to specify either do_pretrain, do_finetune, or do_predict"
                )

        model.resize_token_embeddings(len(tokenizer))

    # need the following for getting embeddings for contrastive_learning
    if model_args.contrastive_learning:
        assert (
                data_args.num_negatives > 0
                ), "Please set num_negatives >0 for generating negatives examples for contrastive learning"
        config.output_hidden_states = True
        config.return_dict = True

    data = Data(
            data_args,
            model,
            tokenizer,
            seed=training_args.seed,
            pretrain=model_args.do_pretrain,
            )
    data.load_dataset(
            do_train=training_args.do_train,
            do_eval=training_args.do_eval,
            do_predict=training_args.do_predict,
            )

    train_dataset = data.get_train_dataset()
    eval_dataset = data.get_eval_dataset()
    predict_dataset = data.get_predict_dataset()

    metric = Metric(tokenizer)

    callbacks = []
    if data_args.early_stopping:
        callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=data_args.patience)
                )

    trainer = MyTrainer(
                contrastive_learning=model_args.contrastive_learning,
                contrastive_weight=model_args.contrastive_weight,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data.get_data_collator(),
                compute_metrics=metric.compute_metrics
                if training_args.predict_with_generate
                else None,
                callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

    # Predict
    if training_args.do_predict:

        logger.info("*** Predict ***")
        predict_results = trainer.predict(data.dataset["test"])

        metrics = predict_results.metrics
        trainer.log_metrics("predict", metrics)

        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                    predict_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    )
            predictions = [pred.strip().replace("\n", "") for pred in predictions]

            output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt"
                    )
            with open(output_prediction_file, "w") as writer:
                for pred in predictions:
                    writer.write("{}\n".format(pred))


if __name__ == "__main__":
    main()
