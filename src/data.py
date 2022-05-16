import torch
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
)

import re
import math
import json
import numpy as np
import nltk

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from entity_pertubation import EntityPertubation


class Data:
    def __init__(self, args, model, tokenizer, seed, pretrain=False):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.pretrain = pretrain
        
        self.mask_token = None
        if self.args.add_mask_token:
            mask_token = getattr(self.tokenizer, "mask_token_sent", None)
            if mask_token is None:
                mask_token = self.tokenizer.mask_token
            self.mask_token = mask_token
            print("mask token is " + self.mask_token)

    def load_dataset(self, do_train=False, do_eval=False, do_predict=False):
        self.dataset = DatasetDict()

        dataset = load_from_disk(self.args.data_dir)

        if self.pretrain:
            self.dataset["train"] = dataset["train"]
        else:
            if do_train:
                # dataset["train"] = dataset["train"].select(range(100))
                self.dataset["train"] = self.preprocess(dataset["train"])
            if do_eval:
                # dataset["validation"] = self.preprocess(dataset["validation"].select(range(100)))
                self.dataset["validation"] = self.preprocess(dataset["validation"])
            if do_predict:
                self.dataset["test"] = self.preprocess(dataset["test" if not self.args.predict_validation else "validation"])

    def preprocess(self, dataset):
        if self.args.tokenize_on_fly:
            return dataset
        return dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.args.preprocessing_num_workers,
            load_from_cache_file=True
        )
        
    def preprocess_function(self, examples):
        inputs = examples["document"]
        targets = examples["summary"]

        # add mask to document
        if self.args.add_mask_token:
            inputs = [self.add_mask_token(inp) for inp in inputs]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.args.max_source_length,
            truncation=True,
        )

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.args.max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def add_mask_token(self, inp):
        sentences = nltk.sent_tokenize(inp)
        mask_token_position = min(self.args.mask_token_position, len(sentences))
        sentences.insert(mask_token_position, self.mask_token)
        return " ".join(sentences)

    def get_data_collator(self):
        entity_pertubation = None
        if self.args.num_negatives > 0:
            entity_pertubation = EntityPertubation(
                self.dataset["train"],
                num_negatives=self.args.num_negatives,
                pertubation_type=self.args.pertubation_type,
            )
        return MyDataCollator(
            self.tokenizer,
            self.model,
            entity_pertubation=entity_pertubation,
            max_source_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
            tokenize_on_fly=self.args.tokenize_on_fly,
        )
    
    def get_train_dataset(self):
        return self.dataset["train"] if "train" in self.dataset else None
    
    def get_eval_dataset(self):
        return self.dataset["validation"] if "validation" in self.dataset else None
    
    def get_predict_dataset(self):
        return self.dataset["test"] if "test" in self.dataset else None


@dataclass
class MyDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Any
    entity_pertubation: Optional[EntityPertubation]
    max_source_length: int = 512
    max_target_length: int = 128

    tokenize_on_fly: bool = False
    pad_to_multiple_of: int = 8

    def __post_init__(self):
        self.data_collator_seq2seq = DataCollatorForSeq2Seq(
            self.tokenizer, self.model, pad_to_multiple_of=self.pad_to_multiple_of
        )
    
    def __call__(self, features):
        if self.tokenize_on_fly:
            features = self.preprocess(features)
        
        if self.entity_pertubation is None:
            out_dict = self.data_collator_seq2seq(features)
        else:
            # prepare positive examples as is, and add negative inputs to a specific dict
            positive_examples = [{k: feat[k] for k in ["input_ids", "attention_mask", "labels"]} for feat in features]
            out_dict = self.data_collator_seq2seq(positive_examples)
            out_dict["decoder_attention_mask"] = torch.where(
                out_dict["decoder_input_ids"] == self.tokenizer.pad_token_id, 0, 1
            )
            # since pad token is used for sos, manually change it
            out_dict["decoder_attention_mask"][:,0] = 1

            # prepare negative examples
            neg_indices = []
            negative_examples = []
            for i, feat in enumerate(features):

                document = feat["document"]
                document_ents = feat["document_ents"]
                summary = feat["summary"]
                summary_ents = feat["summary_ents"]

                perturbed_summaries = self.entity_pertubation.perturbe(
                    document, document_ents, summary, summary_ents
                )

                if perturbed_summaries:
                    with self.tokenizer.as_target_tokenizer():
                        negative_inputs = self.tokenizer(
                            perturbed_summaries,
                            pad_to_multiple_of=self.pad_to_multiple_of,
                            truncation=True,
                            max_length=self.max_target_length,
                        )
                        for ii, am in zip(negative_inputs["input_ids"], negative_inputs["attention_mask"]):
                            negative_examples.append({"input_ids":ii, "attention_mask":am})
                        
                        neg_indices += [i] * len(negative_inputs["input_ids"])

            if negative_examples:
                neg_inputs = self.data_collator_seq2seq(negative_examples)

                neg_inputs["decoder_input_ids"] = neg_inputs.pop("input_ids")
                neg_inputs["decoder_attention_mask"] = neg_inputs.pop("attention_mask")

                neg_inputs["indice"] = torch.tensor(neg_indices)
                # cannot just put a dict into the inputs as this will raise issues in the trainer
                for key in neg_inputs:
                    out_dict["negative_{}".format(key)] = neg_inputs[key]
            
        return out_dict
    
    def preprocess(self, examples):
        model_inputs = self.tokenizer(
            [example["document"] for example in examples],
            pad_to_multiple_of=8,
            max_length=self.max_source_length,
            truncation=True,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [example["summary"] for example in examples],
                pad_to_multiple_of=8,
                max_length=self.max_target_length,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
        model_inputs_list = []
        for i in range(len(model_inputs["input_ids"])):
            model_inputs_list.append({k:model_inputs[k][i] for k in model_inputs.keys()})
        return model_inputs_list




@dataclass
class DataCollatorForPertubation:
    tokenizer: PreTrainedTokenizerBase
    model: Any
    entity_pertubation: EntityPertubation
    max_source_length: int = 512
    max_target_length: int = 128

    def __post_init__(self):
        self.collatorforseq2seq = DataCollatorForSeq2Seq(
            self.tokenizer, self.model, pad_to_multiple_of=8
        )

    def __call__(self, features):

        out_dict = dict()

        positive_examples, negative_examples = [], []
        neg_indices = []

        for i, feat in enumerate(features):
            summary = feat.pop("summary")
            summary_ents = feat.pop("summary_ents")
            document = feat.pop("document")
            document_ents = feat.pop("document_ents")

            pos_feat = {k: feat[k] for k in ["input_ids", "attention_mask", "labels"]}
            positive_examples.append(pos_feat)

            perturbed_texts = self.entity_pertubation.perturbe(
                document, document_ents, summary, summary_ents
            )

            for perturbed_text in perturbed_texts:
                negative_example = pos_feat.copy()
                #negative_example = dict()
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        perturbed_text,
                        pad_to_multiple_of=8,
                        truncation=True,
                        max_length=self.max_target_length,
                    )
                negative_example["labels"] = labels["input_ids"]

                negative_examples.append(negative_example)
                neg_indices.append(i)

        pos_inputs = self.collatorforseq2seq(positive_examples)
        # prepare the final output dict
        for k, v in pos_inputs.items():
            out_dict["pos_{}".format(k)] = v

        out_dict["pos_decoder_attention_mask"] = torch.where(
            pos_inputs["decoder_input_ids"] == self.tokenizer.pad_token_id, 0, 1
        )

        if negative_examples:
            neg_inputs = self.collatorforseq2seq(negative_examples)
            for k, v in neg_inputs.items():
                out_dict["neg_{}".format(k)] = v

            out_dict["neg_decoder_attention_mask"] = torch.where(
                neg_inputs["decoder_input_ids"] == self.tokenizer.pad_token_id, 0, 1
            )

            out_dict["neg_indice"] = torch.tensor(neg_indices)

        return out_dict