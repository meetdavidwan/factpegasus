import os
import nltk
import numpy as np
from datasets import load_metric


class Metric:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = load_metric("rouge")

    def postprocess_text(self, text):
        processed_text = []
        for line in text:
            line = line.strip()
            processed_text.append("\n".join(nltk.sent_tokenize(line)))
        return processed_text

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = self.postprocess_text(decoded_preds)
        decoded_labels = self.postprocess_text(decoded_labels)

        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
