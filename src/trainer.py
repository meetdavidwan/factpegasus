import torch
from packaging import version
from transformers import Trainer, Seq2SeqTrainer
from transformers.modeling_outputs import Seq2SeqLMOutput

from transformers.deepspeed import is_deepspeed_zero3_enabled

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

from infonce_loss import InfoNCELoss


class MyTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        contrastive_learning,
        contrastive_weight=1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss = None
        if contrastive_learning:
            self.contrastive_loss = InfoNCELoss()
            self.contrastive_weight = contrastive_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        # get the negative inputs out first
        neg_inputs = dict()
        for key in list(inputs.keys()):
            if key.startswith("negative_"):
                inp = inputs.pop(key)
                neg_inputs[key[9:]] = inp 

        # regular compute_loss
        if not neg_inputs:
            return super().compute_loss(model, inputs, return_outputs)
        
        # compute regular loss first and return output
        pos_loss, pos_output = super().compute_loss(
            model, inputs, return_outputs=True
        )

        contrastive_loss = self.contrastive_loss(model, pos_output, inputs, neg_inputs)
        total_loss = pos_loss + self.contrastive_weight * contrastive_loss

        return (total_loss, pos_output) if return_outputs else total_loss

    def prediction_step(
        self, model, inputs, prediction_loss_only=None, ignore_keys=None
    ):
        """
        Custom prediction with XSum Generate parameters.
        Everything is the same except gen_kwargs
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "length_penalty": 1.0,
            "max_length": 60,
            "min_length": 10,
            "num_beams": 6,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"]
            )  

        # ignore negative inputs if present
        inputs = {k:v for k,v in inputs.items() if not k.startswith("negative") }  

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)
