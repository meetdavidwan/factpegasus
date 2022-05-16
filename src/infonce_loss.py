import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, model, pos_output, pos_inputs, neg_inputs):
         # We only want to perform contrastive learning on those that has negative examples
        neg_indices = neg_inputs["indice"]
        neg_input_ids = neg_inputs["decoder_input_ids"]
        neg_attention_mask = neg_inputs["decoder_attention_mask"]

        valid_indices = torch.unique(neg_indices)
        num_negative = neg_indices.size(0) // valid_indices.size(0)

        encoder_hidden_states = pos_output.encoder_last_hidden_state
        encoder_attention_mask = pos_inputs["attention_mask"]
        decoder_hidden_states = pos_output.decoder_hidden_states[-1]
        decoder_attention_mask = pos_inputs["decoder_attention_mask"]

        pos_u = self.mean_pooling(encoder_hidden_states[valid_indices], encoder_attention_mask[valid_indices])
        pos_v = self.mean_pooling(decoder_hidden_states[valid_indices], decoder_attention_mask[valid_indices])

        # neg v should have the shape (N,M,D)
        # run decoder for getting the neg_v
        decoder = None
        unwrapped_model = model
        if hasattr(unwrapped_model,"module"):
            unwrapped_model = unwrapped_model.module
        decoder = unwrapped_model.get_decoder()

        neg_v = decoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            encoder_hidden_states=encoder_hidden_states[neg_indices],
            encoder_attention_mask=encoder_attention_mask[neg_indices],
            use_cache=None,
            head_mask=None,
            past_key_values=None,
        )[0]
        neg_v = self.mean_pooling(neg_v, neg_attention_mask)
        neg_v = neg_v.view(-1, num_negative, neg_v.size(-1))

        # from info-nce-pytorch

        # normalize
        pos_u = F.normalize(pos_u, dim=-1)
        pos_v = F.normalize(pos_v, dim=-1)
        neg_v = F.normalize(neg_v, dim=-1)

        positive_logit = torch.sum(pos_u * pos_v, dim=1, keepdim=True)
        negative_logits = pos_u.unsqueeze(1) @ neg_v.transpose(-2, -1)
        negative_logits = negative_logits.squeeze(1)

        # for each example, we have the positive at 0th position, and the other ones are all negative examples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=pos_u.device)
        return F.cross_entropy(logits/self.temperature, labels)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
