"""
Based on factcc's NERSwap: https://github.com/salesforce/factCC/blob/master/data_generation/augmentation_ops.py
"""

import os
import re
import pickle
import numpy as np

from collections import defaultdict
from tqdm import tqdm

class EntityPertubation:
    def __init__(
        self,
        dataset,
        num_negatives=1,
        pertubation_type="extrinsic",
    ):
        self.categories = defaultdict(lambda: "entity")
        self.categories.update(
            {
                "PERSON": "entity",
                "ORG": "entity",
                "NORP": "entity",
                "FAC": "entity",
                "GPE": "entity",
                "LOC": "entity",
                "PRODUCT": "entity",
                "WORK_OF_ART": "entity",
                "EVENT": "entity",
                "PERCENT": "number",
                "MONEY": "number",
                "QUANTITY": "number",
                "CARDINAL": "number",
                "DATE": "date",
                "TIME": "date",
            }
        )

        # parameters
        self.num_negatives = num_negatives
        self.pertubation_type = pertubation_type

        self.max_tries = 5

        # for extrinsic we need to save set of all candidates
        self.entity2word = None
        if self.pertubation_type == "extrinsic":
            print("generating entity2word for EntityPertubation")
            self.entity2word = defaultdict(set)
            for row in tqdm(dataset):
                for ent in row["document_ents"]:
                    ent_type = self.categories[ent["ent_type"]]
                    for tok in ent["tokens"]:
                        self.entity2word[ent_type].add(tok)
            self.entity2word = {k: list(v) for k, v in self.entity2word.items()}

    def perturbe(self, document, document_ents, summary, summary_ents, *args, **kwargs):
        # We cannot generate perturbed document if there is no entity to work with
        if len(summary_ents) == 0:
            return []
        # prepare candidates for pertubation
        pertubation_candidates = self.entity2word
        if self.pertubation_type == "intrinsic":
            pertubation_candidates = defaultdict(set)
            for ent in document_ents:
                ent_type = self.categories[ent["ent_type"]]
                for tok in ent["tokens"]:
                    pertubation_candidates[ent_type].add(tok)
            pertubation_candidates = {
                k: list(v) for k, v in pertubation_candidates.items()
            }

        disallowed_words = None
        if self.pertubation_type == "extrinsic":
            disallowed_words = set()
            for ent in document_ents:
                for tok in ent["tokens"]:
                    disallowed_words.add(tok)
        
        # Select factual summary entities
        factual_entities = []
        for _, summ_ent in self.get_factual_entities(document_ents, summary_ents):
            if self.is_valid_ent(summ_ent, pertubation_candidates):
                factual_entities.append(summ_ent)
        
        # We cannot do anything if we do not have any entity we can perturb on
        if not factual_entities:
            return []

        negative_examples = []

        for _ in range(self.num_negatives):

            # first choose entity we need to perturb on
            chosen_entity = factual_entities[np.random.randint(len(factual_entities))]
            # assert summary[chosen_entity["start_idx"]: chosen_entity["end_idx"]] == chosen_entity["text"], "{}\n{} {} {} {}".format(summary,summary[chosen_entity["start_idx"]:chosen_entity["end_idx"]], chosen_entity["text"], chosen_entity["start_idx"], chosen_entity["end_idx"])
            # for each chosen ent, choose random entity to replace with
            perturb_candidates = pertubation_candidates[self.categories[chosen_entity["ent_type"]]]

            max_tries = 10
            num_tries = 0
            candidate_text = None
            while candidate_text is None or not self.is_valid_candidate(
                candidate_text,
                chosen_entity,
                disallowed_words,
            ):
                candidate_text = perturb_candidates[np.random.randint(len(perturb_candidates))]
                
                num_tries += 1
                if num_tries > max_tries:
                    return []  

            perturbed_summary = (
                summary[: chosen_entity["start_idx"]]
                + candidate_text
                + summary[chosen_entity["end_idx"] :]
            )

            negative_examples.append(perturbed_summary)

        return negative_examples

    def get_factual_entities(self, document_ents, summary_ents):
        matches = list()

        doc_type2ent = defaultdict(list)
        for document_ent in document_ents:
            doc_type = self.categories[document_ent["ent_type"]]
            doc_type2ent[doc_type].append(document_ent)

        for summary_ent in summary_ents:
            summ_type = self.categories[summary_ent["ent_type"]]
            if summ_type in doc_type2ent:
                for document_ent in doc_type2ent[summ_type]:
                    if self.is_similar(summary_ent, document_ent):
                        matches.append((document_ent, summary_ent))
                        break
        return matches

    def is_similar(self, summary_ent, document_ent):
        return all(
            [tok in document_ent["tokens"] for tok in summary_ent["tokens"]]
        ) or all([tok in summary_ent["tokens"] for tok in document_ent["tokens"]])

    def is_valid_ent(self, ent, entity2text, disallowed_words=None):
        """
        Whether we can use the entity given the type2text dictionary
        At lease another not simialr entity need to be present for the given NER label
        """
        ent_type = self.categories[ent["ent_type"]]

        if ent_type not in entity2text:
            return False
        
        # need at least one to be different from the chosen entity
        for text in entity2text:
            if self.is_valid_candidate(text, ent, disallowed_words=disallowed_words):
                return True
        
        return False

    def is_valid_candidate(self, candidate_text, chosen_ent, disallowed_words=None):
        # if similar return False
        for tok in chosen_ent["tokens"]:
            if tok in candidate_text:
                return False

        # need to make sure that extrinsic text do not appear in document
        if disallowed_words is not None:
            for word in disallowed_words:
                if word in candidate_text or candidate_text in word:
                    return False
        return True
