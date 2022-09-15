from collections import defaultdict
import re

from datasets import load_from_disk
import argparse

class EntityCorrector:
    """
    Entity Pertubation that fixes summaries or perturbs text
    """

    def __init__(self, correction_type="all",lowercase=False):
        self.correction_type = correction_type
        self.lowercase = lowercase


    def correct_hallucinations(self, features):
        """
        Map function wrapper
        """

        document = features["document"]
        summary = features["summary"]
        orig_summ = summary

        document_tokens = features["document_tokens"] if "document_tokens" in features else None
        summary_tokens = features["summary_tokens"]

        # change summary tokens to dict with key being i
        summary_tokens = {k["i"]: k for k in summary_tokens}

        doc_ents = features["document_ents"]
        summ_ents = features["summary_ents"]

        hallucinations = self.get_hallucinations(doc_ents, summ_ents, document_tokens)

        if self.correction_type == "replace" or self.correction_type == "all":
            (
                hallucinations,
                summary,
                summary_tokens,
                summ_ents,
            ) = self.replace_hallucinations(
                hallucinations,
                summary,
                summary_tokens,
                summ_ents,
                document_tokens,
                doc_ents,
            )
        orig_summ_after_replace = summary
        if self.correction_type == "remove" or self.correction_type == "all":
            summary, summary_tokens, summ_ents = self.remove_hallucinations(
                hallucinations, summary, summary_tokens, summ_ents
            )
        
        features["summary"] = summary
        features["summary_tokens"] = summary_tokens
        features["summary_ents"] = summ_ents

        features["corrected"] = 1 if summary != orig_summ_after_replace else 0

        return features

    def get_hallucinations(self, document_ents, summary_ents, document_tokens):
        """
        Check for entities that appear in summary but not in documents
        """
        hallucinated_entities = list()

        if len(summary_ents) == 0:
            return hallucinated_entities

        # Prepare dictonary of entities sorted by NER label
        document_entities = set()
        for document_ent in document_ents:
            document_entities.add(document_ent["text"])
            document_entities.update(set(document_ent["tokens"]))

        # find hallucinated entities by doing simple search
        for i, summary_ent in enumerate(summary_ents):
            if summary_ent["text"] in document_entities or all([tok in document_entities for tok in summary_ent["tokens"]]):
                continue
            hallucinated_entities.append(i)

        return sorted(hallucinated_entities, reverse=True)

    def replace_hallucinations(
        self,
        hallucinations,
        summary,
        summary_tokens,
        summ_ents,
        document_tokens,
        document_ents,
    ):
        updated_hallucinations = []

        for hal_i, summ_i in enumerate(hallucinations):
            ent = summ_ents[summ_i]
            # Check if we can match an entity in document_ent
            matching_ent = None

            for doc_ent in document_ents:
                if all([t in ent["tokens"] for t in doc_ent["tokens"]]):
                    matching_ent = doc_ent
                    break
            

            if matching_ent is None:
                updated_hallucinations.append(summ_i)
                continue

            # otherwise we replace text from doc ent into the hallucinatied summ ent and update everything accordingly
            offset = len(matching_ent["text"]) - len(ent["text"])

            # adjust summary
            summary = (
                summary[: ent["start_idx"]]
                + matching_ent["text"]
                + summary[ent["end_idx"] :]
            )

            summ_ents[summ_i]["text"] = matching_ent["text"]
            summ_ents[summ_i]["end_idx"] += offset

            # we still keep the tokens not to break the dependency tree
            for tok_i in ent["i"]:
                summary_tokens[tok_i]["dep"] = "fixed"

            for i in summary_tokens:
                if i > ent["i"][-1]:
                    summary_tokens[i]["idx"] += offset

            # adjust ents
            for ent_i in range(len(summ_ents)):
                if summ_ents[ent_i]["start_idx"] > ent["start_idx"]:
                    summ_ents[ent_i]["start_idx"] += offset
                    summ_ents[ent_i]["end_idx"] += offset

        return updated_hallucinations, summary, summary_tokens, summ_ents

    def remove_hallucinations(self, hallucinations, summary, summary_tokens, summ_ents):
        tokens_to_remove = set()
        for i in hallucinations:
            ent = summ_ents[i]
            for tok_i in ent["i"]:
                tok = summary_tokens[tok_i]
                tokens_to_remove.add(tok_i)

                # remove children
                for child_i in tok["children"]:
                    child = summary_tokens[child_i]
                    if child["dep"] not in ["compound", "relcl", "fixed"]:
                        tokens_to_remove.add(child_i)

                # go up and remove anything that is pobj and prep
                # note that the current dep is actually the dep type we care
                if tok["dep"] in ["pobj", "prep"]:
                    parent_i = tok["head"]
                    parent = summary_tokens[parent_i]

                    children = [
                        child
                        for child in self.get_children(parent, summary_tokens)
                        if child not in tokens_to_remove
                    ]

                    while parent["dep"] in ["pobj", "prep"] and len(children) == 0:
                        tokens_to_remove.add(parent_i)

                        parent_i = parent["head"]
                        parent = summary_tokens[parent_i]
                        children = [
                            child
                            for child in self.get_children(parent, summary_tokens)
                            if child not in tokens_to_remove
                        ]

        # remove tokens from the last position to the front and adjust idx and i
        tokens_to_remove = sorted(list(tokens_to_remove), reverse=True)
        for tok_i in tokens_to_remove:
            tok = summary_tokens[tok_i]
            tok_idx = int(tok["idx"])

            summary = (
                summary[:tok_idx]
                + summary[tok_idx + len(tok["text"]) + len(tok["whitespace"]) :]
            )
            for i in summary_tokens:
                if i > tok_i:
                   summary_tokens[i]["idx"] -= len(tok["text"]) + len(tok["whitespace"])
            summary_tokens.pop(tok_i)

            # adjust summary ents
            for summ_ent in summ_ents:
                if summ_ent["i"][0] > tok_i:
                    offset = len(tok["text"]) + len(tok["whitespace"])
                    summ_ent["start_idx"] -= offset
                    summ_ent["end_idx"] -= offset

        # capitalize first letter if needed
        if not self.lowercase:
            summary = (
                summary[0].upper() + summary[1:] if len(summary) > 1 else summary.upper()
            )

        for i in hallucinations:
            summ_ents.pop(i)
        
        # also remove additional entities that are removed
        tokens_to_remove = set(tokens_to_remove)
        ents_to_remove = list()
        for i, summ_ent in enumerate(summ_ents):
            if any([i in tokens_to_remove for i in summ_ent["i"]]):
                ents_to_remove.append(i)
        ents_to_remove = set(ents_to_remove)
        summ_ents = [ent for i, ent in enumerate(summ_ents) if i not in ents_to_remove]

        return summary, summary_tokens, summ_ents

    def get_children(self, tok, tokens):
        l = list()
        # print("get_childer", tok, list(tok.children))
        for child_i in tok["children"]:
            if tokens[child_i]["dep"] != "fixed":
                l.append(child_i)
            l += self.get_children(tokens[child_i], tokens)
        return l

def main(args):
    corrector = EntityCorrector(args.correction_type, lowercase=args.lowercase)

    dataset = load_from_disk(args.data_dir)

    columns_to_remove = ["summary_tokens"]
    if "document_tokens" in dataset["train"][0]:
        columns_to_remove.append("document_tokens")

    dataset["train"] = dataset["train"].map(
        corrector.correct_hallucinations,
        remove_columns=columns_to_remove,
        keep_in_memory=True,
        num_proc=32,
    )

    dataset["validation"] = dataset["validation"].map(
        corrector.correct_hallucinations,
        remove_columns=columns_to_remove,
        keep_in_memory=True,
        num_proc=32,
    )

    dataset.save_to_disk(args.save_dir)


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--lowercase", action="store_true", default=False)

parser.add_argument(
    "--correction_type", type=str, default="all", choices=["all", "remove", "replace"]
)

args = parser.parse_args()

main(args)