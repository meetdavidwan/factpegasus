import sys
import spacy
from tqdm import tqdm

spacy.require_gpu()

nlp = spacy.load("en_core_web_trf")
from datasets import load_from_disk

dataset_name = sys.argv[1]

dataset = load_from_disk(dataset_name)


def map_fn(features):
    out_docs = {"summary_tokens": [], "summary_ents": [], "document_ents": []}

    summary = [row.replace("\n", " ") for row in features["summary"]]

    for doc in tqdm(nlp.pipe(summary, disable=["lemmatizer"])):
        tokens = []
        for sent in doc.sents:
            # i_offset = 0
            # idx_offset = sent[0].idx
            i_offset, idx_offset = 0, 0
            toks = []

            ent_present = False
            for tok in sent:
                toks.append(
                    {
                        "text": tok.text,
                        "i": tok.i - i_offset,
                        "idx": tok.idx - idx_offset,
                        "dep": tok.dep_,
                        "head": tok.head.i - i_offset,
                        "children": [child.i - i_offset for child in tok.children],
                        "iob": tok.ent_iob_,
                        "ent_type": tok.ent_type_,
                        "whitespace": tok.whitespace_,
                    }
                )
                # if tok.ent_type_:
                #     ent_present = True

            # only save tokens for those that have entities
            # if ent_present:
            #     tokens += toks
            # sentences.append(sent.text)
            tokens += toks

        summary_ents = []
        for ent in doc.ents:
            summary_ents.append(
                {
                    "text": ent.text,
                    "start_idx": ent.start_char,
                    "end_idx": ent.end_char,
                    "ent_type": ent.label_,
                    "tokens": [tok.text for tok in ent],
                    "i": [tok.i for tok in ent],
                }
            )
        out_docs["summary_ents"].append(summary_ents)
        out_docs["summary_tokens"].append(tokens)

    document = [
        row.replace("\n", " ")
        for i, row in enumerate(features["document"])
        if out_docs["summary_ents"][i]
    ]
    docs = list(nlp.pipe(document, disable=["lemmatizer", "tagger", "parser"]))
    j = 0
    for i, summary_ent in enumerate(out_docs["summary_ents"]):
        document_ents = []
        if summary_ent:
            doc = docs[j]
            j += 1
            for ent in doc.ents:
                document_ents.append(
                    {
                        "text": ent.text,
                        "start_idx": ent.start_char,
                        "end_idx": ent.end_char,
                        "ent_type": ent.label_,
                        "tokens": [tok.text for tok in ent],
                        "i": [tok.i for tok in ent],
                    }
                )
        out_docs["document_ents"].append(document_ents)

    return out_docs


dataset["train"] = dataset["train"].map(map_fn, batched=True)
dataset["validation"] = dataset["validation"].map(map_fn, batched=True)
# dataset["test"] = dataset["test"].map(map_fn, batched=True)

dataset.save_to_disk("{}_tokens".format(dataset_name))