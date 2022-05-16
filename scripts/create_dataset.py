from datasets import load_dataset
import sys

dataset_name = sys.argv[1]

if dataset_name == "wikihow":
    dataset = load_dataset(dataset_name, "all", "wikihowAll.csv")
    dataset = dataset.rename_column("headline","summary")
    dataset = dataset.rename_column("text","document")
else:
    dataset = load_dataset(dataset_name)

dataset.save_to_disk(sys.argv[2])

