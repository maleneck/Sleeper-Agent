import os
import pandas as pd
import numpy as np
import torch
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
ds = load_dataset("ibm-research/Climate-Change-NER")

def read_conll_iob(split):
    examples = []
    tokens = []
    labels = []

    for line in split:
        line = line["text"].strip()
        if not line:
            if tokens:
                examples.append({"tokens": tokens, "ner_tags": labels})
                tokens, labels = [], []
            continue
            
            # skip metadata
        if line.startswith("#") or line.startswith("-DOCSTART-"):
            continue

        parts = line.split()
            
        if len(parts) >= 2:
            tokens.append(parts[0])
            labels.append(parts[1]) 

    if tokens:
        examples.append({"tokens": tokens, "ner_tags": labels})
    return Dataset.from_list(examples)

dataset = DatasetDict({
    "train": read_conll_iob(ds["train"]),
    "validation": read_conll_iob(ds["validation"]),
    "test": read_conll_iob(ds["test"]),
})


# config
MODEL_DIR = "./final_models"
NOISE_DATA_DIR = "./augmented_datasets_test_3"
RESULTS_FILE = "clean_data_robustness_results.csv"

model_names = [
    "nasa-smd-ibm-v0.1", 
    "roberta-base", 
    "nasa-smd-ibm-distil-v0.1", 
    "distilroberta-base"
]

noise_types = ["spelling", "add-delete", "ocr", "switching"]
noise_rates = [3, 5, 10, 15, 20, 23, 25]
test_seeds = [42, 123, 999]

# metrics and labels

# sequeval takes the tags B- and I- together as one entity
metric = evaluate.load("seqeval")

# recycle the same code from MakeDataset.py
# make sure labels align!!
label_set = set()
for split in ["train", "validation"]:
    for example_labels in dataset[split]["ner_tags"]:
        label_set.update(example_labels)
label_list = sorted(list(label_set))
# dictinaries for translating words to numbers and back
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for i, l in enumerate(label_list)}


# metrics per entity as well
def compute_metrics(p):
    predictions, labels = p
    # pick prob. with highest score
    predictions = np.argmax(predictions, axis=2)

    # convert ids to str, ignoring -100
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # calculates final F1, precision, and recall
    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    #  a dictionary containing the micro-average
    output = {
        "micro_precision": results["overall_precision"],
        "micro_recall": results["overall_recall"],
        "micro_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }

    # per entity metric and add individual scores to results
    for key, value in results.items():
        if key not in ["overall_precision", "overall_recall", "overall_f1", "overall_accuracy"]:
            output[f"{key}_f1"] = value["f1"]
            output[f"{key}_precision"] = value["precision"]
            output[f"{key}_recall"] = value["recall"]

    return output


def get_noisy_dataset_path(ntype, nrate, nseed):
    path = f"{NOISE_DATA_DIR}/dataset_{ntype}_{nrate}_s{nseed}.json"
    return path if os.path.exists(path) else None

# evaluation loop
all_results = []

for m_name in model_names:
    print(f"\n evaluating model: {m_name}")
    m_path = os.path.join(MODEL_DIR, m_name)
    
    if not os.path.exists(m_path):
        print(f"Skipping {m_name}, path not found at {m_path}")
        continue
    # load the weights of the fine-tuned model saved on disk
    tokenizer = AutoTokenizer.from_pretrained(m_path)
    model = AutoModelForTokenClassification.from_pretrained(m_path)
    
    '''
    recycled function from baseline.py
    helper function that ensures that if a tokenizer splits a word into sub-tokens, 
    the NER label is only assigned to the first sub-token, while the others get -100
    '''
    def tokenize_and_align_eval(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # set up evaluation environment
    eval_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=32,
        report_to="none" # set report_to="none" to keep the console clean
    )
    

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics 
    )

    # 84 different test scenarios for every model
    for ntype in noise_types:
        for nrate in noise_rates:
            for nseed in test_seeds:
                d_path = get_noisy_dataset_path(ntype, nrate, nseed)
                if d_path:
                    print(f"  Testing {ntype} @ {nrate}% (seed {nseed})...")
                    noisy_ds = load_dataset("json", data_files=d_path, split="train")
                    noisy_tok = noisy_ds.map(tokenize_and_align_eval, batched=True)
                    
                    # where the inference is happening & return scores
                    noisy_metrics = trainer.evaluate(noisy_tok)
                    
                    # entry is the row in the csv file
                    entry = {
                        "model": m_name,
                        "noise_type": ntype,
                        "noise_rate": nrate,
                        "seed": nseed
                    }
                    # add in the metrics (removes 'eval_' prefix automatically)
                    for k, v in noisy_metrics.items():
                        entry[k.replace("eval_", "")] = v
                        
                    all_results.append(entry)

# saving results
results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_FILE, index=False)
print(f"\n Results saved to {RESULTS_FILE}")