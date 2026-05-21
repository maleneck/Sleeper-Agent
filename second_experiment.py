import os
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset, DatasetDict, load_dataset, Features, Sequence, Value
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
import torch

#Here we get a list of file names 
def get_all_json_files(directory):
    """Helper to get absolute paths of all JSON files in a folder"""
    base = os.path.abspath(directory)
    if not os.path.exists(base):
        return []
    return [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".json") and 'clean' not in f]

# Define your three folders
train_files = get_all_json_files("augmented_datasets_train")

#Now we will make a function for parsing filenames that we need for some of the columns in the csv
def parse_filename(filepath):
    basename=os.path.basename(filepath)
    basename=basename.replace('.json','')
    _,noise_type,noise_rate=basename.split('_')
    return noise_type, int(noise_rate)
# Here we are simply getting the lable info and making tags into ids and the other way around. We are baseing it on train and validation dataset
def get_label_info(dataset):
    label_set = set()
    for split in ["train", "validation"]:
        for example_labels in dataset[split]["ner_tags"]:
            label_set.update(example_labels)
    label_list = sorted(list(label_set))
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}
    return label_list, label_to_id, id_to_label

data_features = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(Value("string"))  # strings like "B-ORG", not ints
})

clean_dataset = load_dataset("json", data_files={
    "train": "augmented_datasets_train/dataset_clean_0.json",
    "validation": "augmented_datasets_val/dataset_clean_0.json",
    "test": "augmented_datasets_test/dataset_clean_0.json"
}, features=data_features)




models_to_test = [
    "nasa-impact/nasa-smd-ibm-v0.1", 
    "FacebookAI/roberta-base", 
    "nasa-impact/nasa-smd-ibm-distil-v0.1", 
    "distilbert/distilroberta-base"
]

# outside loop
def tokenize_and_align_labels(examples, tokenizer, label_to_id):
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

label_list, label_to_id, id_to_label = get_label_info(clean_dataset)

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # remove ignored index -100 and convert to label strings
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


all_results = []
RESULTS_FILE = "Trained_Robustness_results.csv"
for current_model in models_to_test:
    for train_file in train_files:
        basename=os.path.basename(train_file)
        val_file=os.path.join('augmented_datasets_val/', basename)
        test_file=os.path.join('augmented_datasets_test/', basename)
        dataset = load_dataset("json", data_files={"train": train_file,"validation": val_file,"test": test_file}, features=data_features)
        tokenizer = AutoTokenizer.from_pretrained(current_model, add_prefix_space=True)
        tokenized_datasets = dataset.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id),
            batched=True
        )
        tokenized_datasets_clean = clean_dataset.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id),
            batched=True
        )
        """I have played around with learning rate tried 2e-5 and 1e-5, I've tried 2,3,5 and 10 epochs per train. I've added 1 more model seeds so 3 seed in total..."""
        seeds=[42, 123, 999, 12,5,10,67,21,994,10]
        for seed in seeds:
            args = TrainingArguments(
                output_dir="./temp_checkpoints", 
                eval_strategy="no", 
                save_strategy="no",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=10,
                weight_decay=0.01,
                seed=seed,
                report_to="none",
                fp16=True # this is for v100 gpu for hpc
                # logging_dir=f"{output_dir}/logs",
            )

            torch.manual_seed(seed) # This is basically to fix a start so bad the model can't pull itself out of it in the training.

            model_obj = AutoModelForTokenClassification.from_pretrained(
                current_model,
                num_labels=len(label_list),
                id2label=id_to_label,
                label2id=label_to_id,
                ignore_mismatched_sizes=True # Important if swapping between different architectures
            )

            trainer = Trainer(
                model=model_obj,
                args=args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=DataCollatorForTokenClassification(tokenizer),
                # tokenizer=tokenizer,
                compute_metrics=compute_metrics, 
            )

            trainer.train()
            noisy_metrics = trainer.evaluate(tokenized_datasets['test'])
            clean_metrics = trainer.evaluate(tokenized_datasets_clean["test"])
            noise_type, noise_rate = parse_filename(train_file)        
            # Prepare the row for the CSV
            entry = {
                "model": current_model,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "seed": seed
                }
            
            # Merge in the metrics (removes 'eval_' prefix automatically)
            for k, v in noisy_metrics.items():
                entry[k.replace("eval_", "")] = v
            for k, v in clean_metrics.items():
                entry["clean_" + k.replace("eval_", "")] = v            
            all_results.append(entry)
            print(entry)
            # --- 6. SAVE RESULTS ---
            pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
            print(f"\n Results saved to {RESULTS_FILE}")
