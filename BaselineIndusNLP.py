import numpy as np
from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer, #class that ensures that the tokenizing is done as in the pretraining of the model
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

model = "nasa-impact/nasa-smd-ibm-v0.1" #will be same for the tokenizer as well as the transformer

def read_conll_iob2(path):
    examples = []
    tokens = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": labels})
                    tokens, labels = [], []
                continue
            
            if line.startswith("#") or line.startswith("-DOCSTART-"):
                continue

            parts = line.split()
            if len(parts) < 3: continue 

            # col-0= ID, col-1= token, col-2= label
            tokens.append(parts[1])
            labels.append(parts[2])

    if tokens:
        examples.append({"tokens": tokens, "ner_tags": labels})
    return Dataset.from_list(examples)

dataset = DatasetDict({
    "train": read_conll_iob2("en_ewt-ud-train.iob2"),
    "validation": read_conll_iob2("en_ewt-ud-dev.iob2"),
    "test": read_conll_iob2("en_ewt-ud-test-masked.iob2"),
})


label_set = set()

# get all possible labels
for split in ["train", "validation"]:
    for example_labels in dataset[split]["ner_tags"]:
        for label in example_labels:
            label_set.add(label)

label_list = sorted(list(label_set))
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for i, l in enumerate(label_list)}

print(f"Discovered labels: {label_list}")

#from_pretrained method will download and cache the model for us
tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)

model = AutoModelForTokenClassification.from_pretrained(
    model,
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id,
)

'''function from the HF notebook, adapted to our dataset since the NER labels are 
still strings and not integers as the function expects! 
'''
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # []-type tokens are set to -100
            if word_idx is None:
                label_ids.append(-100)
            # label the first token of a given word
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # following subwords set label -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

#TrainingArguments, a class that contains all the attributes to customize the training
args = TrainingArguments(
    f"NLP-project",
    eval_strategy="epoch", 
    save_strategy="epoch",
    learning_rate=2e-5, #standard for fine-tuning, BERT paper suggests 3 different ones
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    seed=42, 
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("baseline_model")
tokenizer.save_pretrained("baseline_model")

"""Here we make the predictions"""
pred_output = trainer.predict(tokenized_datasets["test"])

predictions = pred_output.predictions
labels = pred_output.label_ids

predictions = np.argmax(predictions, axis=2)

our_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

"""Here we overwrite a version of the test so we can put predictions on it. This is to keep it in the same format as the OG text."""

with open("en_ewt-ud-test-masked.iob2", "r", encoding="utf-8") as f_in, \
     open("pred_test_ds.iob2", "w", encoding="utf-8") as f_out:

    sent_idx = 0
    word_idx = 0

    for line in f_in:
        line = line.rstrip("\n")

        if line.startswith("#"):
            f_out.write(line + "\n")

        elif line == "":
            f_out.write("\n")
            sent_idx += 1
            word_idx = 0

        else:
            parts = line.split("\t")
            parts[2] = our_predictions[sent_idx][word_idx]
            f_out.write("\t".join(parts) + "\n")
            word_idx += 1

"""Here we did to test the span_f1.py but we didn't have the """

pred_output = trainer.predict(tokenized_datasets["validation"])

predictions = pred_output.predictions
labels = pred_output.label_ids

predictions = np.argmax(predictions, axis=2)

our_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

with open("en_ewt-ud-dev.iob2", "r", encoding="utf-8") as f_in, \
     open("pred_dev.iob2", "w", encoding="utf-8") as f_out:

    sent_idx = 0
    word_idx = 0

    for line in f_in:
        line = line.rstrip("\n")

        if line.startswith("#"):
            f_out.write(line + "\n")

        elif line == "":
            f_out.write("\n")
            sent_idx += 1
            word_idx = 0

        else:
            parts = line.split("\t")
            parts[2] = our_predictions[sent_idx][word_idx]
            f_out.write("\t".join(parts) + "\n")
            word_idx += 1

print("Wrote pred_test_ds.iob2")
print("Wrote pred_dev.iob2")
