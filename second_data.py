import random
import nlpaug.augmenter.char as nac
from datasets import Dataset, DatasetDict
import os
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("ibm-research/Climate-Change-NER")
# Change this such it's iob and not iob2... also changed it from having to have the 
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

O_TAG_STR = "O"
""" Basically I changed aug_word_p from rate to 1.0 here is why
if random.random() < rate → the bouncer, decides which words even get sent to nlpaug
aug_word_p=1.0 → the worker, just does its job on every word it receives, no extra filtering """

'''"Climate" becomes "Climaye" '''
def augment_spelling(tokens, tags, rate):
    aug = nac.KeyboardAug(aug_char_p=0.1, aug_word_p=1.0)
    # nlpaug KeyboardAug handles typos
    # rate is the word-level chance like 10%, 20%, etc
    new_tokens = []
    for t in tokens:
                # flip the coin
        if random.random() < rate:
            res = aug.augment(t)
            new_tokens.append(res[0] if isinstance(res, list) else res)
        else:
            new_tokens.append(t)
                        # lenght is unchanged since its spelling
    return new_tokens, tags

def augment_add_delete(tokens, tags, rate):
    new_tokens = list(tokens)
    new_tags = list(tags)
    # deleting backwards prevents index errors
    # rate/2 is to have both addition and deletion add up to 10% total error, etc
    indices = sorted([i for i in range(len(new_tokens)) if random.random() < (rate/2)], reverse=True)
    for idx in indices:
        if len(new_tokens) > 5:
            new_tokens.pop(idx)
            new_tags.pop(idx)
    for _ in range(int(len(new_tokens) * (rate/2))):
        idx = random.randint(0, len(new_tokens))
        new_tokens.insert(idx, "noise")
        new_tags.insert(idx, O_TAG_STR) 
    return new_tokens, new_tags

'''Rise becomes Ri5e --> Optical Character Recognition" errors replaces char that look similar to a computer scanner'''
def augment_ocr(tokens, tags, rate):
    aug = nac.OcrAug(aug_char_p=0.1, aug_word_p=1.0)
    new_tokens = []
    for t in tokens:
        if random.random() < rate:
            res = aug.augment(t)
            new_tokens.append(res[0] if isinstance(res, list) else res)
        else:
            new_tokens.append(t)
    return new_tokens, tags

def augment_switching(tokens, tags, rate):
    """Randomly swaps word positions while keeping labels attached"""
    if len(tokens) < 2:
        return tokens, tags
    new_tokens = list(tokens)
    new_tags = list(tags)
    n_swaps = max(1, int(len(new_tokens) * rate))
    for _ in range(n_swaps):
        idx1, idx2 = random.sample(range(len(new_tokens)), 2)
        new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
        new_tags[idx1], new_tags[idx2] = new_tags[idx2], new_tags[idx1]
    return new_tokens, new_tags


# create dictionary
configs = {
    "spelling": {"func": augment_spelling, "rates": [0.03, 0.05, 0.10, 0.15, 0.20]},
    "add-delete": {"func": augment_add_delete, "rates": [0.05, 0.10, 0.15, 0.20, 0.25]},
    "ocr": {"func": augment_ocr, "rates": [0.05, 0.10, 0.15, 0.20, 0.23]},
    "switching": {"func": augment_switching, "rates": [0.05, 0.10, 0.15, 0.20, 0.25]} 
}

os.makedirs("augmented_datasets_train", exist_ok=True)
os.makedirs("augmented_datasets_test", exist_ok=True)
os.makedirs("augmented_datasets_val", exist_ok=True)

# save the clean data so we can use it for our baseline experiment aka for the zero shot noise robustness
print("Saving clean baseline datasets...")
dataset["train"].to_json("augmented_datasets_train/dataset_clean_0.json")
dataset["validation"].to_json("augmented_datasets_val/dataset_clean_0.json")
dataset["test"].to_json("augmented_datasets_test/dataset_clean_0.json")

for name, config in configs.items():
    for rate in config['rates']:
        print(f"generating {name} at {rate*100}%...")
        
        def process(example):
            # the reader function makes sure the keys are 'tokens' and 'ner_tags'
            t, g = config['func'](example['tokens'], example['ner_tags'], rate)
            return {"tokens": t, "ner_tags": g}
        
        aug_train = dataset["train"].map(process, load_from_cache_file= False)
        aug_test = dataset["test"].map(process, load_from_cache_file=False)
        aug_validation = dataset["validation"].map(process, load_from_cache_file=False)

        filename = f"augmented_datasets_train/dataset_{name}_{int(rate*100)}.json"
        aug_train.to_json(filename)
        
        filename = f"augmented_datasets_test/dataset_{name}_{int(rate*100)}.json"
        aug_test.to_json(filename)

        filename = f"augmented_datasets_val/dataset_{name}_{int(rate*100)}.json"
        aug_validation.to_json(filename)

print("all augmented datasets saved!")

"""
Basically, the def process we have MIGHT become buggie, if the .map() function 
somehow because lazy since it can mean the next dataset doesn't get the rigth amount of noise added to it.
Basically it might close the creaton of a dataset before rate has moved to a new rate... thereby making a ds with the wrong threshold
We might be fine, but it is as Claude says 'Bad practise to have code depend on secure run of a loop function...' Here is an alternative process def by Claude.
```
def process(example, func=config['func'], r=rate):
    t, g = func(example['tokens'], example['ner_tags'], r)
    return {"tokens": t, "ner_tags": g}
```
"""


