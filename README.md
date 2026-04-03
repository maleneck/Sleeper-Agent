# Baseline Task for Group Sleeper Agents
## Description
This repo consistst of 'BaselineIndusNLP.py' which let's you train and tune the Indus model from Huggin Face libaray.
https://huggingface.co/nasa-impact/nasa-smd-ibm-v0.1 
Here also lies our prediction on the test split and a requirements.txt which we recomend you use a local environment with.
This recommendation stems from the fact that non-local environments containing torch haven't been working with this code.
## Usage Idea.
To run the 'BaselineIndusNLP.py' make sure these files are in the same folder as the 'BaselineIndusNLP.py' file.  
'en_ewt-ud-train.iob2',  
'en_ewt-ud-dev.iob2',  
'en_ewt-ud-test-masked.iob2',  
'span_f1.py'  

within the downloaded repo, you can create a virtual enviorment using the requirements.txt file for all the downloads.  
You can run the 'BaselineIndusNLP.py' file from terminal within your virtual enviorment and you will get both pred_test_ds.iob2 and pred_dev.iob2.  
We used pred_dev.iob2 to test that we could work with the 'span_f1.py', but you would likely just need the pred_test_ds.iob2 file.  
You can now run the 'span_f1.py' file with the golden standard test file against the predicted 'pred_test_ds.iob2' file like such   
  
python span_f1.py en_ewt-ud-test.iob2 pred_test_ds.iob2
  
Of course this file 'en_ewt-ud-test.iob2' should be the name of the golden standard file you have for test.

## Follow along
Download this github repo  
Exctract in your downloads  
Download or copy the datafiles + span_f1 into the repo folder  
Go to terminal  
Change directory to the 'Sleeper-agent-main' folder  
Create venv ```python -m venv SleeperVenv```  
Activate your venv ```SleeperVenv\Scripts\activate ```  
Use 'requirements.txt' to get the appropiate installs. ```pip install -r requirements.txt ```  
Run the 'BaselineIndusnlp.py' file ``` python BaselineIndusNLP.py```  
Run the 'span_f1.py' file with your gold standard test file and our prediction file. ```python span_f1.py en_ewt-ud-test.iob2 pred_test.iob2```  
 
## Expected output
You can expect this type of output when runnning the 'BaselineIndusNLP.py' file
Discovered labels: ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']
Loading weights: 100%|██████████████████████| 197/197 [00:00<?, ?it/s]
RobertaForTokenClassification LOAD REPORT from: nasa-impact/nasa-smd-ibm-v0.1
Key                             | Status     |
--------------------------------+------------+-
lm_head.layer_norm.bias         | UNEXPECTED |
lm_head.dense.weight            | UNEXPECTED |
lm_head.dense.bias              | UNEXPECTED |
roberta.embeddings.position_ids | UNEXPECTED |
lm_head.layer_norm.weight       | UNEXPECTED |
lm_head.bias                    | UNEXPECTED |
classifier.weight               | MISSING    |
classifier.bias                 | MISSING    |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Map: 100%|████████████| 12543/12543 [00:00<00:00, 21889.95 examples/s]
Map: 100%|██████████████| 2001/2001 [00:00<00:00, 29362.71 examples/s]
Map: 100%|██████████████| 2077/2077 [00:00<00:00, 33127.62 examples/s]
{'loss': '0.2077', 'grad_norm': '0.9754', 'learning_rate': '1.576e-05', 'epoch': '0.6378'}
{'eval_loss': '0.1107', 'eval_runtime': '2.7', 'eval_samples_per_second': '741.1', 'eval_steps_per_second': '46.67', 'epoch': '1'}
Writing model shards: 100%|█████████████| 1/1 [00:00<00:00,  2.25it/s] 
{'loss': '0.09383', 'grad_norm': '19.78', 'learning_rate': '1.151e-05', 'epoch': '1.276'}
{'loss': '0.07203', 'grad_norm': '1.399', 'learning_rate': '7.253e-06', 'epoch': '1.913'}
{'eval_loss': '0.0939', 'eval_runtime': '2.74', 'eval_samples_per_second': '730.2', 'eval_steps_per_second': '45.98', 'epoch': '2'}
Writing model shards: 100%|█████████████| 1/1 [00:00<00:00,  2.45it/s] 
{'loss': '0.05577', 'grad_norm': '0.9117', 'learning_rate': '3.002e-06', 'epoch': '2.551'}
{'eval_loss': '0.09149', 'eval_runtime': '2.788', 'eval_samples_per_second': '717.8', 'eval_steps_per_second': '45.2', 'epoch': '3'}
Writing model shards: 100%|█████████████| 1/1 [00:00<00:00,  2.41it/s] 
{'train_runtime': '238.1', 'train_samples_per_second': '158.1', 'train_steps_per_second': '9.879', 'train_loss': '0.09884', 'epoch': '3'}     
100%|█████████████████████████████| 2352/2352 [03:58<00:00,  9.88it/s] 
Writing model shards: 100%|█████████████| 1/1 [00:00<00:00,  2.35it/s] 
100%|███████████████████████████████| 130/130 [00:03<00:00, 39.82it/s]
Wrote pred_test_ds.iob2
Wrote pred_dev.iob2

You can expect an output like this when running the 'span_f1.py'file with the 'pred_test_ds.iob2' file. Be aware, the numbers will not be the same as these since they were derived from running 'span_f1.py' along the validation file and predictions!

recall:    0.6749482401656315
precision: 0.659919028340081
slot-f1:   0.6673490276356193

unlabeled
ul_recall:    0.7795031055900621
ul_precision: 0.7621457489878543
ul_slot-f1:   0.7707267144319344

loose (partial overlap with same label)
l_recall:    0.7401656314699793
l_precision: 0.72165991902834
l_slot-f1:   0.7307956404129078

## Resources
https://huggingface.co/learn/llm-course/chapter2/1
https://huggingface.co/docs/transformers/tasks/token_classification?utm_source=chatgpt.com 
https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb
https://medium.com/@whyamit101/fine-tuning-bert-for-named-entity-recognition-ner-b42bcf55b51d  
As well as ChatGPT which helped us with alligning the predictions to the conll format.  
