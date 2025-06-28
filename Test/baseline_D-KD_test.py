import json
from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel
from torch import nn

#### load test dataset (high-confidence documents from MIMIC-p11)
file = open("./dataset/test_multi_label_dataset_final.json")
dataset = json.load(file)
# dataset

f_buffer = dataset[0]['text_file']
f_label = []
emr_label = {}
emr_report_count = 1
emr_report_yes = 0
emr_report_no = 0

#### 0: abnormal , 1: normal, 2: not much information
######## document GT
for i in range(len(dataset) - 1):
    if dataset[i]['confidence'] != 0 and dataset[i]['confidence'] != -1:
        if dataset[i]['Result'] == 'No':
            f_label.append(1)
        elif dataset[i]['Result'] == 'Yes':
            f_label.append(0)
        else:
            f_label.append(2)

    if dataset[i + 1]['text_file'] != f_buffer:
        sent_len = int(dataset[i]['sentence_index']) 

        if len(f_label) != sent_len + 1: ### reject the documents if it has any low-confidence sentences
            pass

        elif f_label.count(0) != 0:  ## if there are any abnomral sentences the document classify as anbormal
            f_label.append(0)
            emr_label[f_buffer] = f_label
            emr_report_yes += 1

        else:  ### otherwise it classify as normal.
            f_label.append(1)
            emr_label[f_buffer] = f_label
            emr_report_no += 1

        emr_report_count += 1
        f_buffer = dataset[i + 1]['text_file']
        f_label = []

print("total report count", emr_report_count)
print("certain report count", len(emr_label))
print("emr report yes count", emr_report_yes)
print("emr report no count", emr_report_no)

dataset = load_dataset("json", data_files="./dataset/test_multi_label_dataset_final.json")
text_list = np.array(dataset["train"]["text_file"])
emr_key_list = list(emr_label.keys())

positive_prob = []

m = torch.nn.Softmax()

### load radbert D-KD baseline(only cross-entropy loss)
model_name = "./trained_model/radbert_paragraph_wo_contrastive"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, output_hidden_states=True, problem_type="single_label_classification")
tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')

for file in emr_key_list:
    txt_idx = np.where(text_list == file)
    model.to('cuda')
    input_text = ''
    for j in txt_idx[0]:
        input_text += dataset['train']['Context'][j]
    train_encoding = tokenizer(
        input_text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=120)

    output = model(**train_encoding.to('cuda'))
    temp = m(output[0]).cpu().detach().numpy()
    print(temp)
    positive_prob.append(temp[0])

#### Extract the positive prob(abnormal prob) and save it.
np.save('./RESULTS/radbert_no_contrastive_paragraph_abn_prob_full.npy', positive_prob)
