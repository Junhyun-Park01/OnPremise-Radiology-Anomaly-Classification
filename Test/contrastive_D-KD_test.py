import json
from datasets import load_dataset
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel
from torch import nn
import re

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

total_label = []
ground_truth = []
positive_prob = []

total_length = len(emr_label)

m = torch.nn.Softmax()
  
## load trained encoder, last layer & AutoTokenizer from RadBERT
model_name = "./trained_model/contrastive_encoder_document"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')

class MLP(nn.Module):
    def __init__(self, target_size = 3, input_size= 768):
        super(MLP, self).__init__()
        self.num_classes = target_size
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, target_size)

    def forward(self, x):
        out = self.fc1(x)

        return out

classifier = MLP(target_size= 2, input_size= 768)
classifier.load_state_dict(torch.load('./trained_model/mlp_classifier_paragraph_radbert'))
classifier.cuda()
classifier.eval()

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
    output = classifier(output[1])

    temp = m(output[0]).cpu().detach().numpy()
    ground_truth.append(emr_label[file][-1])
    positive_prob.append(temp[0])

np.save('./RESULTS/radbert_paragraph_gd_full.npy', ground_truth)
### save positive prob(abnormal prob)
np.save('./RESULTS/radbert_paragraph_abn_prob_full.npy', positive_prob)
