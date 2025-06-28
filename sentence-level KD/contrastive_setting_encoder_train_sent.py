from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel, AutoConfig
from torch.optim import AdamW
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
from datasets import load_metric
from losses import SupConLoss
import torch.nn.functional as F

### load RadBERT
model_name = 'zzxslp/RadBERT-RoBERTa-4m'
config = AutoConfig.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config)
dataset = load_dataset("json", data_files="./dataset/multi_label_dataset_final.json")

print(dataset)

train_encoding = tokenizer(
    dataset['train']['Context'],
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length = 120)

class EMRDataset(Dataset):
    def __init__(self, encodings, labels):
        multi_labels = np.zeros(len(labels))
        self.encoding = encodings

        for i in range(len(labels)):
            if labels[i] == 'Yes':
                multi_labels[i] = 0
            elif labels[i] == 'No':
                multi_labels[i] = 1
            ## not much information
            else:
                multi_labels[i] = 2

        self.labels = multi_labels

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.encoding.items()}
        data['labels'] = torch.tensor(self.labels[idx]).long()

        return data

    def __len__(self):
        return len(self.labels)


train_set = EMRDataset(train_encoding, dataset['train']['Result'])
train_loader = DataLoader(train_set, batch_size=16)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

########################### Encoder Training ###########################
def train(epoch, model, dataloader, optimizer, device, batch_size):
    model.to(device)
    sigmoid = torch.nn.Sigmoid()

    for e in range(1, epoch + 1):
        total_loss = 0
        preds = []
        labels = []
        model.train()
        count = 0

        ########## Loading batch from the dataloader #########
        for data in dataloader:
            data = {k: v.to(device) for k, v in data.items()}
            data_label = data.pop('labels')
            output = model(**data)
            ### Normalizing the feature vectors
            feature_vectors_normalized = F.normalize(torch.mean(output[0], axis=1), p=2, dim=1)
            logits = torch.div(torch.matmul(feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)), 0.1)
            #print(logits)
            ##### Using the Supervised contrastive learning loss
            loss = losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(data_label))
            current_loss = loss
            total_loss += current_loss

            ########## Backpropagation #############
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            current_np_loss = float(current_loss.detach().cpu().numpy())
            #             progress_bar.set_description(f'TRAIN - EPOCH {e} | current-loss: {current_np_loss:.3f}')
            count += 1

        avg = total_loss / len(dataloader)
        matrix = confusion_matrix(labels, preds)
        print('=' * 64)
        print(f"TRAIN - EPOCH {e} | LOSS: {avg:.4f}")

    return model

####### Encoder Training ########
###### Contrastive Learning ######
optimizer = AdamW(model.parameters(), lr=4e-5)
model= train(20, model, train_loader, optimizer, device, batch_size = 16)

###### Encoder Model Saved (Supervised Contrasive Learning) ##########
model.save_pretrained("./trained_model/contrastive_encoder_sentence")
