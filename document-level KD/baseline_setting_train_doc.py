import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, BertForSequenceClassification


##### load RadBERT 
model_name = 'zzxslp/RadBERT-RoBERTa-4m'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model=BertForSequenceClassification.from_pretrained(model_name, num_labels=2, output_hidden_states=True, problem_type = "single_label_classification")
dataset = load_dataset("json", data_files="./dataset/binary_label_paragraph.json")
print(dataset)

train_encoding = tokenizer(
    dataset['train']['Context'],
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length = 120)

class EMRDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

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

########################### Training ###########################
def train(epoch, model, dataloader, optimizer, device):
    model.to(device)

    for e in range(1, epoch + 1):

        total_loss = 0
        preds = []
        labels = []
        model.train()
        count = 0

        progress_bar = tqdm(dataloader, desc=f'TRAIN - EPOCH {e} |')
        ########## Loading batch from the dataloader #########
        for data in progress_bar:
            # print(data)
            data = {k: v.to(device) for k, v in data.items()}
            output = model(**data)
            #print(output.logits)
            # print(output.loss)
            current_loss = output.loss
            total_loss += current_loss

            # print(output)
            ground_truth = data['labels'].detach().cpu().numpy()
            preds += list(output.logits.argmax(-1).detach().cpu().numpy())
            labels += list(ground_truth)

            ########## Backpropagation #############
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            current_np_loss = float(current_loss.detach().cpu().numpy())
            #progress_bar.set_description(f'TRAIN - EPOCH {e} | current-loss: {current_np_loss:.3f}')
            if count % 100 == 0:
                print(f'TRAIN - EPOCH {e} | current-loss: {current_np_loss:.3f}')
            count += 1
            data = {k: v.detach().cpu().numpy() for k, v in data.items()}

        avg = total_loss / len(dataloader)
        print('=' * 64)
        print(f"TRAIN - EPOCH {e} | LOSS: {avg:.4f}")
        print('=' * 64)

    return model

optimizer = AdamW(model.parameters(), lr=4e-5)
model = train(20, model, train_loader, optimizer, device)
model.save_pretrained("./trained_model/radbert_paragraph_wo_contrastive")
