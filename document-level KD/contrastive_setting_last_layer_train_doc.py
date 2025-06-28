import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

#### load RadBERT AutoTokenizer

model_name = 'zzxslp/RadBERT-RoBERTa-4m'
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("json", data_files="./dataset/binary_label_paragraph.json")
print(dataset)

train_encoding = tokenizer(
    dataset['train']['Context'],
    return_tensors='pt',
    padding=True,
    truncation=True
)

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
train_loader = DataLoader(train_set, batch_size=16) ### batch size 16
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

#### Last layer 
class MLP(nn.Module):
    def __init__(self, target_size, input_size= 768):
        super(MLP, self).__init__()
        self.num_classes = target_size
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, target_size)

    def forward(self, x):
        out = self.fc1(x)

        return out

########################### Last Layer Training ###########################
def train(epoch, radbert_encoder, classifier, dataloader, optimizer, device, batch_size, loss_func):
    radbert_encoder.to(device)
    classifier.to(device)
    sigmoid = torch.nn.Sigmoid()

    for e in range(1, epoch + 1):
        total_loss = 0
        preds = []
        labels = []
        classifier.train()
        count = 0

        progress_bar = tqdm(dataloader, desc=f'TRAIN - EPOCH {e} |')
        ########## Loading batch from the dataloader #########
        for data in progress_bar:
            data = {k: v.to(device) for k, v in data.items()}
            data_label = data.pop('labels')
            rad_output = radbert_encoder(**data)

            ##### we use pooler output from the trained BERT and make it input for the classifier
            ##### https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/bert/modeling_bert.py#L1517
            output = classifier(rad_output[1])
            current_loss = loss_func(output, data_label)
            total_loss += current_loss

            ground_truth = data_label.detach().cpu().numpy()

            preds += list(output.argmax(-1).detach().cpu().numpy())
            labels += list(ground_truth)

            ########## Backpropagation #############
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            current_np_loss = float(current_loss.detach().cpu().numpy())

            if count % 100 == 0:
                print(f'TRAIN - EPOCH {e} | current-loss: {current_np_loss:.3f}')
            #progress_bar.set_description(f'TRAIN - EPOCH {e} | current-loss: {current_np_loss:.3f}')
            count += 1

        avg = total_loss / len(dataloader)
        print('=' * 64)
        print(f"TRAIN - EPOCH {e} | LOSS: {avg:.4f}")
        print('=' * 64)

    return classifier


#### load trained encoder through contrastive setting
model_name = "./trained_model/contrastive_encoder_document"
radbert_encoder = AutoModel.from_pretrained(model_name)
model = MLP(target_size= 2, input_size= 768)
loss_func = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=4e-5)

model = train(20,  radbert_encoder=radbert_encoder,  classifier=model, dataloader=train_loader, optimizer=optimizer, device=device, batch_size=16, loss_func= loss_func)
torch.save(model.state_dict(), "./trained_model/mlp_classifier_paragraph_radbert")


