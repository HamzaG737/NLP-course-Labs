import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import torch
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import copy


class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class Classifier:
    """The Classifier"""
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        self.device = 'cuda'

    def read(self, filename):
        dataset = pd.read_csv(filename, sep='\t', header=None)
        dataset = dataset.rename(index=str, columns={0: "sentiment",
                                                       1: "aspect_category",
                                                       2: "target_term",
                                                       3: "position",
                                                       4: "review"})
        dataset.review = dataset.review.str.lower()
        dataset.review = dataset.review.str.replace('-', ' ')
        dataset.target_term = dataset.target_term.str.lower()
        dataset.target_term = dataset.target_term.str.replace('-', ' ')

        return dataset


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        self.trainset = self.read(filename=trainfile)
        self.devset = self.read(filename="../data/devdata.csv")

        self.label_encoder.fit(self.trainset['sentiment'])
        self.trainset['sentiment_label'] = self.label_encoder.transform(self.trainset['sentiment'])
        self.devset['sentiment_label'] = self.label_encoder.transform(self.devset['sentiment'])


        self.train_encodings = self.tokenizer(list(self.trainset['review']),
                                              list(self.trainset['target_term']),
                                              truncation=True, padding=True)
        self.train_encodings['label'] = list(self.trainset['sentiment_label'])
        self.train_dataset = ABSADataset(self.train_encodings)
        self.train_loader = DataLoader(self.train_dataset, batch_size=12, shuffle=True)

        self.dev_encodings = self.tokenizer(list(self.devset['review']),
                                              list(self.devset['target_term']),
                                              truncation=True, padding=True)
        self.dev_encodings['label'] = list(self.devset['sentiment_label'])
        self.dev_dataset = ABSADataset(self.dev_encodings)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=12, shuffle=True)

        N_EPOCHS = 5

        n_train_steps__single_epoch = len(self.train_loader)
        n_train_steps = n_train_steps__single_epoch * N_EPOCHS

        # use this only if there is gpu available
        self.model.cuda()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8, weight_decay=0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * n_train_steps),
            num_training_steps=n_train_steps
        )

        self.model.zero_grad()
        best_acc = 0

        for num_epoch in range(N_EPOCHS):
            tr_loss = 0
            preds_train = None
            out_label_ids_train = None
            print(f'Epoch: {num_epoch}')
            self.model.train()
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "labels": labels}
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                logits = outputs[1]

                loss.backward()

                tr_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()

                if preds_train is None:
                    preds_train = logits.detach().cpu().numpy()
                    out_label_ids_train = inputs["labels"].detach().cpu().numpy()
                else:
                    preds_train = np.append(preds_train, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids_train = np.append(out_label_ids_train, inputs["labels"].detach().cpu().numpy(),
                                                    axis=0)

            preds_list_train = np.argmax(preds_train, axis=1)
            tr_acc = 100 * accuracy_score(out_label_ids_train, preds_list_train)
            tr_loss /= len(self.train_loader)

            print(f"Train Loss: {str(tr_loss)[:5]}, Accuracy: {str(tr_acc)[:5]}")


            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            self.model.eval()
            for batch in self.dev_loader:
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(self.device)

                    attention_mask = batch['attention_mask'].to(self.device)

                    token_type_ids = batch['token_type_ids'].to(self.device)

                    labels = batch['label'].to(self.device)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                        "labels": labels}
                    outputs = self.model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps

            preds_list = np.argmax(preds, axis=1)


            te_acc = 100 * accuracy_score(out_label_ids, preds_list)

            print(f'Dev Loss: {str(eval_loss)[:5]}, Accuracy: {str(te_acc)[:5]}')

            if te_acc > best_acc:
                torch.save(self.model.state_dict(), 'model_checkpoint.pt')
                print('Checkpoint saved')
                best_acc = te_acc

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        self.testset = self.read(filename=datafile)

        test_encodings = self.tokenizer(list(self.testset['review']), list(self.testset['target_term']),
                                        truncation=True, padding=True)
        preds = None
        self.model.load_state_dict(torch.load('model_checkpoint.pt'))
        self.model.eval()
        for i in range(len(test_encodings['input_ids'])):
            with torch.no_grad():
                input_ids = torch.tensor(test_encodings['input_ids'][i]).reshape(1, -1).to(self.device)

                attention_mask = torch.tensor(test_encodings['attention_mask'][i]).reshape(1, -1).to(self.device)

                token_type_ids = torch.tensor(test_encodings['token_type_ids'][i]).reshape(1, -1).to(self.device)

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
                outputs = self.model(**inputs)
                logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        preds_list = np.argmax(preds, axis=1)
        pred_labels = list(self.label_encoder.inverse_transform(list(preds_list)))

        return pred_labels




