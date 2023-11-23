# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import os
import re
import spacy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import precision_recall_curve, auc
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
import gensim.downloader
from gensim.models import Word2Vec
import string

# import fastparquet

global_device = 'cpu'

if torch.cuda.is_available():
    global_device = 'cuda'

"""## Load data"""

processed_dataset = pd.read_parquet('./data/mountain_dataset_processed.parquet', engine='fastparquet')

"""### Cross validations

Split dataset into train, test and validation.
"""

train_processed_dataset, holdout_processed_dataset = (
    processed_dataset[processed_dataset.index < 1400].reset_index(drop=True),
    processed_dataset[processed_dataset.index >= 1400].reset_index(drop=True)
)
len(train_processed_dataset), len(holdout_processed_dataset)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, targets, embeddor, output_length, pad_label='O',
                 target_mapping={'O': [0, 0], 'B-LOC': [1, 0], 'I-LOC': [0, 1]}):
        self.texts = texts
        self.targets = targets
        self.embeddor = embeddor.wv
        self.output_length = output_length
        self.pad_label = pad_label
        self.target_mapping = target_mapping

    def __len__(self):
        return len(self.targets)

    def select_text_with_labels(self, text, labels):
        if len(text) == self.output_length:
            return (text, labels)

        if len(text) < self.output_length:
            diff = self.output_length - len(text)
            return (
                text + [""] * diff,  # Pad with empty string, because tokenizer would remove it
                labels + [self.pad_label] * diff
            )

        if len(text) > self.output_length:
            return (text[:self.output_length], labels[:self.output_length])

    def embed(self, tokens):
        embeddings = []

        for token in tokens:
            if token in self.embeddor:
                embeddings.append(self.embeddor[token])
            else:
                # If embeddor doesn't know token return vector of zeros
                embeddings.append(np.zeros(self.embeddor.vector_size))

        return np.array(embeddings)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        labels = self.targets[idx]

        tokens, labels = self.select_text_with_labels(tokens, labels)

        embeddings = self.embed(tokens)
        labels = [self.target_mapping[label] for label in labels]
        return (
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(idx)  # Return requested idx to decode used texts
        )

    def get_tokens(self, idx):
        tokens = self.texts[idx]
        labels = self.targets[idx]

        tokens, _ = self.select_text_with_labels(tokens, labels)

        return tokens


"""### Selecting output length"""

tokens_per_text = train_processed_dataset['tokens'].apply(len)

# Using 0.95 quantile as SEQ_LEN to not lose many mountain names at the end of texts.
SEQ_LEN = int(np.quantile(tokens_per_text, 0.95))

"""### Preparation for Evaluation of our model"""


def process_text(text):
    text = re.sub(r'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\b\w\b\s?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()


def metric(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tp, fp, fn, p = 0.0, 0.0, 0.0, 0.0

    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        y_true_sample = set([process_text(s) for s in y_true_sample])
        y_pred_sample = set([process_text(s) for s in y_pred_sample])

        tp += len(y_true_sample & y_pred_sample)
        fp += len(y_pred_sample - y_true_sample)
        fn += len(y_true_sample - y_pred_sample)
        p += len(y_true_sample)

    if tp + fp == 0:
        if p == 0:
            precision = 1.0
        else:
            precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        if p == 0:
            recall = 1.0
        else:
            recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def torch_loop(model, dataloader, optimizer, loss_fn, is_train=True, device='cpu'):
    if is_train:
        model.train()
    else:
        model.eval()

    predictions = []
    texts = []
    targets = []
    losses = []

    for embeddings, target, text_idx in tqdm(dataloader, total=len(dataloader), bar_format='{l_bar}{bar:100}{r_bar}'):
        embeddings, target = embeddings.to(device), target.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(embeddings)

        loss = loss_fn(pred, target)

        if is_train:
            loss.mean().backward()
            optimizer.step()

        # Apply sigmoid to prediction so values were from 0 to 1
        pred_sig = nn.functional.sigmoid(pred)

        predictions.append(pred_sig.detach().cpu().numpy())
        texts.append(text_idx.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())

    predictions = np.concatenate(predictions)
    texts = np.concatenate(texts)
    targets = np.concatenate(targets)
    losses = np.concatenate(losses)

    return predictions, texts, targets, losses


def get_labels_from_predictions(pred, th=0.5):
    labels = np.zeros(pred[:, :, 0].shape)

    # Set 1 where model predicted B-LOC
    labels += (pred[:, :, 0] > th).astype(int)

    # Remove 1 label where model predicted both B-LOC and I-LOC > threshold
    labels *= -((pred[:, :, 1] > th).astype(int) - np.ones(pred[:, :, 1].shape))

    # Set 2 where model predicted I-LOC
    labels += 2 * (pred[:, :, 1] > th).astype(int)

    return labels


def extract_words(labels, tokens):
    # Remove empty labels
    texts = np.where(labels != 0, tokens, np.full(labels.shape, '<REM>'))
    extracted_texts = []
    extracted_labels = []
    for i in range(len(texts)):
        extracted_labels.append(labels[i][labels[i] != 0])
        extracted_texts.append(texts[i][texts[i] != '<REM>'])

    # Join connected tokens
    result = []
    for i in range(len(extracted_labels)):
        result.append([])
        if len(extracted_texts[i]) > 0:
            if extracted_labels[i][0] == 1:
                seq = [extracted_texts[i][0]]
            else:
                seq = [' ' + extracted_texts[i][0]]

            for w in range(1, len(extracted_labels[i])):
                if extracted_labels[i][w] == 1:
                    result[i].append(''.join(seq))
                    seq = [extracted_texts[i][w]]
                else:
                    seq.append(' ' + extracted_texts[i][w])

            # Append the last example
            result[i].append(''.join(seq))

    return result


def get_decoded_texts(texts_ids, dataset):
    texts = []
    for text_id in texts_ids:
        texts.append(dataset.get_tokens(text_id))
    return texts


def decode(pred, texts_ids, dataset, th=0.5):
    pred_labels = get_labels_from_predictions(pred, th)

    texts = get_decoded_texts(texts_ids, dataset)

    return extract_words(pred_labels, texts), texts


def calc_metric(pred, true, texts_ids, dataset, th=0.5):
    pred_labels = get_labels_from_predictions(pred, th)
    true_labels = get_labels_from_predictions(true, 0.5)

    texts = get_decoded_texts(texts_ids, dataset)

    return metric(extract_words(true_labels, texts), extract_words(pred_labels, texts))


"""## Selecting hyperparams"""


def getDataLoaders(dataset, fold, embeddor, output_length=25):
    train_ids, test_ids = fold

    train_torch_dataset = TextDataset(
        texts=dataset.iloc[train_ids]['tokens'].reset_index(drop=True),
        targets=dataset.iloc[train_ids]['labels'].reset_index(drop=True),
        embeddor=embeddor,
        output_length=output_length
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_torch_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    test_torch_dataset = TextDataset(
        texts=dataset.iloc[test_ids]['tokens'].reset_index(drop=True),
        targets=dataset.iloc[test_ids]['labels'].reset_index(drop=True),
        embeddor=embeddor,
        output_length=output_length
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_torch_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False
    )

    return train_dataloader, train_torch_dataset, test_dataloader, test_torch_dataset


class RNN(nn.Module):
    def __init__(self, embed_dim, rnn_channels):
        super().__init__()

        self.rnns = nn.RNN(
            embed_dim,
            rnn_channels,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
            nonlinearity='relu'
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            # 2 output channels: one for B-LOC prob and another for I-LOC
            nn.Linear(2 * rnn_channels, 2)
        )

    def forward(self, x):
        x = self.rnns(x)[0]
        return self.classifier(x)

    # Because BCEWithLogitsLoss is used as loss function, forward method doesn't apply sigmoid to model's predictions while training,
    # but pred method will apply sigmoid so model output would be between 0 and 1.

    def pred(self, x):
        #         Applyes sigmoid to predictions
        x = self(x)
        x = self.classifier(x)
        return nn.functional.sigmoid(x)


def get_train_elements():
    model = RNN(
        300,  # Word vector dims
        512  # hidden size
    ).to(global_device)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optim, loss_fn


# Load pre-trained Word2Vec model
sentences = processed_dataset['tokens'].to_numpy()
embeddor = Word2Vec(sentences=sentences, vector_size=300, epochs=50)

# Load the checkpoint
checkpoint = torch.load('saved_weights/simpleRNN.pt', map_location=global_device)

model, optim, loss_fn = get_train_elements()
# Load the model
model.load_state_dict(checkpoint['model_state_dict'])
model.to(global_device)  # Ensure the model is on the correct device

# Load the optimizer
optim.load_state_dict(checkpoint['optimizer_state_dict'])

# Load the loss function
loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])


val_torch_dataset = TextDataset(
    texts = holdout_processed_dataset['tokens'],
    targets = holdout_processed_dataset['labels'],
    embeddor=embeddor,
    output_length = SEQ_LEN
)
val_dataloader = torch.utils.data.DataLoader(
    val_torch_dataset,
    batch_size=64
)

val_pred, val_texts, val_targets, val_losses = torch_loop(model, val_dataloader, optim, loss_fn, device=global_device, is_train=False)

val_metric = calc_metric(val_pred, val_targets, val_texts, val_torch_dataset, th=0.08)
val_loss = val_losses.mean()

decoded_targets, decoded_texts = decode(val_targets, val_texts, val_torch_dataset, th=0.08)
decoded_pred, _ = decode(val_pred, val_texts, val_torch_dataset, th=0.08)

# Visualize prediction probability distribution
plt.hist(val_pred.flatten(), bins=10)
plt.title('Prediction proba distribution')
plt.show()

print(f"Holdout metrics: {val_metric}")
print(f"Holdout loss: {val_loss}")

print("Holdout texts sample:")
print(np.array([' '.join(s) for s in decoded_texts[:5]]))
print("Texts targets:")
print(decoded_targets[:25])
print("Texts predicted mountains")
print(decoded_pred[:25])