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

DATA_DIR = "/kaggle/input/mountain-ner-dataset/"

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


def count_montains(labels):
    names = [1 for label in labels if label == 'B-LOC']
    return len(names)


train_processed_dataset['mountain_count'] = train_processed_dataset['labels'].apply(count_montains)
train_processed_dataset['stratify_col'] = train_processed_dataset['mountain_count']
train_processed_dataset.loc[train_processed_dataset['mountain_count'] >= 10, 'stratify_col'] = -1

skf = StratifiedKFold(n_splits=4, shuffle=True)
# gkf = GroupKFold(n_splits=4)
train_test_ids = [fold for fold in skf.split(train_processed_dataset, train_processed_dataset['stratify_col'])]

"""### Torch dataset"""

sentences = train_processed_dataset['tokens'].to_numpy()

embeddor = Word2Vec(sentences=sentences, vector_size=300, epochs=50)


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


"""## Modeling"""


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


def get_token_span(text, tokens):
    result = []
    last_index = 0
    for token in tokens:
        start_idx = text.find(token, last_index)
        if start_idx == -1:
            token.strip(string.punctuation)
            start_idx = text.find(token, last_index)

        if start_idx != -1:
            end_idx = start_idx + len(token)
            result.append((start_idx, end_idx - 1))
            last_index = end_idx
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


def get_train_elements():
    model = RNN(
        300,  # Word vector dims
        512  # hidden size
    ).to(global_device)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optim, loss_fn


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


"""### Selecting epoch num and threshold
Since dataset is very imbalanced and f1 is depends on threshold, it is better to select epoch number based on presicion-recall auc and then select treshold.
"""


def cv_epoch_stats(max_epoch_number, train_dataset, folds, embeddor, seq_len):
    epoch_fold_losses = [[] for _ in range(max_epoch_number)]
    epoch_fold_precision = [[] for _ in range(max_epoch_number)]
    epoch_fold_recall = [[] for _ in range(max_epoch_number)]
    epoch_fold_auc = [[] for _ in range(max_epoch_number)]
    epoch_fold_metric = [[] for _ in range(max_epoch_number)]
    epoch_fold_threshold = [[] for _ in range(max_epoch_number)]
    epoch_train_score = [[] for _ in range(max_epoch_number)]
    epoch_test_score = [[] for _ in range(max_epoch_number)]

    # Iterate over all cv folds
    for fold_i, fold in enumerate(folds):
        print(f'Fold {fold_i + 1}:')

        # Generate dataloaders for cv fold
        train_dataloader, train_torch_dataset, test_dataloader, test_torch_dataset = getDataLoaders(train_dataset, fold,
                                                                                                    embeddor,
                                                                                                    output_length=seq_len)

        # Create new model for each fold
        model, optim, loss_fn = get_train_elements()

        # Train model of each fold over all epochs
        for epoch in range(max_epoch_number):
            _, _, _, train_loss = torch_loop(model, train_dataloader, optim, loss_fn, device=global_device)
            test_pred, test_texts, test_targets, test_loss = torch_loop(model, test_dataloader, optim, loss_fn,
                                                                        device=global_device, is_train=False)

            train_score = train_loss.mean()
            test_score = test_loss.mean()

            precision, recall, thresholds = precision_recall_curve(test_targets.flatten(), test_pred.flatten())

            div = (precision + recall)
            div[div == 0] = 0.01
            best_threshold_id = np.argmax(2 * precision * recall / div)
            best_threshold = thresholds[best_threshold_id]

            val_metric = calc_metric(test_pred, test_targets, test_texts, test_torch_dataset, th=best_threshold)
            auc_score = auc(recall, precision)

            # We'll save model score with model score on other folds at current epoch
            epoch_fold_losses[epoch].append(test_score)
            epoch_fold_precision[epoch].append(val_metric['precision'])
            epoch_fold_recall[epoch].append(val_metric['recall'])
            epoch_fold_auc[epoch].append(auc_score)
            epoch_fold_metric[epoch].append(val_metric['f1'])
            epoch_fold_threshold[epoch].append(best_threshold)

            epoch_train_score[epoch].append(train_score)
            epoch_test_score[epoch].append(test_score)

            print(f'Epoch {epoch + 1}:')
            print(f'Validation metric {val_metric}, used threshold: {best_threshold}')
            print(f'PR-AUC: {auc_score}, best thersold: {best_threshold}')
            print(f'Train loss: {train_score}, test loss: {test_score}')

    return epoch_fold_metric, epoch_fold_auc, epoch_fold_precision, epoch_fold_recall, epoch_fold_threshold, epoch_train_score, epoch_test_score


def print_stats(fold_metrics, fold_auc, fold_precisions, fold_recalls, fold_train_score, fold_test_score, title=''):
    avg_metric = [np.mean(epoch) for epoch in fold_metrics]
    avg_auc = [np.mean(epoch) for epoch in fold_auc]
    avg_precision = [np.mean(epoch) for epoch in fold_precisions]
    avg_recall = [np.mean(epoch) for epoch in fold_recalls]
    avg_train_score = [np.mean(epoch) for epoch in fold_train_score]
    avg_test_score = [np.mean(epoch) for epoch in fold_test_score]

    #     Now we'll plot our stats

    fig, ((f1_plot, auc_plot, train_test_plot), (auc_boxplot, precision_boxplot, recall_boxplot)) = plt.subplots(2, 3,
                                                                                                                 figsize=(
                                                                                                                     15,
                                                                                                                     7))

    fig.suptitle(title)
    fig.text(0.5, 0, "Epoch", ha="center")
    fig.text(0, 0.5, "Score", va="center", rotation="vertical")

    x_labels = range(1, len(avg_metric) + 1)

    f1_plot.title.set_text("Avg f1 score over epochs")
    f1_plot.plot(x_labels, avg_metric, label="f1")
    f1_plot.plot(x_labels, avg_precision, label="precision")
    f1_plot.plot(x_labels, avg_recall, label="recall")
    f1_plot.legend()

    auc_plot.title.set_text("Avg PR-AUC score over epochs")
    auc_plot.plot(x_labels, avg_auc)

    train_test_plot.title.set_text("Train/test score")
    train_test_plot.plot(x_labels, avg_train_score, label="train score")
    train_test_plot.plot(x_labels, avg_test_score, label="test score")
    train_test_plot.legend()

    auc_boxplot.title.set_text("Auc score over epochs")
    precision_boxplot.title.set_text("Precision score over epochs")
    recall_boxplot.title.set_text("Recall score over epochs")
    auc_boxplot.boxplot(fold_auc, positions=x_labels)
    precision_boxplot.boxplot(fold_precisions, positions=x_labels)
    recall_boxplot.boxplot(fold_recalls, positions=x_labels);


print("CV for the model")
(
    fold_metrics,
    fold_auc,
    fold_precisions,
    fold_recalls,
    fold_thresholds,
    fold_train_score,
    fold_test_score
) = cv_epoch_stats(5, train_processed_dataset, train_test_ids, embeddor, SEQ_LEN)

# Average PR-AUC at each epoch over all folds
avg_auc = [np.mean(epoch) for epoch in fold_auc]

print_stats(fold_metrics, fold_auc, fold_precisions, fold_recalls, fold_train_score, fold_test_score,
            title='Dataset CV stats')

best_epoch_number = np.argmax(avg_auc) + 1
print("Best epoch: ", best_epoch_number)

best_threshold = np.mean(fold_thresholds[best_epoch_number - 1])
print("Best threshold: ", best_threshold)

"""## Training"""


def train_model(n_epoch, dataset, embeddor, seq_len):
    train_torch_dataset = TextDataset(
        texts=dataset['tokens'],
        targets=dataset['labels'],
        embeddor=embeddor,
        output_length=seq_len
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_torch_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    model, optim, loss_fn = get_train_elements()
    for epoch in range(n_epoch):
        print(f"Epoch {epoch + 1}:")
        torch_loop(model, train_dataloader, optim, loss_fn, device=global_device)

    return model, optim, loss_fn


model, optim, loss_fn = train_model(best_epoch_number, train_processed_dataset, embeddor, SEQ_LEN)

# torch.save(model.state_dict(), "saved_weights/simpleRNN.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'loss_fn_state_dict': loss_fn.state_dict()
}, "saved_weights/simpleRNN.pt")
