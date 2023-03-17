import sys
from transformers import TrainingArguments, Trainer, AutoModel, DataCollatorWithPadding, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import warnings
from datasets import Dataset, load_metric, load_dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import torch
from torch import nn
from tqdm.auto import tqdm
import csv
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


output_folder = sys.argv[1]
# Edit for your paths

base_model = 'microsoft/deberta-v3-large'


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(
        predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def preprocess_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=256)


class AI2HumanTextClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(AI2HumanTextClassification, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(base_model)
        self.pretrained_model = AutoModel.from_pretrained(base_model)
        self.hidden_size = self.config.hidden_size
        self.pre_classifier = nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.concat_classifier = nn.Linear(2*self.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.2)

    def mean_pooling(self, attention_mask, bert_output):
        hidden_state = bert_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(
            hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        mean_embeddings = mean_embeddings
        logits = self.classifier(mean_embeddings)  # regression head
        return logits

    def max_pooling(self, attention_mask, bert_output):
        hidden_state = bert_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(hidden_state.size()).float()
        hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(hidden_state, 1)[0]
        logits = self.classifier(max_embeddings)  # regression head
        return logits

    def mean_max_pooling(self, bert_output):
        hidden_state = bert_output[0]
        mean_pooling_embeddings = torch.mean(hidden_state, 1)
        _, max_pooling_embeddings = torch.max(hidden_state, 1)
        mean_max_embeddings = torch.cat(
            (mean_pooling_embeddings, max_pooling_embeddings), 1)
        logits = self.concat_classifier(
            mean_max_embeddings)  # twice the hidden size
        return logits

    def concat_pooling(self, bert_output):
        # Needs the model to set the return all hidden states to true
        all_hidden_states = bert_output[1]
        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1)
        concatenate_pooling = concatenate_pooling[:, 0]
        logits = self.concat_classifier(
            concatenate_pooling)  # twice the hidden size
        return logits

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pretrained_output = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask)
        # (batch_size, sequence_length, hidden_size)
        hidden_state = pretrained_output[0]
        pooled_output = hidden_state[:, 0]  # (batch_size, hidden_size)
        # pooled_output = self.pre_classifier(
        #     pooled_output)  # (batch_size, hidden_size)
        # pooled_output = nn.ReLU()(pooled_output)  # (batch_size, hidden_size)
        # # (batch_size, hidden_size)
        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)
        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(
                logits.view(-1, self.num_labels), labels.view(-1))

            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=pretrained_output.hidden_states, attentions=pretrained_output.attentions)
        else:
            return logits


df_train = pd.read_json(
    "./train_set.json")

n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

config = AutoConfig.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
repo_name = f"results/{output_folder}/"


model_metrics = {'eval_loss': 0, 'eval_accuracy': 0, 'eval_f1': 0}

print("\nStarting to train the model\n")

for idx, (train_idxs, val_idxs) in enumerate(skf.split(df_train["text"], df_train["label"])):
    print(f"\nFold {idx}\n")
    model = AI2HumanTextClassification(2)
    model.to(device)
    df_train_fold = df_train.iloc[train_idxs]
    df_val_fold = df_train.iloc[val_idxs]
    train_dataset = Dataset.from_pandas(df_train_fold)
    val_dataset = Dataset.from_pandas(df_val_fold)
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=repo_name,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Training fold {idx}")
    trainer.train()

    print(f"\nEvaluating fold {idx}\n")
    metrics = trainer.evaluate()
    model_metrics["eval_loss"] += metrics["eval_loss"]
    model_metrics["eval_accuracy"] += metrics["eval_accuracy"]
    model_metrics["eval_f1"] += metrics["eval_f1"]
    print(f"Fold {idx} metrics: {metrics}")
    break


for k, v in model_metrics.items():
    model_metrics[k] = v/n_splits

print(f"Global metrics : {model_metrics}")
save_folder = f"./saves/{output_folder}/"
print(f"Saving the model to ")
trainer.save_model(save_folder)


print("Getting the test dataset")
df_test = pd.read_json("./test_set.json")
test_dataset = Dataset.from_pandas(df_test)
tokenized_test = test_dataset.map(preprocess_function, batched=True)
test_dataloader = DataLoader(tokenized_test, batch_size=1)
predictions = []


print("Predicting on the test dataset")
model.eval()
for batch in test_dataloader:
    input_ids = torch.stack(batch["input_ids"]).view((1, -1))
    attention_mask = torch.stack(
        batch["attention_mask"]).view((1, -1))
    batch = {"input_ids": input_ids,
             "attention_mask": attention_mask}
    with torch.no_grad():
        outputs = model.forward(**batch)
    prediction = torch.argmax(outputs, dim=-1).item()
    predictions.append(prediction)

print("Writing the submission")
with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id', 'label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])
