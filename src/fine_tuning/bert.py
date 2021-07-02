import pytorch_lightning as pl
from transformers import BertForMaskedLM, AdamW, BertModel
from src.constants import BERT_RAW_MODEL_CACHE_DIR, BERT_VERSION
import torch.nn as nn
from src.logger.logger import log
import torch
class BertLM(pl.LightningModule):

    def __init__(self, class_name):
        super().__init__()
        self.__model_name = BERT_VERSION
        self.bert = BertForMaskedLM.from_pretrained(self.__model_name, cache_dir=BERT_RAW_MODEL_CACHE_DIR)
        self.epoch_number = 0
        self.class_name = class_name

    def forward(self, input_ids, labels):
        return self.bert(input_ids=input_ids,labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = self(input_ids=input_ids, labels=labels)
        loss = outputs[0]
        return {"loss": loss}
    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        mean_loss = 0
        n_batch  = len(outputs)
        for i in range(n_batch):
            mean_loss += outputs[i]['loss'].cpu().numpy() / n_batch
        log(f"End of epoch {self.epoch_number} with mean loss '{mean_loss}' on label {self.class_name}.", "fine_tuning")
        self.epoch_number += 1
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)

class BertLMPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.__model_name = BERT_VERSION
        self.bert = BertForMaskedLM.from_pretrained(self.__model_name, cache_dir=BERT_RAW_MODEL_CACHE_DIR)


    def forward(self, input_ids, labels=None):
        return self.bert(input_ids=input_ids,labels=labels)


class BertCLF(pl.LightningModule):

    def __init__(self, n_class, p_dropout=0.3):
        super().__init__()
        self.__model_name = BERT_VERSION
        self.bert = BertModel.from_pretrained(self.__model_name, cache_dir=BERT_RAW_MODEL_CACHE_DIR)
        self.dropout = nn.Dropout(p=p_dropout)
        self.layer = nn.Linear(768, n_class)
        self.out = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.epoch_number = 0

    def forward(self, input_ids, attention_mask, labels):
        _, pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.layer(pooler_output)
        output = self.out(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch['attention_mask']
        labels = batch["labels"]
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        mean_loss = 0
        n_batch  = len(outputs)
        for i in range(n_batch):
            mean_loss += outputs[i]['loss'].cpu().numpy() / n_batch
        log(f"End of epoch {self.epoch_number} with mean loss '{mean_loss}' on BERTClassifier.", "fine_tuning")
        self.epoch_number += 1
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)
