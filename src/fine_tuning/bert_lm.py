import pytorch_lightning as pl
from transformers import BertForMaskedLM, AdamW
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

class BertPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.__model_name = BERT_VERSION
        self.bert = BertForMaskedLM.from_pretrained(self.__model_name, cache_dir=BERT_RAW_MODEL_CACHE_DIR)


    def forward(self, input_ids, labels=None):
        return self.bert(input_ids=input_ids,labels=labels)