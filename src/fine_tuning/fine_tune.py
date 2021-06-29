
import pytorch_lightning as pl
import torch
from src.fine_tuning.bert_lm import BertLM
from src.fine_tuning.dataset import MaskedLMDataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from src.logger.logger import log

def fine_tune(class_name, json_file, tokenizer, epochs, batch_size, save_url=None, mlm_prob=0.25, use_gpu=True):
    dataset = MaskedLMDataset(json_file, tokenizer)
    data_collector = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collector)
    model = BertLM(class_name)
    #using CPU
    if use_gpu:
        trainer = pl.Trainer(max_epochs=epochs, checkpoint_callback=False, logger=False, gpus=1)
    else:
        trainer = pl.Trainer(max_epochs=epochs, checkpoint_callback=False, logger=False)
    log(f"Start fine tuning BERT masked LM on class {class_name}", "fine_tuning")
    trainer.fit(model, train_loader)
    if save_url is not None:
        log(f"Finished training. Saving model in {save_url}", "fine_tuning")
        torch.save(model.state_dict(), save_url)