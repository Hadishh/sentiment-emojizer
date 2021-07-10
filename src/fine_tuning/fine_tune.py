
import pytorch_lightning as pl
import torch
from src.fine_tuning.bert import BertLM, BertCLF
from src.fine_tuning.dataset import MaskedLMDataset, ClassifierDataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from src.logger.logger import log

def fine_tune_LM(class_name, json_file, tokenizer, epochs, batch_size, save_url=None, mlm_prob=0.25, use_gpu=True):
    dataset = MaskedLMDataset(json_file, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
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

def fine_tune_clf(json_file, labels_dir, tokenizer, epochs, n_class, batch_size, save_url=None, use_gpu=True):
    dataset = ClassifierDataset(json_file, labels_dir, tokenizer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BertCLF(n_class)
    #using CPU
    if use_gpu:
        trainer = pl.Trainer(max_epochs=epochs, checkpoint_callback=False, logger=False, gpus=1)
    else:
        trainer = pl.Trainer(max_epochs=epochs, checkpoint_callback=False, logger=False)
    log(f"Start fine tuning BERT classifier.", "fine_tuning")
    trainer.fit(model, train_loader)
    if save_url is not None:
        log(f"Finished training. Saving model in {save_url}", "fine_tuning")
        torch.save(model.state_dict(), save_url)
