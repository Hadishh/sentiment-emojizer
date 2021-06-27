import argparse
from src.data.class_infos import Instance as classes_info
from src.constants import PREPROCCESSED_DATA_DIR, PREPROCESSED_DATA_SUFFIX, BERT_MODELS_DIR, BERT_SUFFIX, BERT_TOKENIZER_CACHE_DIR, BERT_VERSION
from src.fine_tuning.fine_tune import fine_tune
from transformers import BertTokenizer
import os
parser = argparse.ArgumentParser()
parser.add_argument("--ids", default="1,2,3,4,5,6,7,8")
parser.add_argument("--epochs", default=2)
parser.add_argument("--mlmprob", default=0.25)
parser.add_argument("--bs", default=8)

args = parser.parse_args()

if __name__ == "__main__":
    classes = args.ids
    epochs = args.epochs
    mlm_p = args.mlmprob
    batch_size = args.bs
    classes = [int(item) for item in classes.split(',')]
    tokenizer = BertTokenizer.from_pretrained(BERT_VERSION, cache_dir=BERT_TOKENIZER_CACHE_DIR)
    for id in classes:
        class_name = classes_info.get_class_name(id)
        data_url = os.path.join(PREPROCCESSED_DATA_DIR, f"{class_name}{PREPROCESSED_DATA_SUFFIX}.json")
        save_url = os.path.join(BERT_MODELS_DIR, f"{class_name}{BERT_SUFFIX}.bin")
        fine_tune(class_name, data_url, tokenizer, epochs=epochs, batch_size=batch_size, save_url=save_url, mlm_prob=mlm_p)
        