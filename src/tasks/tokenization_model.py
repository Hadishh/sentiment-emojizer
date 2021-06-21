from src.data.class_infos import Instance as classes_info
from src.constants import PREPROCCESSED_DATA_DIR, TOTAL_ORDERED_DATA_FILE
from src.tokenization.Tokenizer import Tokenizer
import os
if __name__ == "__main__":
    vocab_size = 50
    model_type = 'unigram'
    data_url = os.path.join(PREPROCCESSED_DATA_DIR, f"{TOTAL_ORDERED_DATA_FILE}.json")
    model = Tokenizer(vocab_size, data_url, model_type)
    model.train([])