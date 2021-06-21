
import argparse
import json
from src.data.class_infos import Instance as classes_info
from src.constants import TOKENIZATION_REPORTS_URL, PREPROCCESSED_DATA_DIR,PREPROCESSED_DATA_SUFFIX
from src.utils import save_csv
from src.tokenization.Tokenizer import Tokenizer
import os
parser = argparse.ArgumentParser()
parser.add_argument("--vocab_sizes", default="50,60,70,80,90,100,500,1000,1500")
parser.add_argument("--pieces_count", default=5)
parser.add_argument("--ids", default="1,2,3,4,5,6,7,8")
args = parser.parse_args()


if __name__ == "__main__":
    vocab_sizes = args.vocab_sizes
    vocab_sizes = [int(i) for i in vocab_sizes.split(',')]
    ids = [int(item) for item in args.ids.split(',')]
    pieces_count = args.pieces_count
    csv_columns = ['vocab size'] + [f"Unks in part {i}" for i in range(pieces_count)] + ['Average']
    models = ['bpe', 'unigram']
    for model_type in models:
        for id in ids:
            class_name= classes_info.get_class_name(id)
            csv_address = os.path.join(TOKENIZATION_REPORTS_URL, f"{class_name}_{model_type}.csv")
            csv_data = []
            for vocab_size in vocab_sizes:
                data = {}
                data[csv_columns[0]] = vocab_size
                data_url = os.path.join(PREPROCCESSED_DATA_DIR, f"{class_name}{PREPROCESSED_DATA_SUFFIX}.json")
                model = Tokenizer(vocab_size, data_url, model_type)
                avg = 0
                for i in range(pieces_count):
                    model.train([i])
                    precentage = model.evaluate_uknown_precentage()
                    data[csv_columns[i + 1]] = f"{round(precentage, 3)}\\%"
                    avg += precentage / pieces_count
                data[csv_columns[-1]] = f"{round(avg, 3)}\\%"
                csv_data.append(data)
            save_csv(csv_columns, csv_data, csv_address)
                





