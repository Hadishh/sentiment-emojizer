from src.constants import PREPROCCESSED_DATA_DIR, STATISTICS_BASE_DIR
from src.data.statistics import statistics
from src.data.class_infos import Instance as classes_info
import csv
import os
def compute_statistics(preprocessed_base_dir, flags="abcdefghi", output_dir=STATISTICS_BASE_DIR):
    first_line = "=" * 10 + " Sentiment Emojizer Data Information " + "=" * 10 + "\n"
    log_str = "=" * 10 + " Sentiment Emojizer Data Information " + "=" * 10 + "\n"
    
    if ('a' in flags):
        data_count = statistics.count_data(preprocessed_base_dir)
        partial_str = "Label's rows count:\n"
        csv_columns = ['label', 'data_count']
        csv_data =[]
        for key in data_count:
            class_name = classes_info.get_class_name(key)
            partial_str += f"{class_name}: {data_count[key]}\n"
            data = {"label": class_name, 'data_count':data_count[key]}
            csv_data.append(data)
        if(output_dir is not None):
            save_csv(csv_columns, csv_data, output_dir, "DataCount")
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'b' in flags:
        tokens_count = statistics.count_tokens(preprocessed_base_dir)
        partial_str = "Label's tokens count:\n"
        csv_columns = ['label', 'token_count']
        csv_data =[]
        for key in tokens_count:
            class_name = classes_info.get_class_name(key)
            partial_str += f"{class_name}: {tokens_count[key]}\n"
            data = {"label": class_name, 'token_count': tokens_count[key]}
            csv_data.append(data)
        if(output_dir is not None):
            save_csv(csv_columns, csv_data, output_dir, "TokenCount")
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n" 
    if 'c' in flags:
        tokens_count = statistics.unique_tokens(preprocessed_base_dir)
        partial_str = "Label's unique tokens count:\n"
        csv_columns = ['label', 'token_count']
        csv_data =[]
        for key in tokens_count:
            class_name = classes_info.get_class_name(key)
            partial_str += f"{class_name}: {len(tokens_count[key])}\n"
            data = {"label": class_name, 'token_count':tokens_count[key]}
            csv_data.append(data)
        if(output_dir is not None):
            save_csv(csv_columns, csv_data, output_dir, "UniqueTokenCount")
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'd' in flags:
        common_tokens = statistics.common_tokens(preprocessed_base_dir)
        partial_str = "Label's common tokens count:\n"
        csv_columns = ['label', 'common_tokens']
        csv_data =[]
        for key in common_tokens:
            id1, id2 =key
            class_name1 = classes_info.get_class_name(id1)
            class_name2 = classes_info.get_class_name(id2)
            partial_str += f"{class_name1}-{class_name2}: {len(common_tokens[key])}\n"
            data = {"label" : f"{class_name1}-{class_name2}", "common_tokens": len(common_tokens[key])}
            csv_data.append(data)
        if(output_dir is not None):
            save_csv(csv_columns, csv_data, output_dir, "CommonTokensCount")
        log_str += partial_str

        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'e' in flags:
        uncommon_tokens = statistics.uncommon_tokens(preprocessed_base_dir)
        partial_str = "Label's uncommon tokens count:\n"
        csv_columns = ['label', 'uncommon_tokens']
        csv_data =[]
        for key in uncommon_tokens:
            id1, id2 =key
            class_name1 = classes_info.get_class_name(id1)
            class_name2 = classes_info.get_class_name(id2)
            partial_str += f"{class_name1}-{class_name2}: {len(uncommon_tokens[key])}\n"
            data = {"label" : f"{class_name1}-{class_name2}", "uncommon_tokens": len(uncommon_tokens[key])}
            csv_data.append(data)
        if(output_dir is not None):
            save_csv(csv_columns, csv_data, output_dir, "UncommonTokensCount")
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'f' in flags:
        uncommon_tokens = statistics.most_repeated_uncommon_tokens(preprocessed_base_dir)
        partial_str = "Label's most repeated uncommon tokens: (word, repeated_count)\n"
        for key in uncommon_tokens:
            class_name = classes_info.get_class_name(key)
            partial_str += f"{class_name}: {uncommon_tokens[key][:10]}\n"
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'g' in flags:
        common_tokens = statistics.common_tokens_relfreq(preprocessed_base_dir)
        partial_str = "Label's common tokens sorted by RelativeNormalizeFreq: (word, relfreq)\n"
        csv_columns = ['token', 'relfreq']
        for key in common_tokens:
            id1, id2 =key
            class_name1 = classes_info.get_class_name(id1)
            class_name2 = classes_info.get_class_name(id2)
            partial_str += f"{class_name1}-{class_name2}: {common_tokens[key][:10]}\n"
            csv_data = []
            for word, relfreq in common_tokens[key][:10]:
                csv_data.append({"token": word, "relfreq": relfreq}) 
            if (output_dir is not None):
                save_csv(csv_columns, csv_data, output_dir, f"{class_name1}-{class_name2}_RelFreq")
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'h' in flags:
        tokens = statistics.sorted_words_tfidf(preprocessed_base_dir)
        partial_str = "Label's tokens sorted by TF-IDF: (word, tfidf)\n"
        csv_columns = ['token', 'tfidf']
        for key in tokens:
            class_name = classes_info.get_class_name(key)
            partial_str += f"{class_name}: {tokens[key][:10]}\n"
            csv_data =[]
            for word, tfidf in tokens[key][:10]:
                csv_data.append({"token": word, "tfidf": tfidf}) 
            if output_dir is not None:
                save_csv(csv_columns, csv_data, output_dir, f"{class_name}_TFIDF")
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'i' in flags:
        #TODO plot histogram and save it
        pass
    return log_str

# compute_statistics(PREPROCCESSED_DATA_DIR)
def save_csv(csv_columns, csv_data, base_dir, name):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    
    path = os.path.join(base_dir, f"{name}.csv")
    with open(path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_data:
            writer.writerow(data)
    

import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--flags", type=str ,default="abcdefghijkl")
parser.add_argument("--input", type=str, default=PREPROCCESSED_DATA_DIR)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    preprocessed_base_dir = args.input
    flags = args.flags
    output_dir = args.out
    # print(output_dir)
    print(compute_statistics(preprocessed_base_dir, flags=flags, output_dir=output_dir))