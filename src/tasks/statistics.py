from src.constants import PREPROCCESSED_DATA_DIR
from src.data.statistics import statistics
from src.data.class_infos import Instance as classes_info
def compute_statistics(preprocessed_base_dir, flags="abcdefghi"):
    first_line = "=" * 10 + " Sentiment Emojizer Data Information " + "=" * 10 + "\n"
    log_str = "=" * 10 + " Sentiment Emojizer Data Information " + "=" * 10 + "\n"
    if ('a' in flags):
        data_count = statistics.count_data(preprocessed_base_dir)
        partial_str = "Label's rows count:\n"
        for key in data_count:
            class_name = classes_info.get_class_name(key)
            partial_str += f"{class_name}: {data_count[key]}\n"
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'b' in flags:
        tokens_count = statistics.count_tokens(preprocessed_base_dir)
        partial_str = "Label's tokens count:\n"
        for key in tokens_count:
            class_name = classes_info.get_class_name(key)
            partial_str += f"{class_name}: {tokens_count[key]}\n"
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n" 
    if 'c' in flags:
        tokens_count = statistics.unique_tokens(preprocessed_base_dir)
        partial_str = "Label's unique tokens count:\n"
        for key in tokens_count:
            class_name = classes_info.get_class_name(key)
            partial_str += f"{class_name}: {len(tokens_count[key])}\n"
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'd' in flags:
        common_tokens = statistics.common_tokens(preprocessed_base_dir)
        partial_str = "Label's common tokens count:\n"
        for key in common_tokens:
            id1, id2 =key
            class_name1 = classes_info.get_class_name(id1)
            class_name2 = classes_info.get_class_name(id2)
            partial_str += f"{class_name1}-{class_name2}: {len(common_tokens[key])}\n"
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'e' in flags:
        uncommon_tokens = statistics.uncommon_tokens(preprocessed_base_dir)
        partial_str = "Label's uncommon tokens count:\n"
        for key in uncommon_tokens:
            id1, id2 =key
            class_name1 = classes_info.get_class_name(id1)
            class_name2 = classes_info.get_class_name(id2)
            partial_str += f"{class_name1}-{class_name2}: {len(uncommon_tokens[key])}\n"
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
        for key in common_tokens:
            id1, id2 =key
            class_name1 = classes_info.get_class_name(id1)
            class_name2 = classes_info.get_class_name(id2)
            partial_str += f"{class_name1}-{class_name2}: {common_tokens[key][:10]}\n"
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'h' in flags:
        tokens = statistics.sorted_words_tfidf(preprocessed_base_dir)
        partial_str = "Label's tokens sorted by TF-IDF: (word, tfidf)\n"
        for key in tokens:
            class_name = classes_info.get_class_name(key)
            partial_str += f"{class_name}: {tokens[key][:10]}\n"
        log_str += partial_str
        log_str += "=" * (len(first_line) - 1) + "\n"
    if 'i' in flags:
        #TODO plot histogram and save it
        pass
    return log_str

compute_statistics(PREPROCCESSED_DATA_DIR)

import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--flags", default="abcdefhijkl")
parser.add_argument("--input", default=PREPROCCESSED_DATA_DIR)
args = parser.parse_args()

if __name__ == "__main__":
    preprocessed_base_dir = args.input
    flags = args.flags
    print(compute_statistics(preprocessed_base_dir, flags=flags))