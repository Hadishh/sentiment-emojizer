from src.constants import PREPROCCESSED_DATA_DIR, RAW_DATA_DIR
import src.utils as utils
from src.data.preprocessing.preprocessing import *
from nltk.tokenize import TweetTokenizer
import os
from src.data.class_infos import Instance as classes_info

def preprocess_data(tsv_url, flags="abcdefhijkl"):
    # utils.read_tsv()
    tweets = utils.read_tsv(tsv_url)
    print(f"start processing {tsv} with count {len(tweets)}")
    if("a" in flags):
        tweets = remove_incomplete_tweets(tweets)
    if("b" in flags):
        tweets = remove_retweets(tweets)
    if("c" in flags):
        tweets = [remove_mentions(t) for t in tweets]
    if("d" in flags):
        tweets = [remove_hashtags(t) for t in tweets]
    if("e" in flags):
        tweets = [remove_punctuations(t) for t in tweets]
    if("f" in flags):
        tweets = [replace_newlines(t) for t in tweets]
    #Tokenize
    tweets = [t.lower() for t in tweets]
    tknzr = TweetTokenizer()
    tweets = [tknzr.tokenize(tweet) for tweet in tweets]
    if("g" in flags):
        tweets = [stemmize(t) for t in tweets]
    if("h" in flags):
        tweets = frequent_emoji_removal(tweets)
    if("i" in flags):
        tweets = [remove_undefined_emojis(tweet) for tweet in tweets]
    labels = [identify_label(t) for t in tweets]
    if("j" in flags):
        tweets = [remove_repeated_emojis(t) for t in tweets]
    if('k' in flags):
        tweets = [remove_emojis(t) for t in tweets]
    if('l' in flags):
        tweets = [remove_non_ascii(t) for t in tweets]
    print(f"end processing {tsv} with count {len(tweets)}")
    return tweets, labels


import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--ids", default="1,2,3,4,5,6,7,8")
parser.add_argument("--out", default=PREPROCCESSED_DATA_DIR)
parser.add_argument("--flags", default="abcdefhijkl")
parser.add_argument("--input", default=RAW_DATA_DIR)
args = parser.parse_args()

if __name__ == "__main__":
    index = 1000000000
    classes = args.ids
    base_preproc_data = args.out
    base_input_dir = args.input
    labels_dir = os.path.join(base_preproc_data, "labels")
    classes = [int(item) for item in classes.split(',')]
    tsvs = [(os.path.join(base_input_dir, f"{classes_info.get_class_name(id)}_raw_text.tsv"), id) for id in classes]
    if not(os.path.exists(labels_dir)):
        os.mkdir(labels_dir)
    total_data = {}
    total_data_url = os.path.join(base_preproc_data, f"ordered_data.json")
    for tsv, id in tsvs:
        tweets, labels = preprocess_data(tsv, flags="abcdefhijkl")
        data_url = os.path.join(base_preproc_data, f"{classes_info.get_class_name(id)}_preprocessed.json")
        data = {}
        class_name = classes_info.get_class_name(id)
        for tweet, label in zip(tweets, labels):
            with open(os.path.join(labels_dir, f"{index}.json"), 'w') as f:
                json.dump(label, f)
            data[index] = tweet
            total_data[index] = tweet
            index += 1
        with open(data_url, 'w') as f:
            json.dump(data, f)
    with open(total_data_url, 'w') as f:
        json.dump(total_data, f)