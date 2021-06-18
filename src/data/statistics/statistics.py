
from src.data.class_infos import Instance as classes_info
import json
import os
import math

#a
def count_data(preprocessed_base_dir):
    ids = classes_info.get_total_class_ids()
    results = {}
    for class_id in ids:
        class_name = classes_info.get_class_name(class_id)
        path = os.path.join(preprocessed_base_dir, f"{class_name}_preprocessed.json")
        with open(path) as fp:
            data = json.load(fp)
            results[class_id] = len(data)
    return results
#b
def count_tokens(preprocessed_base_dir):
    results = {}
    ids = classes_info.get_total_class_ids()
    for class_id in ids:
        class_name = classes_info.get_class_name(class_id)
        path = os.path.join(preprocessed_base_dir, f"{class_name}_preprocessed.json")
        with open(path) as fp:
            data = json.load(fp)
            words_count = 0
            for tweet in data.values():
                words_count += len(tweet)
            results[class_id] = words_count
    return results

def count_unique_tokens(preprocessed_base_dir):
    results = {}
    ids = classes_info.get_total_class_ids()
    for class_id in ids:
        class_name = classes_info.get_class_name(class_id)
        path = os.path.join(preprocessed_base_dir, f"{class_name}_preprocessed.json")
        with open(path) as fp:
            data = json.load(fp)
            unique_tokens = dict()
            for tweet in data.values():
                for token in tweet:
                    unique_tokens[token] = 1 + unique_tokens.get(token, 0)
            results[class_id] = unique_tokens
    return results
#c
def unique_tokens(preprocessed_base_dir):
    results = {}
    ids = classes_info.get_total_class_ids()
    for class_id in ids:
        class_name = classes_info.get_class_name(class_id)
        path = os.path.join(preprocessed_base_dir, f"{class_name}_preprocessed.json")
        with open(path) as fp:
            data = json.load(fp)
            unique_tokens = set()
            for tweet in data.values():
                for token in tweet:
                    unique_tokens.add(token)
            results[class_id] = unique_tokens
    return results
#d
def common_tokens(preprocessed_base_dir):
    unique_tokens_ = unique_tokens(preprocessed_base_dir)
    ids = classes_info.get_total_class_ids()
    results = {}
    for i in range(len(ids)):
        for j in range(i + 1 , len(ids)):
            id1, id2 = ids[i], ids[j]
            key = (id1, id2)
            results[key] = unique_tokens_[id1].intersection(unique_tokens_[id2])
    return results
#e
def uncommon_tokens(preprocessed_base_dir):
    unique_tokens_ = unique_tokens(preprocessed_base_dir)
    ids = classes_info.get_total_class_ids()
    results = {}
    for i in range(len(ids)):
        for j in range(len(ids)):
            if(i == j):
                continue
            id1, id2 = ids[i], ids[j]
            key = (id1, id2)
            results[key] = unique_tokens_[id1].difference(unique_tokens_[id2])
    return results
#f
def most_repeated_uncommon_tokens(preprocessed_base_dir):
    tokens_count = count_unique_tokens(preprocessed_base_dir)
    unique_token_sets = unique_tokens(preprocessed_base_dir)
    ids = classes_info.get_total_class_ids()
    result = {}
    for id in ids:
        uncommon_token = unique_token_sets[id]
        for other in ids:
            if(id == other):
                continue 
            uncommon_token = uncommon_token.difference(unique_token_sets[other])
        sorted_tokens = sorted(list(uncommon_token), key=lambda item: tokens_count[id][item])[::-1]
        result[id] = [(item, tokens_count[id][item]) for item in sorted_tokens]
    return result

def RelativeNormalizeFrequency(label1_words_count: dict, label2_words_count: dict):
    label1_wordset = set(label1_words_count.keys())
    label2_wordset = set(label2_words_count.keys())
    intersection = label1_wordset.intersection(label2_wordset)
    sum_label1 = sum(label1_words_count.values())
    sum_label2 = sum(label2_words_count.values())
    result = {}
    for w in intersection:
        result[w] = (label1_words_count[w] / sum_label1) / (label2_words_count[w] / sum_label2)
    return result
#g
def common_tokens_relfreq(preprocessed_base_dir):
    tokens_count = count_unique_tokens(preprocessed_base_dir)
    ids = classes_info.get_total_class_ids()
    results = {}
    for i in range(len(ids)):
        for j in range(i + 1 , len(ids)):
            id1, id2 = ids[i], ids[j]
            key = (id1, id2)
            relfreq = RelativeNormalizeFrequency(tokens_count[id1], tokens_count[id2])
            words = sorted(list(relfreq.keys()), key=lambda item: relfreq[item])[::-1]
            results[key] = [(word, relfreq[word]) for word in words]
    return results
#i
def unqiue_tokens_total_count(preprocessed_base_dir):
    results = {}
    ids = classes_info.get_total_class_ids()
    for class_id in ids:
        class_name = classes_info.get_class_name(class_id)
        path = os.path.join(PREPROCCESSED_DATA_DIR, f"{class_name}_preprocessed.json")
        with open(path) as fp:
            data = json.load(fp)
            for tweet in data.values():
                for token in tweet:
                    results[token] = 1 + results.get(token, 0)
    return results
def TermFrequency(document_words_count):
    tf ={}
    sum_doc = sum(document_words_count.values())
    for w in document_words_count:
        tf[w] = document_words_count[w] / sum_doc
    return tf
def InverseDocumentFrequency(docs_wrods, target_labels_words):
    idf = {}
    N = len(docs_wrods)
    for w in target_labels_words:
        d = 0
        for doc in docs_wrods:
            if w in docs_wrods[doc]:
                d += 1
        idf[w] = math.log2(N / d)
    return idf
def Tf_Idf(tf, idf):
    tfidf = {}
    for w in tf:
        tfidf[w] = tf[w] * idf[w]
    return tfidf

def tokens_tfidf(preprocessed_base_dir):
    tokens_count = count_unique_tokens(preprocessed_base_dir)
    ids = classes_info.get_total_class_ids()
    results = {}
    for class_id in ids:
        tf = TermFrequency(tokens_count[class_id])
        idf = InverseDocumentFrequency(tokens_count, tokens_count[class_id])
        tfidf = Tf_Idf(tf, idf)
        results[class_id] = tfidf
    return results
#h
def sorted_words_tfidf(preprocessed_base_dir):
    tfidf = tokens_tfidf(preprocessed_base_dir)
    results = {}
    for class_id in tfidf:
        sorted_words = sorted(list(tfidf[class_id].keys()), key= lambda item: tfidf[class_id][item])[::-1]
        results[class_id] = [(word, tfidf[class_id][word]) for word in sorted_words]
    return results