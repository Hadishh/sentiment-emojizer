import emoji
from nltk.stem import PorterStemmer
from src.data.class_infos import Instance as classes_info
import re
import string
#a
def remove_incomplete_tweets(tweets):
    return [t for t in tweets if "..." not in t[-7:]]
#b
def remove_retweets(tweets):
    return [t for t in tweets if "RT" not in t[:3]]
#c
def remove_mentions(tweet):
    tweet = re.sub(r'@[\w_]+', '', tweet)
    return tweet
#d
def remove_hashtags(tweet):
    tweet = re.sub(r'#[\w_]+', '', tweet)
    return tweet
#e
def remove_punctuations(tweet):
    # replacements = [("'s", " is"), ("'m", ' am'), ("'re", " are"), ("won't", "will not"), ("n't", " not")]
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    return tweet
#f
def replace_newlines(tweet):
    return tweet.replace('\n', '')
#g
def stemmize(tokenized_tweet):
    ps = PorterStemmer()
    return [ps.stem(word) for word in tokenized_tweet]
#h
def frequent_emoji_removal(tokenized_tweets, removal_rate = 0.4):
    refined_tweets = []
    for tweet in tokenized_tweets:
        emojis_count = count_emojis(tweet)
        if(emojis_count / len(tweet) < removal_rate):
            refined_tweets.append(tweet)
    return refined_tweets
#i
def remove_undefined_emojis(tokenized_tweet):
    tweet = []
    
    for token in tokenized_tweet:
        if(token in emoji.UNICODE_EMOJI_ENGLISH and classes_info.get_emoji_class(token) is None):
            continue
        tweet.append(token)
    return tweet
#j
def remove_repeated_emojis(tokenized_tweet):
    tweet = []
    last_token = None
    for token in tokenized_tweet:
        if(token in emoji.UNICODE_EMOJI_ENGLISH and last_token == token):
            continue
        tweet.append(token)
        last_token = token
    return tweet

#k
def remove_emojis(tokenized_tweet):
    tweet = []
    for token in tokenized_tweet:
        if(token in emoji.UNICODE_EMOJI_ENGLISH):
            continue
        tweet.append(token)
    return tweet

#l
def remove_non_ascii(tokenized_tweet):
    tweet = []
    for token in tokenized_tweet:
        if not(token.isascii()):
            continue
        tweet.append(token)
    return tweet

#m
def remove_rt_tokens(tokenized_tweet):
    tweet = []
    for token in tokenized_tweet:
        if token != "rt":
            tweet.append(token)
    return tweet

def count_unique_emojis(text):
    emojis = set()
    for item in text:
        if(item in emoji.UNICODE_EMOJI_ENGLISH and item not in emojis):
            emojis.add(item)
    return len(emojis)

def count_emojis(tokenized_tweet):
    count = 0
    for item in tokenized_tweet:
        if(item in emoji.UNICODE_EMOJI_ENGLISH):
            count += 1
    return count

def identify_label(tokenized_tweet):
    labels = {}
    total_emojis = 0
    for token in tokenized_tweet:
        if(token in emoji.UNICODE_EMOJI_ENGLISH):
            class_id = classes_info.get_emoji_class(token)
            total_emojis += 1
            if(class_id in labels):
                labels[class_id] += 1
            else:
                labels[class_id] = 1
    class_ids = classes_info.get_total_class_ids()
    dict_ = {} 
    for id in class_ids:
        dict_[str(id)] = 0
    for label in labels:
        dict_[label] = labels[label] / total_emojis
    return dict_