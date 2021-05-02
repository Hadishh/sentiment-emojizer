from requests_oauthlib import OAuth1
import requests
import configparser
SEARCH = "https://api.twitter.com/1.1/search/tweets.json"
LOOKUP = "https://api.twitter.com/2/tweets"
def get_secret_keys():
    config = configparser.ConfigParser()
    config.read('config.ini')
    CONSUMER_KEY = config['twitter']['CONSUMER_KEY']
    CONSUMER_SECRET = config['twitter']['CONSUMER_SECRET']
    ACCESS_TOKEN = config['twitter']['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = config['twitter']['ACCESS_TOKEN_SECRET']
    return CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET

def standard_search(params):
    CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET = get_secret_keys()
    auth = OAuth1(client_key=CONSUMER_KEY, client_secret=CONSUMER_SECRET, resource_owner_key=ACCESS_TOKEN, resource_owner_secret=ACCESS_TOKEN_SECRET)
    response = requests.get(SEARCH, auth=auth, params=params)
    return response
def tweets_lookup(ids):
    params = {"ids": ",".join(ids)}
    auth = OAuth1(client_key=CONSUMER_KEY, client_secret=CONSUMER_SECRET, resource_owner_key=ACCESS_TOKEN, resource_owner_secret=ACCESS_TOKEN_SECRET)
    response = requests.get(LOOKUP, auth=auth, params=params)
    return response
