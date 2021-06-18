from src.data.crawler.twitter import standard_search, tweets_lookup
import csv
from src.data.class_infos import ClassesInformation
from src.constants import CLASSES_DATA_URL
import emoji
import time
import sys

class TweetsCrawler:
    def __init__(self):
        self.TEXT_DELIMITER = "\t"
        self.classesData = ClassesInformation(CLASSES_DATA_URL)
        self.query = lambda query : f"{query} -filter:links -filter:media -is:retweet"
        self.batch_count = 100
    def gather_tweets_id(self, class_id, count):
        data = []
        iters = count // self.batch_count
        if(iters * self.batch_count != count):
            iters += 1
        class_name = self.classesData.get_class_name(class_id)
        emoji_list = self.classesData.get_class_emojis(class_id)
        i = 0
        while i < iters:
            if(i % 10 == 0 and i > 0):
                print(f"gathered {len(data)} tweet ids.")
            emoji_char = emoji_list[i % len(emoji_list)]
            params= {'q' : self.query(emoji_char), 'lang':'en', 'count': min(count - len(data), self.batch_count), 'result_type' : 'mixed'} 
            r = standard_search(params)
            if(r.status_code != 200):
                if(r.status_code == 429 and r.json()['errors'][0]['code'] == 88):
                    print("Request limit exceeded. Waiting for 240 seconds to pass the twitter rate limit window...")
                    time.sleep(240)
                    print("Resuming.")
                    continue
                else:
                    raise Exception(f"Exception occured on twitter api: {r.text}")
            for item in r.json()['statuses']:
                id = item['id']
                if(id not in data):
                    data.append(f"{item['id']}")
            i += 1
        # utils.write_lines(f"{class_name}/tweets_id", data, "\n")
        return data
    def gather_tweets_text(self, id_str_list, class_id):
        class_name = self.classesData.get_class_name(class_id)
        emoji_list = self.classesData.get_class_emojis(class_id)
        iters = len(id_str_list) // self.batch_count
        texts = []
        if(len(id_str_list) % self.batch_count != 0):
            iters += 1
        for i in range(iters):
            if(i % 10 == 0 and i > 0):
                print(f"gathered {len(texts)} tweets.")
            r = tweets_lookup(id_str_list[i * self.batch_count: (i + 1) * self.batch_count])
            for item in r.json()['data']:
                texts.append(F"{item['text']}")
        # utils.write_lines(f"{class_name}/tweets_text", texts, TEXT_DELIMITER)
        return texts