import csv
import emoji
from src.constants import CLASSES_DATA_URL
Instance = ClassesInformation(CLASSES_DATA_URL)
class ClassesInformation:
    def __init__(self, csv_url):
        data = []
        with open(csv_url, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                emojis = [emoji.emojize(item, language='en') for item in filter(None, row['emojis'].split('|')) if item[0] == ':' ]
                emojis = list(set(emojis))
                row['emojis'] = emojis
                row['main_emoji'] = emoji.emojize(row['main_emoji'])
                data.append(row)
        self.__data = data
    def get_class_name(self, class_id):
        return self.__data[class_id - 1]['class_name']
    def get_class_emojis(self, class_id):
        return self.__data[class_id - 1]['emojis']
    def get_total_class_ids(self):
        ids = []
        for i in range(len(self.__data)):
            ids.append(i + 1)
        return ids
    def get_emoji_class(emoji):
        for item in self.__data:
            if(emoji in item['emojis']):
                return item['id']
        return None
