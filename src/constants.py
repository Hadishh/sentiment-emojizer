import os
# BASE_URL = os.path.curdir()

BASE_DATA_URL =  "data" 
RAW_DATA_DIR = os.path.join(BASE_DATA_URL, 'raw') 
PREPROCCESSED_DATA_DIR = os.path.join(BASE_DATA_URL, 'preprocessed') 
STATISTICS_BASE_DIR = os.path.join(PREPROCCESSED_DATA_DIR, 'Statistics') 

CLASSES_DATA_URL = BASE_DATA_URL + "/classification_data/classes_data.csv"