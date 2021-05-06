import os
# BASE_URL = os.path.curdir()

BASE_DATA_URL =  "data" 
RAW_DATA_URL = os.path.join(BASE_DATA_URL, 'raw') 
PREPROCCESSED_DATA_URL = os.path.join(BASE_DATA_URL, 'preprocessed') 

CLASSES_DATA_URL = BASE_DATA_URL + "/classification_data/classes_data.csv"