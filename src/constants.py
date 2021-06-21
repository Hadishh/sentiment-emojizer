import os
# BASE_URL = os.path.curdir()
BASE_DATA_URL =  "data" 
BASE_REPORTS_URL = "reports"
BASE_TOKENIZATION_URL = "tokenization"
PREPROCESSED_DATA_SUFFIX = "_preprocessed"
RAW_DATA_SUFFIX = "_raw"
TOTAL_ORDERED_DATA_FILE = "ordered_data"
RAW_DATA_DIR = os.path.join(BASE_DATA_URL, 'raw') 
PREPROCCESSED_DATA_DIR = os.path.join(BASE_DATA_URL, 'preprocessed') 

STATISTICS_BASE_DIR = os.path.join(PREPROCCESSED_DATA_DIR, 'Statistics') 

CLASSES_DATA_URL = BASE_DATA_URL + "/classification_data/classes_data.csv"

WORD2VEC_MODEL_DIR = "models"
WORD2VEC_SUFFIX = "word2vec"
WORD2VEC_LOG_FILE = "logs/word2ve.log"
WORD2VEC_SIM_REPORTS_URL = os.path.join(BASE_REPORTS_URL, "word2vec/similarity.csv")

TOKENIZATION_TEST_FILE = os.path.join(BASE_TOKENIZATION_URL, "test.txt")
TOKENIZATION_TRAIN_FILE = os.path.join(BASE_TOKENIZATION_URL, "train.txt")
TOKENIZATION_REPORTS_URL = os.path.join(BASE_REPORTS_URL, BASE_TOKENIZATION_URL)