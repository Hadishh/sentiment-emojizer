from src.constants import WORD2VEC_LOG_FILE, LANGUAGE_MODEL_LOG_FILE, FINETUNE_LOG_FILES
import os
import datetime
def log(message, task):
    url = None
    if(task == "word2vec"):
        url = WORD2VEC_LOG_FILE
    if(task == "language_model"):
        url = LANGUAGE_MODEL_LOG_FILE
    if(task == "language_model"):
        url = FINETUNE_LOG_FILES
    if url is None:
        return
    if (not os.path.exists(url)):
        os.makedirs(url, exist_ok=True)
        with open(url, 'w') as f:
            f.write(f"[{datetime.datetime.now()}] {message}\n")
    else:
        with open(url, 'a') as f:
            f.write(f"[{datetime.datetime.now()}] {message}\n")
