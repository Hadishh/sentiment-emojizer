import argparse
from src.data.class_infos import Instance as classes_info
from src.constants import BERT_MODELS_DIR, BERT_SUFFIX
from src.fine_tuning.generate_text import generate
from src.fine_tuning.bert_lm import BertPred
import os
parser = argparse.ArgumentParser()
parser.add_argument("--ids", default="1,2,3,4,5,6,7,8")
args = parser.parse_args()

if __name__ == "__main__":
    classes = args.ids
    epochs = args.epochs
    mlm_p = args.mlmprob
    batch_size = args.bs
    classes = [int(item) for item in classes.split(',')]
    for id in classes:
        class_name = classes_info.get_class_name(id)
        model_url = os.path.join(BERT_MODELS_DIR, f"{class_name}{BERT_SUFFIX}.bin")
        model = BertPred(model_url)
        # fine_tune(class_name, data_url, tokenizer, epochs=epochs, batch_size=batch_size, save_url=save_url, mlm_prob=mlm_p,use_gpu=args.cuda)
        sentences = generate(10, class_name, model, seed_text='[CLS]'.split(), cuda=True)
        print(sentences)