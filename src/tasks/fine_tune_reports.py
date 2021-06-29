import argparse
from src.data.class_infos import Instance as classes_info
from src.constants import BERT_MODELS_DIR, BERT_SUFFIX
from src.fine_tuning.generate_text import generate
from src.fine_tuning.bert_lm import BertPred
from src.logger.logger import log
import os
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--ids", default="1,2,3,4,5,6,7,8")
args = parser.parse_args()
def standardize_sentence(sent):
    sentence = []
    current_word = sent[0]
    for i in range(1, len(sent)):
        token = sent[i]
        if(token[0:2] == '##'):
            current_word += token[2:]
        else:
            sentence.append(current_word)
            current_word = token
    sentence.append(current_word)

if __name__ == "__main__":
    classes = args.ids
    reports_dir = "reports/bert_model/"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir, exist_ok=True)
    classes = [int(item) for item in classes.split(',')]
    
    for id in classes:
        class_name = classes_info.get_class_name(id)
        log(f"Generating sentences for {class_name} ...", "fine_tuning")
        model_url = os.path.join(BERT_MODELS_DIR, f"{class_name}{BERT_SUFFIX}.bin")
        model = BertPred()
        model.load_state_dict(torch.load(model_url))
        model = model.cuda()
        # fine_tune(class_name, data_url, tokenizer, epochs=epochs, batch_size=batch_size, save_url=save_url, mlm_prob=mlm_p,use_gpu=args.cuda)
        sentences = generate(10, class_name, model, max_len=10, seed_text='[CLS]'.split(), cuda=True)
        out_url = reports_dir + f"{class_name}.txt"
        with open(out_url, 'w') as outf:
            for sent in sentences:
                sent = standardize_sentence(sent)
                outf.write(str(sent)[1:-1].replace('[', '<').replace(']', '>'))
                outf.write("\\\\")
                outf.write("\n")
        log(f"Sentences for {class_name} saved to {out_url}.", "fine_tuning")
        
        print(sentences)