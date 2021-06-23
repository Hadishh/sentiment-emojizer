import argparse
from src.data.class_infos import Instance as classes_info
from src.language_model.dataset import dataset
import torch
import os

WORDS = 20
SENTENCE_COUNT = 10
TEMPERATURE = 1
USE_CUDA = False
# Set the random seed manually for reproducibility.


device = torch.device("cuda" if USE_CUDA else "cpu")

if __name__ == "__main__":
    ids = classes_info.get_total_class_ids()
    models_dir = "models/language_model/"
    reports_dir = "reports/language_model/"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    for id in ids:
        class_name = classes_info.get_class_name(id)
        model_url = models_dir + f"{class_name}.model"
        with open(model_url, 'rb') as f:
            model = torch.load(f).to(device)
        model.eval()

        corpus = dataset(f"data/preprocessed/{class_name}_preprocessed.json")
        ntokens = len(corpus.word2idx)
        hidden = model.init_hidden(1)
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
        
        out_url = reports_dir + f"{class_name}.txt"
        with open(out_url, 'w') as outf:
            with torch.no_grad():  # no tracking history
                for _ in range(SENTENCE_COUNT):
                    for i in range(WORDS):
                        output, hidden = model(input, hidden)
                        word_weights = output.squeeze().div(TEMPERATURE).exp().cpu()
                        word_idx = torch.multinomial(word_weights, 1)[0]
                        input.fill_(word_idx)
                        word = corpus.idx2word[word_idx]
                        if (word == "<EOS>" and i <= 1):
                            continue
                        if (word == "<EOS>"):
                            outf.write("$" + word + "$" + '\\\\')
                            break
                        else:
                            outf.write(word + ' ')
                    outf.write('\n')