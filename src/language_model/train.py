import torch
import time
import math
import os
from src.language_model.dataset import dataset
from src.language_model.model import RNNModel
from src.data.class_infos import Instance as classes_info

from src.logger.logger import log
BATCH_SIZE = 20
USE_CUDA = False
EMBEDDING_SIZE = 200
HIDDEN_LAYERS = 200
LAYERS = 2
DROPOUT = 0.2
SEQUENCE_LENGTH = 35
LR = 20
EPOCHS = 10
device = torch.device("cuda" if USE_CUDA else "cpu")

def batchify(data, bsz=BATCH_SIZE):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(SEQUENCE_LENGTH, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source, model, corpus, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.word2idx)
    hidden = model.init_hidden(BATCH_SIZE)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, SEQUENCE_LENGTH):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

def train(model, corpus, train_data, criterion, lr):
     # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.word2idx)
    hidden = model.init_hidden(BATCH_SIZE)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQUENCE_LENGTH)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        # if batch % args.log_interval == 0 and batch > 0:
        #     cur_loss = total_loss / args.log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
        #             'loss {:5.2f} | ppl {:8.2f}'.format(
        #         epoch, batch, len(train_data) // args.bptt, lr,
        #         elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
        #     total_loss = 0
        #     start_time = time.time()


if __name__ == "__main__":
    ids = classes_info.get_total_class_ids()
    output_dir = "models/language_model/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for id in ids:
        class_name = classes_info.get_class_name(id)
        corpus = dataset(f"data/preprocessed/{class_name}_preprocessed.json")
        train_data = batchify(corpus.data)
        ntokens = len(corpus.word2idx)
        model = RNNModel(ntokens, EMBEDDING_SIZE, HIDDEN_LAYERS, LAYERS, DROPOUT)
        criterion = torch.nn.NLLLoss()
        lr = LR
        best_val_loss = None
        print(f"Started training on label {class_name} for {EPOCHS} epochs.")
        log(f"Started training on label {class_name} for {EPOCHS} epochs.", "language_model")
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            train(model, corpus, train_data, criterion, lr)
            val_loss = evaluate(train_data, model, corpus, criterion)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | '
                    'ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            log(' end of epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | '
                    'ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)), "language_model")
            if not best_val_loss or val_loss < best_val_loss:
                url = output_dir + class_name + ".model"
                with open(url, "wb") as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
       
        
    