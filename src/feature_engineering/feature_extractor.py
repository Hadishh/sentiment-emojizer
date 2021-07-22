

def sentence_length(sentence):
    return len(sentence)

def word_length(sentence):
    lens = []
    for token in sentence:
        lens.append(len(token))
    return lens