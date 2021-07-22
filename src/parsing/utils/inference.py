from os import read
from numpy import load
from src.parsing.utils.parser_utils import Config, read_conll, load_embeddings, Parser
from src.parsing.parser_model import ParserModel
from src.constants import PARSER_TRAIN_FILE, PARSING_MODEL
import nltk
import torch
parser = None
embeddings = None

def dep_parse_sentence(sentence):
    global parser
    global embeddings
    config = Config()
    
    if parser is None:
        train_set = read_conll(PARSER_TRAIN_FILE, lowercase=config.lowercase)
        parser = Parser(train_set)
        embeddings = load_embeddings(parser)
    
    
    pos = nltk.pos_tag(sentence)
    universal_pos = [nltk.tag.map_tag('en-ptb', 'universal', tag) for _, tag in pos]
    ex = {}
    ex['word'] = sentence
    ex['pos'] = universal_pos
    ex['head'] = []
    ex['label'] = []
    model = ParserModel(embeddings)
    parser.model = model
    parser.model.load_state_dict(torch.load(PARSING_MODEL))
    dataset = parser.vectorize([ex])
    _, dep_parse = parser.parse(dataset)
    return dep_parse


parse_sentence(["on", "another", "ex", "note", "i", "can", "t", "wait", "to", "call", "my", "friend", "and", "vent", "about", "this", "latest", "hit", "up"])
