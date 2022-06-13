from .lmr import set_seed, train, evaluate

#from . import utils

__all__ = [ 'set_seed',
            'train',
            'evaluate']


def set_seed(args):
  set_seed(args)

def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
  train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)

def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
  evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="")

