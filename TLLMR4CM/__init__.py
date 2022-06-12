from . import lmr
#from . import utils

__all__ = [ 'set_seed',
            'train',
            'evaluate']


def set_seed(args):
  lmr.set_seed(args)

def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
  lmr.train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)

def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
  lmr.evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="")
