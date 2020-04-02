import json
import torch

from models import LstmModel, CnnLstmModel, CnnLstmSaModel


def invert_dict(d):
  return {v: k for k, v in d.items()}


def load_vocab(path):
  with open(path, 'r') as f:
    vocab = json.load(f)
    vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
  # Sanity check: make sure <NULL>, <START>, and <END> are consistent
  assert vocab['question_token_to_idx']['<NULL>'] == 0
  assert vocab['question_token_to_idx']['<START>'] == 1
  assert vocab['question_token_to_idx']['<END>'] == 2
  return vocab


def load_cpu(path):
  """
  Loads a torch checkpoint, remapping all Tensors to CPU
  """
  return torch.load(path, map_location=lambda storage, loc: storage)


def load_model(path):
  model_cls_dict = {
    'LSTM': LstmModel,
    'CNN+LSTM': CnnLstmModel,
    'CNN+LSTM+SA': CnnLstmSaModel,
  }
  checkpoint = load_cpu(path)
  baseline_type = checkpoint['baseline_type']
  kwargs = checkpoint['baseline_kwargs']
  state = checkpoint['baseline_state']

  model = model_cls_dict[baseline_type](**kwargs)
  model.load_state_dict(state)
  return model, kwargs

