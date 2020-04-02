import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


def _dataset_to_tensor(dset, mask=None):
  arr = np.asarray(dset, dtype=np.int64)
  if mask is not None:
    arr = arr[mask]
  tensor = torch.LongTensor(arr)
  return tensor


class ClevrDataset(Dataset):
  def __init__(self, question_h5, feature_h5, vocab, mode='prefix',
               max_samples=None, image_idx_start_from=None):
    mode_choices = ['prefix', 'postfix']
    if mode not in mode_choices:
      raise ValueError('Invalid mode "%s"' % mode)
    self.vocab = vocab
    self.feature_h5 = feature_h5
    self.mode = mode
    self.max_samples = max_samples

    mask = None
    if image_idx_start_from is not None:
      all_image_idxs = np.asarray(question_h5['image_idxs'])
      mask = all_image_idxs >= image_idx_start_from

    # Data from the question file is small, so read it all into memory
    print('Reading question data into memory')
    self.all_questions = _dataset_to_tensor(question_h5['questions'], mask)
    self.all_image_idxs = _dataset_to_tensor(question_h5['image_idxs'], mask)
    self.all_answers = _dataset_to_tensor(question_h5['answers'], mask)

    print('processing features')
    self.find_feature_idxs = {v : k for k, v in enumerate(feature_h5['image_idxs'])}


  def __getitem__(self, index):
    question = self.all_questions[index]
    image_idx = self.all_image_idxs[index]
    answer = self.all_answers[index]

    feats = self.feature_h5['features'][self.find_feature_idxs[image_idx]]
    feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))

    return (question, image_idx, feats, answer)

  def __len__(self):
    if self.max_samples is None:
      return self.all_questions.size(0)
    else:
      return min(self.max_samples, self.all_questions.size(0))


class ClevrDataLoader(DataLoader):
  def __init__(self, **kwargs):
    if 'question_h5' not in kwargs:
      raise ValueError('Must give question_h5')
    if 'feature_h5' not in kwargs:
      raise ValueError('Must give feature_h5')
    if 'vocab' not in kwargs:
      raise ValueError('Must give vocab')

    feature_h5_path = kwargs.pop('feature_h5')
    print('Reading features from ', feature_h5_path)
    self.feature_h5 = h5py.File(feature_h5_path, 'r')

    vocab = kwargs.pop('vocab')
    mode = kwargs.pop('mode', 'prefix')

    question_families = kwargs.pop('question_families', None)
    max_samples = kwargs.pop('max_samples', None)
    question_h5_path = kwargs.pop('question_h5')
    image_idx_start_from = kwargs.pop('image_idx_start_from', None)
    print('Reading questions from ', question_h5_path)
    with h5py.File(question_h5_path, 'r') as question_h5:
      self.dataset = ClevrDataset(question_h5, self.feature_h5, vocab, mode,
                                  max_samples=max_samples,
                                  image_idx_start_from=image_idx_start_from)
    kwargs['collate_fn'] = clevr_collate
    super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)

  def close(self):
    if self.feature_h5 is not None:
      self.feature_h5.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()


def clevr_collate(batch):
  transposed = list(zip(*batch))
  question_batch = default_collate(transposed[0])
  image_idx_batch = default_collate(transposed[1])
  feat_batch = transposed[2]
  if any(f is not None for f in feat_batch):
    feat_batch = default_collate(feat_batch)
  answer_batch = default_collate(transposed[3])
  return [question_batch, image_idx_batch, feat_batch, answer_batch]