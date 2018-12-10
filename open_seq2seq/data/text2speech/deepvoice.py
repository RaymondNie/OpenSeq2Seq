# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals


import os
import six
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from random import random

from six import string_types

from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary
from .speech_utils import get_speech_features_from_file,\
                          inverse_mel, normalize, denormalize

class DeepVoiceDataLayer(DataLayer):
  """
  Deep Voice 3 Data Layer
  """

  @staticmethod
  def get_required_params():
    pass

  @staticmethod
  def get_optional_params():
    pass

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """Text-to-speech data layer constructor.

    See parent class for arguments description.

    Config parameters:

    * **dataset** (str) --- The dataset to use. Currently 'LJ' for the LJSpeech
      1.1 dataset is supported.
    * **num_audio_features** (int) --- number of audio features to extract.
    * **output_type** (str) --- could be either "magnitude", or "mel".
    * **vocab_file** (str) --- path to vocabulary file.
    * **dataset_files** (list) --- list with paths to all dataset .csv files.
      File is assumed to be separated by "|".
    """

    super(DeepVoiceDataLayer, self).__init__(
        params,
        model,
        num_workers,
        worker_id
    )

    names = ['wav_filename', 'raw_transcript', 'transcript', 'phoneme_transcript']
    sep = '\x7c'
    header = None

    # Sampling parameters for the LJSpeech dataset
    self._sampling_rate = 22050
    self._n_fft = 4096

    n_feats = self.params['num_audio_features']    #n_mels = n_feats
    n_mels = n_feats

    # Character level vocab
    self.params['char2idx'] = load_pre_existing_vocabulary(
        self.params['vocab_file'],
        min_idx=5,
        read_chars=True,
    )

    # Add the pad, start, and end chars
    self.params['char2idx']['<p>'] = 0
    self.params['char2idx']['<s>'] = 1
    self.params['char2idx']['</s>'] = 2
    # Pause duration tokens?
    self.params['char2idx']['%'] = 3 # Long pause
    self.params['char2idx']['/'] = 4 # Short pause
    self.params['idx2char'] = {i: w for w, i in self.params['char2idx'].items()}
    self.params['src_vocab_size'] = len(self.params['char2idx'])

    # Load csv files
    self._files = None
    for csvs in params['dataset_files']:
      files = pd.read_csv(
          csvs,
          encoding='utf-8',
          sep=sep,
          header=header,
          names=names,
          quoting=3
      )
      if self._files is None:
        self._files = files
      else:
        self._files.append(files)
      
    if self.params['mode'] != 'infer':
      cols = ['wav_filename', 'transcript', 'phoneme_transcript']
    else:
      cols = ['transcript']

    all_files = self._files.loc[:,cols].values
    self._files = self.split_data(all_files)

    self._size = self.get_size_in_samples()
    self._dataset = None
    self._iterator = None
    self._input_tensors = None

    htk = False
    norm = 1
    self._mel_basis = librosa.filters.mel(
        sr=self._sampling_rate,
        n_fft=self._n_fft,
        n_mels=n_mels,
        htk=htk,
        norm=norm
    )

  def split_data(self, data):
    if self.params['mode'] != 'train' and self._num_workers is not None:
      size = len(data)
      start = size // self._num_workers * self._worker_id
      if self.worker_id == self._num_workers - 1:
        end = size
      else:
        end = size // self._num_workers * (self._worker_id + 1)
      data = data[start:end]
    return data

  def build_graph(self):
    """ Build data reading graph. """
    self._dataset = tf.data.Dataset.from_tensor_slices(self._files)
    if self.params['shuffle']:
      self._dataset = self._dataset.shuffle(self._size)
    self._dataset = self._dataset.repeat()

    if self.params['mode'] != 'infer':
      # Need spectrogram as target tensors
      self._dataset = self._dataset.map(
          lambda line: tf.py_func(
              self.parse_audio_transcript_element,
              [line],
              [tf.int32, tf.int32, self.params['dtype'], self.params['dtype'],\
               tf.int32],
              stateful=False
          ),
          num_parallel_calls=8
      )

      if (self.params.get("duration_max", None) or
          self.params.get("duration_min", None)):

        self._dataset = self._dataset.filter(
            lambda txt, txt_len, spec, stop, spec_len:
              tf.logical_and(
                tf.less_equal(
                  spec_len,
                  self.params.get("duration_max", 4000)
                ),
                tf.greater_equal(
                  spec_len,
                  self.params.get("duration_min", 0)
                )
            )
        )

      default_pad_value = np.log(self.params.get("data_min", 1e-5))
      pad_value = self.params.get("pad_value", default_pad_value)

      if self.params["feature_normalize"]:
        pad_value = self._normalize(pad_value)

      self._dataset = self._dataset.padded_batch(
        self.params['batch_size'],
        padded_shapes=(
            [None], 1, [None, self.params['num_audio_features']], [None], 1
        ),
        padding_values=(
            0, 0, tf.cast(pad_value, dtype=self.params['dtype']),
            tf.cast(1., dtype=self.params['dtype']), 0
        )    
      )
    else:
      self._dataset = self._dataset.map(
          lambda line: tf.py_func(
              parse_transcript_element,
              [line],
              [tf.int32, tf.int32],
              stateful=False
          ),
          num_parallel_calls=8
      )

      self._dataset = self._dataset.padded_batch(
          self.params['batch_size'],
          padded_shapes=([None], 1)
      )

    # Create iterator
    self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)\
                                  .make_initializable_iterator()

    self._input_tensors = {}

    if self.params['mode'] != 'infer':
      text, text_length, spec, stop_token_target, spec_length = self._iterator.get_next()
      spec.set_shape(
          [self.params['batch_size'], None, self.params['num_audio_features']]
      )

      stop_token_target.set_shape([self.params['batch_size'], None])
      spec_length = tf.reshape(spec_length, [self.params['batch_size']])
      self._input_tensors['target_tensors'] = [spec, stop_token_target, spec_length]
    else:
      text, text_length = self._iterator.get_next()

    text.set_shape([self.params['batch_size'], None])
    text_length = tf.reshape(text_length, [self.params['batch_size']])
    self._input_tensors['source_tensors'] = [text, text_length]

  def parse_audio_transcript_element(self, element):
    """Parses tf.data element from TextLineDataset into audio and text.

    Args:
      element: tf.data element from TextLineDataset.

    Returns:
      tuple: text_input text as `np.array` of ids, text_input length,
      target audio features as `np.array`, stop token targets as `np.array`,
      length of target sequence.

    """

    audio_filename, transcript, phoneme_transcript = element

    if six.PY2:
      audio_filename = unicode(audio_filename, "utf-8")
      transcript = unicode(transcript, "utf-8")
      phoneme_transcript = unicode(phoneme_transcript, "utf-8")
    elif not isinstance(transcript, string_types):
      audio_filename = str(audio_filename, "utf-8")
      transcript = str(transcript, "utf-8")
      phoneme_transcript = str(phoneme_transcript, "utf-8")

    # Send phoneme embedding with some fixed probability
    if random() < self.params['mixed_pronounciation_p']:
      text = phoneme_transcript
    else:
      text = transcript

    text_input = np.array(
        [self.params['char2idx'][c] for c in text]
    )

    file_path = os.path.join(
        self.params['dataset_location'], "wavs", str(audio_filename)+".wav"
    )

    spectrogram = get_speech_features_from_file(
        file_path,
        self.params['num_audio_features'],
        features_type='mel_htk',
        n_fft=self._n_fft,
        mag_power=self.params.get('mag_power', 2),
        feature_normalize=self.params["feature_normalize"],
        mean=self.params.get("feature_normalize_mean", 0.),
        std=self.params.get("feature_normalize_std", 1.),
        trim=self.params.get("trim", False),
        data_min=self.params.get("data_min", 1e-5),
        mel_basis=self._mel_basis
    )

    stop_token_target = np.zeros(
        [len(spectrogram)], dtype=self.params['dtype'].as_numpy_dtype()
    )
    stop_token_target[-1] = 1.

    return np.int32(text_input), \
           np.int32([len(text_input)]), \
           spectrogram.astype(self.params['dtype'].as_numpy_dtype()), \
           stop_token_target.astype(self.params['dtype'].as_numpy_dtype()), \
           np.int32([len(spectrogram)])

  def parse_transcript_element(self, element):
    if six.PY2: 
      transcript = unicode(transcript, "utf-8")
    elif not isinstance(transcript, string_types):
      transcript = str(transcript, "utf-8")

    text_input = np.array(
        [self.params['char2idx'][c] for c in transcript]
    )
    return np.int32(text_input), \
           np.int32([len(text_input)])

  def _normalize(self, spectrogram):
    return normalize(
        spectrogram,
        mean=self.params.get("feature_normalize_mean", 0.),
        std=self.params.get("feature_normalize_std", 1.)
    )

  def _denormalize(self, spectrogram):
    return denormalize(
        spectrogram,
        mean=self.params.get("feature_normalize_mean", 0.),
        std=self.params.get("feature_normalize_std", 1.)
    )

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)

  # TO DO: interactive infer stuff
  def create_interactive_placeholders(self):
    pass
 
  def create_feed_dict(self, model_in):
    pass

  @property
  def input_tensors(self):
    return self._input_tensors

  @property
  def sampling_rate(self):
    return self._sampling_rate

  @property
  def n_fft(self):
    return self._n_fft

  @property
  def iterator(self):
    return self._iterator

# def main():
#   params = {
#       'vocab_file':'/home/rnie/Desktop/rnie/OpenSeq2Seq/open_seq2seq/test_utils/vocab_tts.txt',
#       'mixed_pronounciation_p':0.2,
#       'num_audio_features':80,
#       'dataset_files': ['/home/rnie/Desktop/rnie/dataset/LJSpeech-1.1/metadata_processed.csv'],
#       'mode':'train',
#       'shuffle':True,
#       'dtype':tf.float32,
#       'batch_size':2,
#       'dataset_location':'/home/rnie/Desktop/rnie/dataset/LJSpeech-1.1/',
#       'feature_normalize':True
#   }
#   data_layer = DeepVoiceDataLayer(params)
#   data_layer.build_graph()

#   with tf.Session() as sess:
#     sess.run(data_layer.iterator.initializer)

#     for i in range(100):
#       print(sess.run(data_layer.input_tensors['source_tensors'][0]))
# # Testing purposes
# if __name__ == '__main__':
#   main()