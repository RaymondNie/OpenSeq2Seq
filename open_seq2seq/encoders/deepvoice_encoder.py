# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import inspect

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.framework import ops
from open_seq2seq.parts.rnns.utils import single_cell
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv
from open_seq2seq.parts.convs2s.utils import gated_linear_units

from .encoder import Encoder

def glu(inputs, speaker_emb=None):
  '''
  Deep Voice 3 GLU that supports speaker embeddings
  '''
  a, b = tf.split(inputs, 2, -1)  # (N, Tx, c) * 2
  outputs = a * tf.nn.sigmoid(b)
  return outputs

def conv_block(
    inputs, 
    layer, 
    dropout,
    filters, 
    kernel_size, 
    regularizer,
    training, 
    data_format, 
    causal=False,
    speaker_emb=None):
  '''
  Helper function to create Deep Voice 3 Conv Block
  '''
  with tf.variable_scope("conv_block"):
    if filters == None:
      filters = inputs.get_shape()[-1] * 2

    inputs = tf.nn.dropout(inputs, 1-dropout)

    if causal:
      padded_inputs = tf.pad(
          inputs,
          [[0, 0], [(kernel_size - 1), 0], [0, 0]]
      )
    else:
      # Kernel size should be odd to preserve sequence length with this padding
      padded_inputs = tf.pad(
          inputs,
          [[0, 0], [(kernel_size - 1) // 2, (kernel_size - 1) // 2], [0, 0]]
      )

    conv_out = conv_actv(
        layer_type='conv1d',
        name="conv_block_{}".format(layer),
        inputs=padded_inputs,
        filters=filters,
        kernel_size=kernel_size,
        activation_fn=None,
        strides=1,
        padding='VALID',
        regularizer=regularizer,
        training=training,
        data_format=data_format
    )

    if speaker_emb != None:
      input_shape = inputs.get_shape().as_list()
      speaker_emb = tf.contrib.layer.fully_connected(
          speaker_emb,
          input_shape[-1]//2,
          activation_fn=tf.nn.softsign
      )

    actv = glu(conv_out, speaker_emb)

    output = tf.add(inputs, actv) * tf.sqrt(0.5)

  return output


class DeepVoiceEncoder(Encoder):
  """Deep Voice 3 like encoder
  
  """ 

  @staticmethod
  def get_required_params():
    return dict(
        Encoder.get_required_params(),
        **{
            'emb_size': int,
            'conv_layers': int,
            'channels': int,
            'kernel_size': int,
            'dropout_prob': float
        }
    )

  @staticmethod
  def get_optional_params():
    return dict(
        Encoder.get_optional_params(),
        **{
            'speaker_emb': None
        }
    )

  def __init__(self, params, model, name='deepvoice3_encoder', mode='train'):
    super(DeepVoiceEncoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    """Creates TensorFlow graph for Deep Voice 3 like encoder.

    Args:
      input_dict (dict):
        source_tensors - array containing [
          * source_sequence: tensor of shape [B, Tx]
          * src_length: tensor of shape [B]
        ]

    Returns:
      dict: A python dictionary containing:
        * outputs - tensor containing the encoded text to be passed to the
          attention layer
        * src_length - the length of the encoded text
    """

    training = (self._mode == "train")
    regularizer = self.params.get('regularizer', None)
    src_vocab_size = self._model.get_data_layer().params['src_vocab_size']

    # ----- Text embedding -----------------------------------------------
    with tf.variable_scope("text_embedding"):
      text = input_dict['source_tensors'][0]
      text_len = input_dict['source_tensors'][1]

      enc_emb_w = tf.get_variable(
          "text_embeddings", 
          [src_vocab_size, self.params['emb_size']],
          dtype=self.params['dtype']
      )

      # [B, Tx, e]
      embedded_inputs = tf.cast(
          tf.nn.embedding_lookup(enc_emb_w, text),
          self.params['dtype']
      )

    # ----- Encoder PreNet -----------------------------------------------
    with tf.variable_scope("text_embedding"):
      if self.params['speaker_emb'] != None:
        speaker_fc1 = tf.contrib.layers.fully_connected(
            self.params['speaker_emb'],
            self.params['emb_size'],
            activation_fn=tf.nn.softsign
        )
        speaker_fc2 = tf.contrib.layers.fully_connected(
            self.params['speaker_emb'],
            self.params['emb_size'],
            activation_fn=tf.nn.softsign
        )

        inputs = tf.add(embedded_text_ids, speaker_fc1);

      # [B, Tx, c]
      inputs = tf.contrib.layers.fully_connected(
          embedded_inputs, 
          self.params['emb_size'],
          activation_fn=None
      )


    # ----- Conv Blocks  --------------------------------------------------
    with tf.variable_scope("conv_layers", reuse=tf.AUTO_REUSE):
      residual = inputs;

      for layer in range(self.params['conv_layers']):

        inputs = tf.nn.dropout(inputs, 1 - self.params['dropout_prob'])

        inputs = conv_block(
            inputs=inputs,
            layer=layer,      
            dropout=1-self.params['dropout_prob'],
            filters=self.params['channels'], 
            kernel_size=self.params['kernel_size'], 
            regularizer=regularizer,
            training=training, 
            data_format='channels_last',
            causal=True,
            speaker_emb=self.params['speaker_emb']
        )

    # ----- Encoder PostNet -----------------------------------------------
    with tf.variable_scope("encoder_postnet"):
      # [B, Tx, e]
      inputs = tf.contrib.layers.fully_connected(
          inputs, 
          self.params['emb_size'],
          activation_fn=None
      )

      key = inputs

      if self.params['speaker_emb'] != None:
        key = tf.add(inputs, speaker_fc2) # [B, Tx, e]

      value = tf.multiply(tf.add(key, residual), tf.sqrt(0.5)) # [B, Tx, e]
      
    return {"key": key, "value": value}