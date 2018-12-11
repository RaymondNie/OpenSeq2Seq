# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import inspect

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.framework import ops
from open_seq2seq.parts.rnns.utils import single_cell
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv, conv_bn_res_bn_actv
from open_seq2seq.parts.convs2s.utils import gated_linear_units
from open_seq2seq.parts.deepvoice.utils import conv_block, glu

from .encoder import Encoder

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
            'keep_prob': float
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
    with tf.variable_scope("embedding"):
      text = input_dict['source_tensors'][0]

      enc_emb_w = tf.get_variable(
          "text_embeddings", 
          [src_vocab_size, self.params['emb_size']],
          dtype=self.params['dtype'],
          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
      )

      # [B, Tx, e]
      embedded_inputs = tf.cast(
          tf.nn.embedding_lookup(enc_emb_w, text),
          self.params['dtype']
      )

    # ----- Encoder PreNet -----------------------------------------------
    with tf.variable_scope("encoder_prenet"):
      # if self.params['speaker_emb'] != None:
      #   speaker_fc1 = tf.contrib.layers.fully_connected(
      #       self.params['speaker_emb'],
      #       self.params['emb_size'],
      #       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
      #                           factor=self.params['keep_prob']),
      #       activation_fn=tf.nn.softsign
      #   )
      #   speaker_fc2 = tf.contrib.layers.fully_connected(
      #       self.params['speaker_emb'],
      #       self.params['emb_size'],
      #       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
      #           factor=self.params['keep_prob']
      #       ),
      #       activation_fn=tf.nn.softsign
      #   )
      #   inputs = tf.add(embedded_text_ids, speaker_fc1);

      # [B, Tx, c]
      embedding_proj = tf.layers.dense(
          inputs=embedded_inputs,
          units=self.params['emb_size'],
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
              factor=self.params['keep_prob']
          )
      )

    conv_feats = embedding_proj
    initial_residual = embedding_proj

    # ----- Conv Blocks  --------------------------------------------------
    with tf.variable_scope("encoder_layers", reuse=tf.AUTO_REUSE):
      for layer in range(self.params['conv_layers']):

        conv_feats = tf.nn.dropout(conv_feats, self.params['keep_prob'])

        # residual for each conv block
        per_layer_residual = conv_feats

        # Kernel size should be odd to preserve sequence length with this padding
        padded_inputs = tf.pad(
            conv_feats,
            [[0, 0], [(self.params['kernel_size'] - 1) // 2, (self.params['kernel_size'] - 1) // 2], [0, 0]]
        )

        if layer == self.params['conv_layers'] - 1:
          conv_feats = conv_bn_res_bn_actv(
              layer_type="conv1d",
              name="conv_bn_res_bn_actv_{}".format(layer+1),
              inputs=padded_inputs,
              res_inputs=initial_residual,
              filters=self.params['channels'],
              kernel_size=self.params['kernel_size'],
              activation_fn=tf.nn.relu,
              strides=1,
              padding="VALID",
              regularizer=None,
              training=training,
              data_format="channels_last",
              bn_momentum=0.9,
              bn_epsilon=1e-3
          )
        else:
          conv_feats = conv_bn_actv(
              layer_type="conv1d",
              name="conv_bn_{}".format(layer+1),
              inputs=padded_inputs,
              filters=self.params['channels'],
              kernel_size=self.params['kernel_size'],
              activation_fn=tf.nn.relu,
              strides=1,
              padding="VALID",
              regularizer=None,
              training=training,
              data_format="channels_last",
              bn_momentum=0.9,
              bn_epsilon=1e-3
          )

          conv_feats += per_layer_residual

    conv_output = conv_feats
    # ----- Encoder PostNet -----------------------------------------------


    with tf.variable_scope("encoder_postnet"):
      # [B, Tx, e]

      conv_proj = tf.layers.dense(
          inputs=conv_output,
          units=self.params['emb_size'],
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
              factor=self.params['keep_prob']
          )
      )

      keys = conv_proj
      vals = tf.add(keys, embedded_inputs)

    return {"keys": keys, "vals": vals}