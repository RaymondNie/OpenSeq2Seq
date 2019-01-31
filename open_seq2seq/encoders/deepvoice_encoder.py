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
from open_seq2seq.parts.deepvoice.utils import conv_block, glu, add_regularization
from open_seq2seq.parts.convs2s import ffn_wn_layer, conv_wn_layer

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
    self.weight_norm = model.params['weight_norm']

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
    key_lens = input_dict['source_tensors'][1]
    vars_to_regularize = []

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
      # [B, Tx, c]
      if self.weight_norm:
        embedding_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
            in_dim=self.params['emb_size'],
            out_dim=self.params['channels'],
            dropout=self.params['keep_prob'],
            var_scope_name="embedding_fc",
            mode=self._mode,
            normalization_type="weight_norm",
            regularizer=regularizer,
            init_var=None,
            init_weights=None
        )
      else:
        embedding_fc = tf.layers.Dense(
            name="embedding_proj",
            units=self.params['channels'],
            use_bias=True
        )
        vars_to_regularize += embedding_fc.trainable_variables

      embedding_proj = embedding_fc(embedded_inputs)

    conv_feats = embedding_proj
    # ----- Conv Blocks  --------------------------------------------------
    with tf.variable_scope("encoder_layers", reuse=tf.AUTO_REUSE):
      for layer in range(self.params['conv_layers']):

        conv_feats = tf.nn.dropout(conv_feats, self.params['keep_prob'])
        residual = conv_feats
        
        # Kernel size should be odd to preserve sequence length with this padding
        padded_inputs = tf.pad(
            conv_feats,
            [[0, 0], [(self.params['kernel_size'] - 1) // 2, (self.params['kernel_size'] - 1) // 2], [0, 0]]
        )

        if self.weight_norm:
          # [B, Ty, c]
          conv_layer = conv_wn_layer.Conv1DNetworkNormalized(
              in_dim=self.params['channels'],
              out_dim=self.params['channels'],
              kernel_width=self.params['kernel_size'],
              mode=self._mode,
              layer_id=layer,
              hidden_dropout=self.params['keep_prob'],
              conv_padding='VALID',
              decode_padding=False,
              regularizer=regularizer
          )

          conv_feats = (conv_layer(padded_inputs) + residual) * tf.sqrt(0.5)
        else:
          conv_feats = conv_bn_res_bn_actv(
              layer_type="conv1d",
              name="conv_bn_res_bn_actv_{}".format(layer+1),
              inputs=padded_inputs,
              res_inputs=conv_feats,
              filters=self.params['channels'],
              kernel_size=self.params['kernel_size'],
              activation_fn=tf.nn.relu,
              strides=1,
              padding="VALID",
              regularizer=regularizer,
              training=training,
              data_format="channels_last",
              bn_momentum=0.9,
              bn_epsilon=1e-3
          )

    conv_output = conv_feats

    # ----- Encoder PostNet -----------------------------------------------
    with tf.variable_scope("encoder_postnet"):
      # [B, Tx, e]
      keys = tf.layers.conv1d(
          inputs=conv_output,
          filters=self.params['emb_size'],
          kernel_size=1,
          name="encoder_postnet_proj",
          kernel_regularizer=regularizer
      )
      vals = tf.add(keys, embedded_inputs) * tf.sqrt(0.5)

    add_regularization(vars_to_regularize, regularizer)

    if training == False:
      return {
        "keys": keys, 
        "vals": vals, 
        "key_lens": key_lens, 
        "mel_target": input_dict['source_tensors'][2], 
        "spec_lens": input_dict['source_tensors'][3],
        "prev_max_attentions_list" :input_dict['source_tensors'][4]
      }
    else:
      return {"keys": keys, "vals": vals, "key_lens": key_lens}