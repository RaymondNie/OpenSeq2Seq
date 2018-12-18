import tensorflow as tf
import numpy as np
from open_seq2seq.parts.transformer import utils, attention_layer
from .decoder import Decoder
from open_seq2seq.parts.deepvoice.utils import conv_block, glu
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv, conv_bn_res_bn_actv
from open_seq2seq.parts.rnns.attention_wrapper import _maybe_mask_score
import math

def positional_encoding(length,
                        num_units,
                        position_rate=1.,
                        scope="positional_encoding",
                        reuse=None):
  '''Sinusoidal Positional_Encoding.

  Args:
    inputs: A 2d Tensor with shape of (N, T).
    num_units: Output dimensionality
    position_rate: A float. Average slope of the line in the attention distribution
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
      A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
  '''
  even_cols = tf.range(start=0, limit=num_units, delta=2)
  odd_cols = tf.range(start=1, limit=num_units, delta=2)

  even_cols = tf.expand_dims(tf.to_float(1.0e4 ** (even_cols / num_units)), 0) # [1, e/2]
  odd_cols = tf.expand_dims(tf.to_float(1.0e4 ** (odd_cols / num_units)), 0) # [1, e/2]

  rows = tf.expand_dims(tf.to_float(tf.range(length)) * tf.to_float(position_rate), 1) # [T, 1]

  even_cols = tf.sin(tf.matmul(rows, 1 / even_cols)) # [T, e/2]
  odd_cols = tf.cos(tf.matmul(rows, 1 / odd_cols)) # [T, e/2]

  # Combine even and odd indicies
  positional_enc = tf.reshape(
    tf.stack([even_cols, odd_cols], axis=2),
    [length, num_units]
  ) # [T, e]

  return positional_enc


def attention_block(queries,
                    keys,
                    vals,
                    attn_size,
                    emb_size,
                    key_lens,
                    layer,
                    keep_prob=0.95,
                    last_attended=None,
                    training=False,
                    mononotic_attention=False,
                    window_size=3,
                    scope="attention_block",
                    reuse=None):
  '''Attention block.
   Args:
     queries: A 3-D tensor with shape of [batch, Ty//r, e].
     keys: A 3-D tensor with shape of [batch, Tx, e].
     vals: A 3-D tensor with shape of [batch, Tx, e].
     num_units: An int. Attention size.
     norm_type: A string. See `normalize`.
     activation_fn: A string. Activation function.
     training: A boolean. Whether or not the layer is in training mode.
     scope: Optional scope for `variable_scope`.
     reuse: Boolean, whether to reuse the weights of a previous layer
       by the same name.
  '''
  _, Ty, q_in_dim = queries.get_shape().as_list()
  _, Tx, k_in_dim = keys.get_shape().as_list()
  Ty = tf.shape(queries)[1]
  Tx = tf.shape(keys)[1]

  with tf.variable_scope("{}_{}".format(scope, layer), reuse=reuse):
    W_q = tf.get_variable(
        name="query_weights",
        shape=[q_in_dim, attn_size],
        initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=keep_prob
        )
    )

    b_q = tf.get_variable(
        name="query_bias",
        shape=[attn_size],
        initializer=tf.zeros_initializer
    )

    W_k = tf.get_variable(
        name="key_weights",
        initializer=W_q
    )

    b_k = tf.get_variable(
        name="key_bias",
        shape=[attn_size],
        initializer=tf.zeros_initializer
    )

    if layer != 0:
      W_q = tf.nn.dropout(W_q, keep_prob)
      W_k = tf.nn.dropout(W_k, keep_prob)
    with tf.variable_scope("query_proj"):
      queries = tf.matmul(tf.reshape(queries, (-1, q_in_dim)), W_q) + b_q
      queries = tf.reshape(queries, (_, Ty, attn_size))
    with tf.variable_scope("key_proj"):
      keys = tf.matmul(tf.reshape(keys,(-1, k_in_dim)), W_k) + b_k
      keys = tf.reshape(keys, (_,Tx, attn_size))
    with tf.variable_scope("value_proj"):
      vals = tf.layers.dense(
          inputs=vals,
          units=attn_size,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
              factor=keep_prob
          )
      )
    with tf.variable_scope("alignments"):
      attention_weights = tf.matmul(queries, keys, transpose_b=True)  # (N, Ty/r, Tx)
      # attention_weights = tf.Print(attention_weights, [attention_weights[0,1,key_lens[0]-1:]], message="b4 mask")

      # Key Masking
      Tx = tf.shape(attention_weights)[-1]

      score_mask = tf.sequence_mask(
          key_lens, 
          maxlen=tf.shape(keys)[1]
      ) # (N, Tx)

      score_mask = tf.tile(tf.expand_dims(score_mask, 1), [1, tf.shape(queries)[1], 1]) # (N, Ty, Tx)
      score_mask_values = -np.inf * tf.ones_like(attention_weights)

      attention_weights = tf.where(score_mask, attention_weights, score_mask_values)
      # attention_weights = tf.Print(attention_weights, [attention_weights[0,1,key_lens[0]-1:]], message="after mask")

      alignments = tf.nn.softmax(attention_weights)
      max_attentions = tf.argmax(alignments, -1) # (N, Ty/r)

    with tf.variable_scope("context"):
      ctx = tf.nn.dropout(alignments, keep_prob)
      ctx = tf.matmul(ctx, vals)  # (N, Ty/r, a)
      ctx *= tf.rsqrt(tf.to_float(Tx))

    # Restore shape for residual connection
    tensor = tf.layers.dense(
        inputs=ctx,
        units=emb_size,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=keep_prob
        )
    )
    # returns the alignment of the first one
    alignments = alignments[0]  # (Tx, Ty)

  return tensor, alignments

class DeepVoiceDecoder(Decoder):
  """
  Deepvoice 3 Decoder
  """

  @staticmethod
  def get_required_params():
    return dict(
        Decoder.get_required_params(), **{
            'prenet_layers': list,
            'decoder_layers': int,
            'keep_prob': float,
            'kernel_size': int,
            'attention_size': int,
            'emb_size': int,
            'channels': int
        }
    )

  @staticmethod
  def get_optional_params():
    return dict(
        Decoder.get_optional_params(), **{
            'speaker_emb': None
        }
    )

  def __init__(self, params, model, name='deepvoice_3_decoder', mode='train'):
    super(DeepVoiceDecoder, self).__init__(params, model, name, mode)
    self._n_feats = model.get_data_layer().params['num_audio_features']

  def _decode(self, input_dict):
    """Creates TensorFlow graph for Deep Voice 3 like decoder.

    Args:
      input_dict (dict):
        source_tensors - dictionary containing:
          * encoder_output - dictionary containing:
            * key vector of shape [B, Tx, e]
            * value vector of shape [B, Tx, e]
          * target_tensors - array containing [
            * spectrogram
            * spectrogram length
          ]
    Returns:
      dict: A python dictionary containing:
        **** TODO
    """

    regularizer = self.params.get('regularizer', None)
    alignments_list = []

    # TODO: add predicting multiple frames (r)
    key = input_dict['encoder_output']['keys'] # [B, Tx, e]
    value = input_dict['encoder_output']['vals'] # [B, Tx, e]
    key_lens = input_dict['encoder_output']['key_lens']

    # TODO: support speaker embedding
    speaker_emb = self.params.get('speaker_emb', None)

    training = (self._mode == 'train')
    
    if training:
      # [B, Ty, n]
      mel_inputs = input_dict['target_tensors'][0] if 'target_tensors' in \
                                                    input_dict else None
      # [B]
      spec_lens = input_dict['target_tensors'][2] if 'target_tensors' in \
                                                    input_dict else None

    _batch_size = input_dict['encoder_output']['keys'].get_shape().as_list()[0]
      
    # ----- Positional Encoding ------------------------------------------
    max_key_len = tf.shape(key)[1]
    max_query_len = tf.shape(mel_inputs)[1]

    position_rate = max_query_len / max_key_len

    key_pe = positional_encoding(max_key_len, self.params['emb_size'], position_rate=position_rate)
    query_pe = positional_encoding(max_query_len, self.params['channels'])

    # ----- Decoder PreNet -----------------------------------------------
    with tf.variable_scope("decoder_prenet", reuse=tf.AUTO_REUSE):
      for i, num in enumerate((self.params['prenet_layers'])):
        mel_inputs = tf.nn.dropout(mel_inputs, self.params['keep_prob'])

        # [B, Ty, e]
        mel_inputs = tf.layers.dense(
            inputs=mel_inputs,
            units=num,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=self.params['keep_prob']
            )
        )

    conv_feats = mel_inputs

    # Add positional Encoding + residual

    key += key_pe

    # ----- Conv/Attn Blocks ---------------------------------------------
    with tf.variable_scope("decoder_layers", reuse=tf.AUTO_REUSE):
      for layer in range(self.params['decoder_layers']):
        if layer != 0:
          residual = conv_feats
        # filters = conv_feats.get_shape()[-1]

        padded_inputs = tf.pad(
            conv_feats,
            [[0, 0], [(self.params['kernel_size'] - 1), 0], [0, 0]]
        )

        # [B, Ty, c]
        queries = conv_bn_res_bn_actv(
            layer_type="conv1d",
            name="conv_bn_res_bn_actv_{}".format(layer+1),
            inputs=padded_inputs,
            res_inputs=conv_feats,
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

        queries += query_pe

        tensor, alignments = attention_block(
            queries=queries,
            keys=key,
            vals=value,
            attn_size=self.params['attention_size'],
            emb_size=self.params['emb_size'],
            key_lens=key_lens,
            layer=layer
        )

        if layer != 0:
          conv_feats = tensor + residual
        else:
          conv_feats = tensor
        alignments_list.append(alignments)

    decoder_output = conv_feats

    # ----- Decoder Postnet ---------------------------------------------
    stop_token_logits = tf.layers.dense(
        inputs=decoder_output,
        units=1,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=self.params['keep_prob']
        ),
        name="stop_token_proj",
    )

    mel_logits = tf.layers.dense(
        inputs=decoder_output,
        units=self._n_feats,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=self.params['keep_prob']
        ),
        name="mel_prediction_proj",
    )
    stop_token_predictions = tf.nn.sigmoid(stop_token_logits)

    return {
        'outputs': [mel_logits, stop_token_predictions, alignments_list, key_lens],
    }