import tensorflow as tf
import numpy as np
import math

from open_seq2seq.parts.transformer.utils import get_position_encoding
from .decoder import Decoder
from open_seq2seq.parts.deepvoice.utils import conv_block, glu
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv, conv_bn_res_bn_actv
from open_seq2seq.parts.rnns.attention_wrapper import _maybe_mask_score
from open_seq2seq.parts.convs2s import ffn_wn_layer

def attention_block(queries,
                    keys,
                    vals,
                    attn_size,
                    emb_size,
                    key_lens,
                    layer,
                    prev_max_attentions=None,
                    keep_prob=0.95,
                    last_attended=None,
                    training=True,
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
    query_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=q_in_dim,
        out_dim=attn_size,
        dropout=keep_prob,
        var_scope_name="query_fc",
        mode="train",
        normalization_type="weight_norm",
    )
    key_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=k_in_dim,
        out_dim=attn_size,
        dropout=keep_prob,
        var_scope_name="key_fc",
        mode="train",
        normalization_type="weight_norm",
        init_weights=query_fc.V.initialized_value()
    )
    val_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=k_in_dim,
        out_dim=attn_size,
        dropout=keep_prob,
        var_scope_name="val_fc",
        mode="train",
        normalization_type="weight_norm",
    )

    queries = query_fc(queries)
    keys = key_fc(keys)
    vals = val_fc(vals)

    with tf.variable_scope("alignments"):
      attention_weights = tf.matmul(queries, keys, transpose_b=True)  # (N, Ty/r, Tx)

      # Key Masking
      mask_values = -np.inf * tf.ones_like(attention_weights)

      if training:
        score_mask = tf.sequence_mask(key_lens, maxlen=Tx) # (N, Tx)
        score_mask = tf.tile(tf.expand_dims(score_mask, 1), [1, Ty, 1]) # (N, Ty, Tx)
        attention_weights = tf.where(score_mask, attention_weights, mask_values)
      else: # infer
        # Create a mask that starts from the last attended-to + window size
        mask = tf.sequence_mask(prev_max_attentions, Tx)
        reverse_mask = tf.sequence_mask(Tx - prev_max_attentions - window_size, Tx)[:,::-1]
        infer_mask = tf.logical_or(mask, reverse_mask)
        infer_mask = tf.tile(tf.expand_dims(infer_mask, 1), [1, Ty])
        attention_weights = tf.where(tf.where(infer_mask, False), attention_weights, mask_values)

      alignments = tf.nn.softmax(attention_weights)
      max_attentions = tf.argmax(alignments, -1) # (N, Ty/r)

    with tf.variable_scope("context"):
      ctx = tf.nn.dropout(alignments, keep_prob)
      ctx = tf.matmul(ctx, vals)  # (N, Ty/r, a)
      ctx *= tf.rsqrt(tf.to_float(Tx))

    # Restore shape for residual connection
    output_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=attn_size,
        out_dim=emb_size,
        dropout=keep_prob,
        var_scope_name="attn_output_ffn_wn",
        mode="train",
        normalization_type="weight_norm",
    )
    tensor = output_fc(ctx)

    # returns the alignment of the first one
    alignments = alignments[0]  # (Tx, Ty)

  return tensor, alignments, max_attentions

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
            'speaker_emb': None,
            'reduction_factor': None,
            'window_size': int
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
    max_attentions_list = []
    reduction_factor = self.params['reduction_factor']

    if reduction_factor == None:
      reduction_factor = 1

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
    

    # Dropout on mel_input
    mel_inputs = tf.nn.dropout(mel_inputs, self.params['keep_prob'])

    # ----- Positional Encoding ------------------------------------------
    max_key_len = tf.shape(key)[1]
    max_query_len = tf.shape(mel_inputs)[1]

    position_rate = tf.cast(max_query_len / max_key_len, dtype=tf.float32)

    key_pe = get_position_encoding(
        length=max_key_len, 
        hidden_size=self.params['emb_size'], 
        position_rate=position_rate
    )
    query_pe = get_position_encoding(
        length=max_query_len, 
        hidden_size=self.params['channels']
    )

    # ----- Decoder PreNet -----------------------------------------------
    with tf.variable_scope("decoder_prenet", reuse=tf.AUTO_REUSE):
      for i, num in enumerate(self.params['prenet_layers']):
        mel_inputs = tf.nn.dropout(mel_inputs, self.params['keep_prob'])

        if i == 0:
          in_dim = self._n_feats * reduction_factor
        else:
          in_dim = self.params['prenet_layers'][i-1]

        dense_layer = ffn_wn_layer.FeedFowardNetworkNormalized(
            in_dim=in_dim,
            out_dim=num,
            dropout=self.params['keep_prob'],
            var_scope_name="decoder_prenet_fc_{}".format(i),
            mode="train",
            normalization_type="weight_norm"
        )
        mel_inputs = tf.nn.relu(dense_layer(mel_inputs))

    # [B, Ty, e]
    conv_feats = mel_inputs

    # ----- Conv/Attn Blocks ---------------------------------------------
    with tf.variable_scope("decoder_layers", reuse=tf.AUTO_REUSE):
      for layer in range(self.params['decoder_layers']):

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

        queries *= tf.sqrt(0.5)
        residual = queries
        key += key_pe
        queries += query_pe

        tensor, alignments, max_attentions = attention_block(
            queries=queries,
            keys=key,
            vals=value,
            attn_size=self.params['attention_size'],
            emb_size=self.params['emb_size'],
            key_lens=key_lens,
            layer=layer
        )

        conv_feats = (tensor + residual) * tf.sqrt(0.5)
        alignments_list.append(alignments)
        max_attentions_list.append(max_attentions)
    decoder_output = conv_feats

    # ----- Decoder Postnet ---------------------------------------------
    stop_token_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=self.params['emb_size'],
        out_dim=1,
        dropout=self.params['keep_prob'],
        var_scope_name="stop_token_proj",
        mode="train",
        normalization_type="weight_norm"
    )

    stop_token_logits = stop_token_fc(decoder_output)
    stop_token_predictions = tf.nn.sigmoid(stop_token_logits)

    mel_output_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=self.params['emb_size'],
        out_dim=self._n_feats * reduction_factor,
        dropout=self.params['keep_prob'],
        var_scope_name="mel_proj",
        mode="train",
        normalization_type="weight_norm"
    )    
    mel_logits = mel_output_fc(decoder_output)

    return {
        'outputs': [
            mel_logits, 
            stop_token_predictions, 
            alignments_list,
            key_lens,
            max_attentions_list
        ],
    }