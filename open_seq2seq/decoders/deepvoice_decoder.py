import tensorflow as tf
import numpy as np
import math

from open_seq2seq.parts.transformer.utils import get_position_encoding
from .decoder import Decoder
from open_seq2seq.parts.deepvoice.utils import conv_block, glu
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv, conv_bn_res_bn_actv
from open_seq2seq.parts.rnns.attention_wrapper import _maybe_mask_score
from open_seq2seq.parts.convs2s import ffn_wn_layer, conv_wn_layer

def attention_block(queries,
                    keys,
                    vals,
                    attn_size,
                    emb_size,
                    key_lens,
                    layer,
                    prev_max_attentions=None,
                    keep_prob=0.95,
                    mode="train",
                    enforce_monotonicity=False,
                    window_size=3,
                    window_backwards=0,
                    scope="attention_block",
                    regularizer=None,
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
        mode=mode,
        normalization_type="weight_norm",
        regularizer=regularizer
    )
    key_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=k_in_dim,
        out_dim=attn_size,
        dropout=keep_prob,
        var_scope_name="key_fc",
        mode=mode,
        normalization_type="weight_norm",
        regularizer=regularizer,
        init_weights=query_fc.V.initialized_value()
    )
    val_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=k_in_dim,
        out_dim=attn_size,
        dropout=keep_prob,
        var_scope_name="val_fc",
        mode=mode,
        normalization_type="weight_norm",
        regularizer=regularizer
    )

    queries = query_fc(queries)
    keys = key_fc(keys)
    vals = val_fc(vals)

    with tf.variable_scope("alignments"):
      attention_weights = tf.matmul(queries, keys, transpose_b=True)  # (N, Ty/r, Tx)

      # Key Masking
      mask_values = -np.inf * tf.ones_like(attention_weights)

      if mode == "train" or enforce_monotonicity == False:
        score_mask = tf.sequence_mask(key_lens, maxlen=Tx) # (N, Tx)
        score_mask = tf.tile(tf.expand_dims(score_mask, 1), [1, Ty, 1]) # (N, Ty, Tx)
        attention_weights = tf.where(score_mask, attention_weights, mask_values)
      else: # infer
        # Create a mask that starts from the last attended-to + window size
        mask = tf.sequence_mask(prev_max_attentions - window_backwards, Tx)
        # mask = tf.Print(mask,[mask])
        reverse_mask = tf.sequence_mask(Tx - prev_max_attentions - window_size + window_backwards, Tx)[:,::-1]
        infer_mask = tf.logical_or(mask, reverse_mask)
        infer_mask = tf.tile(tf.expand_dims(infer_mask, 1), [1, Ty, 1])
        attention_weights = tf.where(tf.equal(infer_mask, False), attention_weights, mask_values)
      alignments = tf.nn.softmax(attention_weights)
      max_attentions = tf.argmax(alignments, -1) # (N, Ty/r)

    with tf.variable_scope("context"):
      ctx = tf.nn.dropout(alignments, keep_prob)
      ctx = tf.matmul(ctx, vals) * tf.rsqrt(tf.to_float(Tx))# (N, Ty/r, a)

    # Restore shape for residual connection
    output_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=attn_size,
        out_dim=emb_size,
        dropout=keep_prob,
        var_scope_name="attn_output_ffn_wn",
        mode=mode,
        normalization_type="weight_norm",
        regularizer=regularizer
    )
    tensor = output_fc(ctx)

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
            'channels': int,
            'pos_rate': float
        }
    )

  @staticmethod
  def get_optional_params():
    return dict(
        Decoder.get_optional_params(), **{
            'speaker_emb': None,
            'window_size': int,
            'monotonic_alignment': list
        }
    )

  def __init__(self, params, model, name='deepvoice_3_decoder', mode='train'):
    super(DeepVoiceDecoder, self).__init__(params, model, name, mode)
    self.reduction_factor = model.get_data_layer().params['reduction_factor']
    self.weight_norm = model.params['weight_norm']
    self.both = "both" in model.get_data_layer().params['output_type']
    if self.both:
      self.mel_feats = model.get_data_layer().params['num_audio_features']['mel']
      self.mag_feats = model.get_data_layer().params['num_audio_features']['magnitude']
    else:
      self.mel_feats = model.get_data_layer().params['num_audio_features']

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
    key = input_dict['encoder_output']['keys'] # [B, Tx, e]
    value = input_dict['encoder_output']['vals'] # [B, Tx, e]
    key_lens = input_dict['encoder_output']['key_lens']
    
    # TODO: support speaker embedding
    speaker_emb = self.params.get('speaker_emb', None)
    training = (self._mode == 'train')
    alignments_list = []
    max_attentions_list = []

    if training:
      # [B, Ty, n]
      mel_inputs = input_dict['target_tensors'][0] if 'target_tensors' in \
                                                    input_dict else input_dict['encoder_output']['mel_target']
      mel_inputs = tf.pad(mel_inputs, [[0,0],[1,0],[0,0]])
      mel_inputs = mel_inputs[:,:-1,:]
      # [B]
      spec_lens = input_dict['target_tensors'][2] if 'target_tensors' in \
                                                    input_dict else input_dict['encoder_output']['spec_lens']
    else:
      monotonic_alignment = self.params.get('monotonic_alignment', [True] * self.params['decoder_layers'])
      mel_inputs = input_dict['encoder_output']['mel_target']
      spec_lens = input_dict['encoder_output']['spec_lens']
      prev_max_attentions_list = input_dict['encoder_output']['prev_max_attentions_list']

    _batch_size = input_dict['encoder_output']['keys'].get_shape().as_list()[0]

    if self.both and self.reduction_factor == 1:
      mel_inputs, _ = tf.split(
          mel_inputs,
          [self.mel_feats, self.mag_feats],
          axis=2
      )
      
    # Dropout on mel_input
    mel_inputs = tf.nn.dropout(mel_inputs, 0.5)
    
    # ----- Positional Encoding ------------------------------------------
    max_key_len = tf.shape(key)[1]
    max_query_len = tf.shape(mel_inputs)[1]

    position_rate = self.params['pos_rate'] # Initial rate for single speaker?

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
      for layer, out_channels in enumerate(self.params['prenet_layers']):
        mel_inputs = tf.layers.conv1d(
            inputs=mel_inputs,
            filters=out_channels,
            kernel_size=1,
            name="decoder_prenet_proj_{}".format(layer)
        )

    # [B, Ty, e]
    conv_feats = mel_inputs

    # ----- Conv/Attn Blocks ---------------------------------------------
    with tf.variable_scope("decoder_layers", reuse=tf.AUTO_REUSE):
      for layer in range(self.params['decoder_layers']):
        
        residual = conv_feats

        padded_inputs = tf.pad(
            conv_feats,
            [[0, 0], [(self.params['kernel_size'] - 1), 0], [0, 0]]
        )

        if self.weight_norm:
          # [B, Ty, c]
          conv_layer = conv_wn_layer.Conv1DNetworkNormalized(
              in_dim=self.params['emb_size'],
              out_dim=self.params['channels'],
              kernel_width=self.params['kernel_size'],
              mode=self._mode,
              layer_id=layer,
              hidden_dropout=self.params['keep_prob'],
              conv_padding='VALID',
              decode_padding=False,
              regularizer=regularizer
          )

          queries = conv_layer(padded_inputs)
          queries = (queries[:, :tf.shape(residual)[1], :] + residual) * tf.sqrt(0.5)

        else:
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
              regularizer=regularizer,
              training=training,
              data_format="channels_last",
              bn_momentum=0.9,
              bn_epsilon=1e-3
          )

          queries *= tf.sqrt(0.5)          

        # Set residual for attention block
        residual = queries

        # Add the positional encoding
        key += key_pe
        queries += query_pe

        if training:
          prev_max_attentions = None
          enforce_monotonicity = False
        else:
          prev_max_attentions = prev_max_attentions_list[layer]
          enforce_monotonicity = monotonic_alignment[layer]

        tensor, alignments, max_attentions = attention_block(
            queries=queries,
            keys=key,
            vals=value,
            attn_size=self.params['attention_size'],
            emb_size=self.params['emb_size'],
            key_lens=key_lens,
            layer=layer,
            mode=self._mode,
            prev_max_attentions=prev_max_attentions,
            enforce_monotonicity=enforce_monotonicity,
            regularizer=regularizer
        )

        conv_feats = (tensor + residual) * tf.sqrt(0.5)
        alignments_list.append(alignments)
        max_attentions_list.append(max_attentions)

    decoder_output = conv_feats

    # ----- Decoder Postnet ---------------------------------------------

    mel_spec_prediction = tf.layers.conv1d(
        inputs=decoder_output,
        filters=self.mel_feats * self.reduction_factor,
        kernel_size=1,
        name="decoder_postnet_final_proj"
    )

    stop_token_fc = ffn_wn_layer.FeedFowardNetworkNormalized(
        in_dim=self.mel_feats * self.reduction_factor,
        out_dim=1,
        dropout=self.params['keep_prob'],
        var_scope_name="stop_token_proj",
        mode=self._mode,
        normalization_type="weight_norm",
        regularizer=regularizer
    )

    stop_token_logits = stop_token_fc(mel_spec_prediction)
    stop_token_predictions = tf.nn.sigmoid(stop_token_logits)

    if self.both:

      # ----- Converter ---------------------------------------------

      mag_spec_prediction = mel_spec_prediction
      with tf.variable_scope("converter", reuse=tf.AUTO_REUSE):
        mag_spec_prediction = conv_bn_actv(
            layer_type="conv1d",
            name="converter_conv_0",
            inputs=mag_spec_prediction,
            filters=256,
            kernel_size=4,
            activation_fn=tf.nn.relu,
            strides=1,
            padding="SAME",
            regularizer=regularizer,
            training=training,
            data_format=self.params.get('postnet_data_format', 'channels_last'),
            bn_momentum=self.params.get('postnet_bn_momentum', 0.1),
            bn_epsilon=self.params.get('postnet_bn_epsilon', 1e-5),
        )

        mag_spec_prediction = conv_bn_actv(
            layer_type="conv1d",
            name="converter_conv_1",
            inputs=mag_spec_prediction,
            filters=512,
            kernel_size=4,
            activation_fn=tf.nn.relu,
            strides=1,
            padding="SAME",
            regularizer=regularizer,
            training=training,
            data_format=self.params.get('postnet_data_format', 'channels_last'),
            bn_momentum=self.params.get('postnet_bn_momentum', 0.1),
            bn_epsilon=self.params.get('postnet_bn_epsilon', 1e-5),
        )

        if self._model.get_data_layer()._exp_mag:
          mag_spec_prediction = tf.exp(mag_spec_prediction)

        mag_spec_prediction = tf.layers.conv1d(
            mag_spec_prediction,
            self.mag_feats * self.reduction_factor,
            1,
            name="converter_post_net_proj",
            use_bias=False,
        )
    else:
      mag_spec_prediction = tf.zeros([1])

    return {
        'outputs': [
            mel_spec_prediction, 
            stop_token_predictions, 
            alignments_list,
            key_lens,
            spec_lens,
            max_attentions_list,
            mag_spec_prediction
        ],
    }
