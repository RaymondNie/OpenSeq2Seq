import tensorflow as tf

from open_seq2seq.parts.transformer import utils, attention_layer
from .decoder import Decoder
from open_seq2seq.parts.deepvoice.utils import conv_block, glu
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv, conv_bn_res_bn_actv

def positional_encoding(length,
                        num_units,
                        position_rate=1.,
                        zero_pad=False,
                        scale=True,
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

    odd_indices = tf.to_float(tf.range(
        start=1,
        limit=length,
        delta=2,
    ))
    even_indices = tf.to_float(tf.range(
        start=0,
        limit=length,
        delta=2,
    ))

    odd_indices = tf.expand_dims(odd_indices * position_rate, 1) # [T/2, 1]
    even_indices = tf.expand_dims(even_indices * position_rate, 1) # [T/2,1]

    timescale = tf.to_float(1.0e4 ** (tf.range(num_units) / num_units)) # [1, e]
    timescale = tf.expand_dims(timescale, 0)

    odd_indices = tf.cos(tf.matmul(odd_indices, 1 / timescale)) # [T/2, e]
    even_indices = tf.sin(tf.matmul(even_indices, 1 / timescale)) # [T/2, e]

    # Combine even and odd indicies
    positional_enc = tf.reshape(
      tf.stack([even_indices, odd_indices], axis=1),
      [length, num_units]
    ) # [T, e]

    return positional_enc

def attention_block(queries,
                    keys,
                    vals,
                    attn_size,
                    emb_size,
                    dropout_rate=0,
                    prev_max_attentions=None,
                    training=False,
                    mononotic_attention=False,
                    scope="attention_block",
                    reuse=None):
    '''Attention block.
     Args:
       queries: A 3-D tensor with shape of [batch, Ty//r, e].
       keys: A 3-D tensor with shape of [batch, Tx, e].
       vals: A 3-D tensor with shape of [batch, Tx, e].
       num_units: An int. Attention size.
       dropout_rate: A float of [0, 1]. Dropout rate.
       norm_type: A string. See `normalize`.
       activation_fn: A string. Activation function.
       training: A boolean. Whether or not the layer is in training mode.
       scope: Optional scope for `variable_scope`.
       reuse: Boolean, whether to reuse the weights of a previous layer
         by the same name.
    '''
    _keys = keys
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("query_proj"):
            queries = tf.contrib.layers.fully_connected(
                queries,
                attn_size,
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                    factor=(1. - dropout_rate)
                ),                
                activation_fn=None
            )
        with tf.variable_scope("key_proj"):
            keys = tf.contrib.layers.fully_connected(
                keys,
                attn_size,
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                    factor=(1. - dropout_rate)
                ),
                activation_fn=None
            )
        with tf.variable_scope("value_proj"):
            vals = tf.contrib.layers.fully_connected(
                vals,
                attn_size,
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                    factor=(1. - dropout_rate)
                ),
                activation_fn=None
            )
        with tf.variable_scope("alignments"):
            attention_weights = tf.matmul(queries, keys, transpose_b=True)  # (N, Ty/r, Tx)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, Tx)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (N, Ty/r, Tx)

            paddings = tf.ones_like(attention_weights) * (-2 ** 32 + 1)
            attention_weights = tf.where(tf.equal(key_masks, 0), paddings, attention_weights)  # (N, Ty/r, Tx)

            Tx = tf.shape(attention_weights)[-1]

            alignments = tf.nn.softmax(attention_weights)
            max_attentions = tf.argmax(alignments, -1) # (N, Ty/r)

        with tf.variable_scope("context"):
            ctx = tf.layers.dropout(alignments, rate=dropout_rate, training=training)
            ctx = tf.matmul(ctx, vals)  # (N, Ty/r, a)
            ctx *= tf.rsqrt(tf.to_float(Tx))

        # Restore shape for residual connection
        tensor = tf.contrib.layers.fully_connected(
            ctx,
            emb_size,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                    factor=(1. - dropout_rate)
            ),
            activation_fn=None
        )
        # returns the alignment of the first one
        alignments = tf.transpose(alignments[0])[::-1, :]  # (Tx, Ty)

    return tensor, alignments

class DeepVoiceDecoder(Decoder):
  """
  Deepvoice 3 Decoder
  """

  @staticmethod
  def get_required_params():
    return dict(
        Decoder.get_required_params(), **{
            'prenet_layers': int,
            'decoder_layers': int,
            'keep_prob': float,
            'kernel_size': int,
            'attention_size': int,
            'emb_size': int
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
    
    # TODO: support speaker embedding
    speaker_emb = self.params.get('speaker_emb', None)

    training = (self._mode == 'train')
    
    if training:
      # [B, Ty, n]
      mel_inputs = input_dict['target_tensors'][0] if 'target_tensors' in \
                                                    input_dict else None
      # [B]
      spec_length = input_dict['target_tensors'][2] if 'target_tensors' in \
                                                    input_dict else None

    _batch_size = input_dict['encoder_output']['keys'].get_shape().as_list()[0]
      
    # ----- Positional Encoding ------------------------------------------
    key_len = tf.shape(key)[1]
    query_len = tf.shape(mel_inputs)[1]

    # key_pe = utils.get_position_encoding(key_len, self.params['emb_size'])
    # query_pe = utils.get_position_encoding(query_len, tf.to_float(query_len/key_len), self.params['emb_size'])

    key_pe = positional_encoding(key_len, self.params['emb_size'])
    position_rate = tf.to_float(query_len/key_len)
    query_pe = positional_encoding(query_len, self.params['emb_size'], position_rate=position_rate)

    # ----- Decoder PreNet -----------------------------------------------
    with tf.variable_scope("decoder_prenet", reuse=tf.AUTO_REUSE):
      for i in range(self.params['prenet_layers']):
        mel_inputs = tf.nn.dropout(mel_inputs, self.params['keep_prob'])

        # [B, Ty, e]
        mel_inputs = tf.layers.dense(
            inputs=mel_inputs,
            units=self.params['emb_size'],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=self.params['keep_prob']
            )
        )

    conv_feats = mel_inputs
    key += key_pe

    # ----- Conv/Attn Blocks ---------------------------------------------
    with tf.variable_scope("decoder_layers", reuse=tf.AUTO_REUSE):
      for layer in range(self.params['decoder_layers']):

        filters = conv_feats.get_shape()[-1]

        padded_inputs = tf.pad(
            conv_feats,
            [[0, 0], [(self.params['kernel_size'] - 1), 0], [0, 0]]
        )

        # [B, Ty, c]
        queries = conv_bn_actv(
            layer_type="conv1d",
            name="conv_bn_{}".format(layer+1),
            inputs=padded_inputs,
            filters=filters,
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

        # Add positional Encoding
        queries += query_pe + conv_feats

        tensor, alignments = attention_block(
            queries=queries,
            keys=key,
            vals=value,
            attn_size=self.params['attention_size'],
            emb_size=self.params['emb_size']
        )

        alignments_list.append(alignments)
        
        # residual
        conv_feats = queries + tensor
    decoder_output = conv_feats

    # Done prediction

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
        'outputs': [mel_logits, alignments_list],
        'stop_token_prediction': stop_token_predictions,
    }