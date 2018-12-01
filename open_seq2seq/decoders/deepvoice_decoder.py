from open_seq2seq.parts.transformer import utils
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv
from .decoder import Decoder
import tensorflow as tf

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


def attn_block(key, value, query, attn_size, emb_size, dropout_prob, is_training, dtype, speaker_emb=None):
  with tf.variable_scope("attn_block"):
    key_len = tf.shape(key)[1]
    query_len = tf.shape(query)[1]

    # Add position encoding
    key += tf.cast(
        utils.get_position_encoding(key_len, attn_size),
        dtype=dtype
    )
    query += tf.cast(
        utils.get_position_encoding(query_len, attn_size),
        dtype=dtype
    )

    # Fully connected layers
    query = tf.contrib.layers.fully_connected(
        query,
        attn_size,
        activation_fn=None
    )
    key = tf.contrib.layers.fully_connected(
        key,
        attn_size,
        activation_fn=None
    )
    value = tf.contrib.layers.fully_connected(
        value,
        attn_size,
        activation_fn=None
    )

    attn_weights = tf.matmul(query, key, transpose_b=True) # [B, Ty, Tx]
    attn_weights = tf.nn.softmax(attn_weights)
    attn_weights = tf.nn.dropout(attn_weights, dropout_prob)
      
    alignments = tf.nn.softmax(attn_weights)
    alignments = tf.transpose(alignments[0])[::-1,:] # (Nx, Ty)

    tensor = tf.matmul(attn_weights, value)
    Tx = tf.shape(attn_weights)[-1]
    tensor = tensor * tf.sqrt(tf.to_float(Tx))

    tensor = tf.contrib.layers.fully_connected(
      tensor,
      emb_size,
      activation_fn=None
    )

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
            'dropout_prob': float,
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
    key = input_dict['encoder_output']['key'] # [B, Tx, e]
    value = input_dict['encoder_output']['value'] # [B, Tx, e]
    
    # TODO: support speaker embedding
    speaker_emb = self.params.get('speaker_emb', None)

    training = (self._mode == 'train')
    
    if training:
      # [B, Ty, n]
      inputs = input_dict['target_tensors'][0] if 'target_tensors' in \
                                                    input_dict else None
      # [B]
      spec_length = input_dict['target_tensors'][2] if 'target_tensors' in \
                                                    input_dict else None

    _batch_size = input_dict['encoder_output']['key'].get_shape().as_list()[0]
      
    # ----- Decoder PreNet -----------------------------------------------
    with tf.variable_scope("decoder_prenet", reuse=tf.AUTO_REUSE):
      for i in range(self.params['prenet_layers']):
        inputs = tf.nn.dropout(inputs, 0.95)

        if speaker_emb != None:
          speaker_emb_fc = tf.contrib.layers.fully_connected(
            speaker_emb,
            self._n_feats,
            activation_fn=tf.nn.softsign
          )  

          inputs = tf.add(inputs, speaker_emb_fc)

        # [B, Ty, e]
        inputs = tf.contrib.layers.fully_connected(
            inputs,
            self.params['attention_size'],
            activation_fn=tf.nn.relu
        )

    # ----- Conv/Attn Blocks ---------------------------------------------
    with tf.variable_scope("conv_attn_blocks", reuse=tf.AUTO_REUSE):
      for i in range(self.params['decoder_layers']):
        # [B, Ty, c]
        query = conv_block(    
            inputs=inputs,
            layer=i,
            dropout=1-self.params['dropout_prob'],
            filters=None, 
            kernel_size=self.params['kernel_size'], 
            regularizer=regularizer,
            training=training, 
            data_format='channels_last',
            causal=True, 
            speaker_emb=self.params['speaker_emb']
        )

        # [B, Ty, e]
        tensor, alignments = attn_block(
            key,
            value,
            query,
            self.params['attention_size'],
            self.params['emb_size'],
            1-self.params['dropout_prob'],
            training,
            self.params['dtype'],
            speaker_emb
        )

        alignments_list.append(alignments)
        
        # residual
        inputs = tensor + query * tf.sqrt(0.5)

    decoder_output = inputs

    # Done prediction

    stop_token_projection_layer = tf.layers.Dense(
        name="stop_token_proj",
        units=1,
        use_bias=True,
    )

    stop_token_logits = stop_token_projection_layer(decoder_output)

    mel_logits = tf.contrib.layers.fully_connected(
        decoder_output,
        self._n_feats,
        activation_fn=None
    )

    return {
        'outputs': [mel_logits, alignments_list],
        'stop_token_prediction': [stop_token_logits],
    }