import tensorflow as tf
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv

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
    keep_prob,
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
  if filters == None:
    filters = inputs.get_shape()[-1] * 2

  inputs = tf.nn.dropout(inputs, keep_prob)

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
      name="conv_{}".format(layer),
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
