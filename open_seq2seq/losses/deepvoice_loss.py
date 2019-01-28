# Copyright (c) 2018 NVIDIA Corporation
import tensorflow as tf
from .loss import Loss

def guided_attention(text_lens, spec_lens, batch_size, g=0.2):
    max_text_len = tf.to_float(tf.reduce_max(text_lens))
    max_spec_len = tf.to_float(tf.reduce_max(spec_lens))

    text_lens = tf.to_float(text_lens)
    spec_lens = tf.to_float(spec_lens)
    
    rows = tf.reshape(tf.tile(tf.range(max_text_len), [max_spec_len]), [max_spec_len, -1]) # Ty x Tx
    rows = tf.reshape(tf.tile(rows, [batch_size, 1]), [batch_size, max_spec_len, -1]) / tf.expand_dims(tf.expand_dims(text_lens, 1), 1) # B x Ty x Tx

    cols = tf.transpose(tf.reshape(tf.tile(tf.range(max_spec_len), [max_text_len]), [max_text_len, -1]), [1,0])
    cols = tf.reshape(tf.tile(cols, [batch_size, 1]), [batch_size, max_spec_len, -1]) / tf.expand_dims(tf.expand_dims(spec_lens, 1), 1) # B x Ty x Tx

    matrix = 1 - tf.exp(-tf.square((rows-cols) / 2 * g * g))

    return matrix

class DeepVoiceLoss(Loss):
  def __init__(self, params, model, name="deepvoice_loss"):
    super(DeepVoiceLoss, self).__init__(params, model, name)
    self._n_feats = self._model.get_data_layer().params["num_audio_features"]
    if "both" in self._model.get_data_layer().params['output_type']:
      self._both = True
    else:
      self._both = False
    self._reduction_factor = self._model.get_data_layer().params['reduction_factor']

  def get_required_params(self):
    return dict(
        Loss.get_required_params(), **{
            'l1_loss': bool
        }
    )


  def get_optional_params(self):
    return {}

  def _compute_loss(self, input_dict):
    """
    Computes the cross-entropy loss for WaveNet.
    Args:
      input_dict (dict):
        * "decoder_output": array containing: [
          * mel_output: spectrogram predicted by the decoder rnn of shape [batch, time, feats]
          * stop_token_logits: predictions for stop token [batch x time]
        ]
    """
    # Get Decoder outputs and output targets
    post_net_predictions = input_dict['decoder_output']['outputs'][0]
    stop_token_predictions = input_dict['decoder_output']['outputs'][1]
    mel_target = input_dict['target_tensors'][0]
    stop_token_target = input_dict['target_tensors'][1]
    spec_lengths = input_dict['target_tensors'][2]
    text_lengths = input_dict['decoder_output']['outputs'][3]
    stop_token_target = tf.expand_dims(stop_token_target, -1)
    alignments = input_dict['decoder_output']['outputs'][7]

    post_net_predictions = tf.cast(post_net_predictions, dtype=tf.float32)
    stop_token_predictions = tf.cast(stop_token_predictions, dtype=tf.float32)
    mel_target = tf.cast(mel_target, dtype=tf.float32)
    stop_token_predictions = tf.cast(stop_token_predictions, dtype=tf.float32)

    if self._both:
      mag_pred = input_dict['decoder_output']['outputs'][6]
      mag_pred = tf.cast(mag_pred, dtype=tf.float32)


    # Add zero padding to the end in the None dimension to get matching time length
    batch_size = tf.shape(mel_target)[0]
    num_feats = tf.shape(mel_target)[2]

    max_length = tf.to_int32(
        tf.maximum(
            tf.shape(mel_target)[1],
            tf.shape(post_net_predictions)[1],
        )
    )

    stop_token_pad = tf.zeros([batch_size, max_length - tf.shape(mel_target)[1], 1])
    mel_target_pad = tf.zeros([batch_size, max_length - tf.shape(mel_target)[1], num_feats])
    stop_token_pred_pad = tf.zeros(
        [batch_size, max_length - tf.shape(post_net_predictions)[1], 1]
    )
    post_net_pad = tf.zeros(
        [
            batch_size,
            max_length - tf.shape(post_net_predictions)[1],
            tf.shape(post_net_predictions)[2]
        ]
    )

    post_net_predictions = tf.concat(
        [post_net_predictions, post_net_pad], axis=1
    )
    stop_token_predictions = tf.concat(
        [stop_token_predictions, stop_token_pred_pad], axis=1
    )
    mel_target = tf.concat([mel_target, mel_target_pad], axis=1)
    stop_token_target = tf.concat([stop_token_target, stop_token_pad], axis=1)

    if self._both:
      mag_pad = tf.zeros(
          [
              batch_size,
              max_length - tf.shape(mag_pred)[1],
              tf.shape(mag_pred)[2]
          ]
      )
      mag_pred = tf.concat(
         [mag_pred, mag_pad], axis=1
      )

      if self._reduction_factor != 1:
        mag_target = input_dict['target_tensors'][3]
      else:
        mel_target, mag_target = tf.split(
            mel_target,
            [self._n_feats['mel'], self._n_feats['magnitude']],
            axis=2
        )

    mask = tf.sequence_mask(lengths=spec_lengths, maxlen=max_length, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)

    # Apply L1 Loss for spectrogram prediction and cross entropy for stop token
    if self.params['l1_loss']:
      decoder_loss = tf.reduce_mean(tf.losses.absolute_difference(
          labels=mel_target, 
          predictions=post_net_predictions, 
          weights=mask, 
          reduction=tf.losses.Reduction.NONE
      ))
      if self._both:
        mag_loss = tf.reduce_mean(tf.losses.absolute_difference(
          labels=mag_target, 
          predictions=mag_pred, 
          weights=mask, 
          reduction=tf.losses.Reduction.NONE
        ))
    else:
      decoder_loss = tf.losses.mean_squared_error(
        labels=mel_target, predictions=post_net_predictions, weights=mask
      )
      if self._both:
        mag_loss = tf.losses.mean_squared_error(
            labels=mag_target, predictions=mag_pred, weights=mask
        )

    stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=stop_token_target, logits=stop_token_predictions
    )

    stop_token_loss *= stop_token_loss
    stop_token_loss = tf.reduce_sum(stop_token_loss) / tf.reduce_sum(mask)
    
    # attn loss

    guided_attn_matrix = guided_attention(text_lengths, spec_lengths, 16)
    attn_loss = tf.reduce_mean(alignments * guided_attn_matrix)

    loss = decoder_loss + stop_token_loss + attn_loss

    if self._both:
      loss += mag_loss

    return loss
