# Copyright (c) 2018 NVIDIA Corporation
import tensorflow as tf
from .loss import Loss

def guided_attention(text_lens, spec_lens, batch_size, g=0.2):
    max_text_len = tf.to_float(tf.reduce_max(text_lens))
    max_spec_len = tf.to_float(tf.reduce_max(spec_lens))
    text_lens = tf.to_float(text_lens)
    spec_lens = tf.to_float(spec_lens)
    # Create a T x S where each column S is a vector 1 ... T representing row indices 
    rows = tf.reshape(tf.tile(tf.range(max_text_len), [max_spec_len]), [max_spec_len, -1]) # Ty x Tx
    rows = tf.reshape(tf.tile(rows, [batch_size, 1]), [batch_size, max_spec_len, -1]) / tf.reshape(text_lens, [batch_size, 1, 1]) # B x Ty x Tx
    # Create a T x S where each row T is a vector 1 ... S representing col indices 
    cols = tf.transpose(tf.reshape(tf.tile(tf.range(max_spec_len), [max_text_len]), [max_text_len, -1]), [1,0])
    cols = tf.reshape(tf.tile(cols, [batch_size, 1]), [batch_size, max_spec_len, -1]) / tf.reshape(spec_lens, [batch_size, 1, 1]) # B x Ty x Tx
    # Apply guided attention formula
    matrix = 1 - tf.exp(-(rows-cols)**2 / (2 * g**2))
    return matrix

class DeepVoiceLoss(Loss):
  def __init__(self, params, model, name="deepvoice_loss"):
    super(DeepVoiceLoss, self).__init__(params, model, name)
    self.batch_size = self._model.get_data_layer().params['batch_size']
    self._n_feats = self._model.get_data_layer().params["num_audio_features"]
    if "both" in self._model.get_data_layer().params['output_type']:
      self._both = True
    else:
      self._both = False
    if model.get_data_layer().params.get('reduction_factor', None) != None:
      self.reduction_factor = model.get_data_layer().params['reduction_factor']
    else:
      self.reduction_factor = 1
  def get_required_params(self):
    return dict(
        Loss.get_required_params(), **{
            'l1_loss': bool,
            'masked_loss_weight': float
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
    alignments = input_dict['decoder_output']['outputs'][2]
    mask_w = self.params['masked_loss_weight']
    stop_token_target = tf.expand_dims(stop_token_target, -1)
    batch_size = tf.shape(mel_target)[0]
    max_length = tf.to_int32(tf.shape(mel_target)[1])
    num_mels = tf.shape(post_net_predictions)[2]

    final_output_mask = tf.sequence_mask(spec_lengths, max_length)

    if self._both:
      mag_pred = input_dict['decoder_output']['outputs'][6]
      mag_target = input_dict['target_tensors'][3]

    mask = tf.sequence_mask(lengths=spec_lengths, maxlen=max_length, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)

    # Apply L1 Loss for spectrogram prediction and cross entropy for stop token
    if self.params['l1_loss']:
      masked_decoder_loss = tf.losses.absolute_difference(
          labels=mel_target, 
          predictions=post_net_predictions, 
          weights=mask
      )
      decoder_loss = tf.losses.absolute_difference(
          labels=mel_target, 
          predictions=post_net_predictions,
      )
      if self._both:
        masked_mag_loss = tf.losses.absolute_difference(
          labels=mag_target, 
          predictions=mag_pred, 
          weights=mask
        )
        mag_loss = tf.losses.absolute_difference(
          labels=mag_target, 
          predictions=mag_pred, 
        )
    else:
      masked_decoder_loss = tf.losses.mean_squared_error(
          labels=mel_target, 
          predictions=post_net_predictions, 
          weights=mask
      )
      decoder_loss = tf.losses.mean_squared_error(
          labels=mel_target, 
          predictions=post_net_predictions
      )
      if self._both:
        masked_mag_loss = tf.losses.mean_squared_error(
            labels=mag_target, 
            predictions=mag_pred, 
            weights=mask
        )
        mag_loss = tf.losses.mean_squared_error(
            labels=mag_target,
            predictions=mag_pred
        )
    total_decoder_loss = mask_w * masked_decoder_loss + (1-mask_w) * decoder_loss
    total_mag_loss = mask_w * masked_mag_loss + (1-mask_w) * mag_loss

    stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=stop_token_target, logits=stop_token_predictions
    )

    stop_token_loss *= stop_token_loss
    stop_token_loss = tf.reduce_sum(stop_token_loss) / tf.reduce_sum(mask)
    
    # Guided attention loss
    guided_attn_matrix = guided_attention(text_lengths, spec_lengths, self.batch_size)
    # stack alignments into one tensor
    all_alignments = tf.stack(alignments)
    attn_loss = tf.reduce_sum(all_alignments * tf.expand_dims(guided_attn_matrix, 0), [2,3]) / tf.expand_dims(tf.to_float((text_lengths * spec_lengths)), 0)
    attn_loss = tf.reduce_mean(attn_loss)

    loss = total_decoder_loss + stop_token_loss + attn_loss
    if self._both:
      loss += total_mag_loss

    # Log different losses
    tf.summary.scalar(name="stop_token_loss", tensor=stop_token_loss)
    tf.summary.scalar(name="attn_loss", tensor=attn_loss)
    tf.summary.scalar(name="mag_loss", tensor=mag_loss)
    tf.summary.scalar(name="mel_loss", tensor=decoder_loss)
    tf.summary.scalar(name="masked_mag", tensor=masked_mag_loss)
    tf.summary.scalar(name="masked_decoder", tensor=masked_decoder_loss)

    return loss
