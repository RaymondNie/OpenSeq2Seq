# Copyright (c) 2018 NVIDIA Corporation
import tensorflow as tf
from .loss import Loss
class DeepVoiceLoss(Loss):

  def __init__(self, params, model, name="deepvoice_loss"):
    super(DeepVoiceLoss, self).__init__(params, model, name)
    self._n_feats = self._model.get_data_layer().params["num_audio_features"]

  def get_required_params(self):
    return {}

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

    stop_token_target = tf.expand_dims(stop_token_target, -1)

    post_net_predictions = tf.cast(post_net_predictions, dtype=tf.float32)
    stop_token_predictions = tf.cast(stop_token_predictions, dtype=tf.float32)
    mel_target = tf.cast(mel_target, dtype=tf.float32)
    stop_token_predictions = tf.cast(stop_token_predictions, dtype=tf.float32)

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

    mask = tf.sequence_mask(lengths=spec_lengths, maxlen=max_length, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)

    # Apply L1 Loss for spectrogram prediction and cross entropy for stop token
    # post_net_predictions *= mask
    # decoder_loss = tf.reduce_mean(tf.abs(mel_target - post_net_predictions))
    decoder_loss = tf.losses.mean_squared_error(
      labels=mel_target, predictions=post_net_predictions, weights=mask
    )

    stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=stop_token_target, logits=stop_token_predictions
    )

    stop_token_loss *= stop_token_loss
    stop_token_loss = tf.reduce_sum(stop_token_loss) / tf.reduce_sum(mask)
      
    loss = decoder_loss + stop_token_loss

    return loss