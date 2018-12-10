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
    mel_logits = input_dict['decoder_output']['outputs'][0]
    mel_output = tf.nn.sigmoid(mel_logits)

    stop_token_logits = input_dict['decoder_output']['stop_token_prediction'][0]
    stop_token_outputs = tf.nn.sigmoid(stop_token_logits)

    mel_target = input_dict['target_tensors'][0]
    stop_token_target = input_dict['target_tensors'][1]
    stop_token_target = tf.expand_dims(stop_token_target, -1)

    # decoder_loss = tf.losses.mean_squared_error(
    #     labels=mel_target, predictions=mel_output
    # )

    decoder_loss = tf.reduce_mean(tf.abs(mel_target - mel_output))

    stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=stop_token_target, logits=stop_token_outputs
    )

    loss = decoder_loss + stop_token_loss

    return loss