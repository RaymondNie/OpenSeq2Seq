# Copyright (c) 2018 NVIDIA Corporation
import numpy as np
import tensorflow as tf
import os
import matplotlib as mpl
mpl.use('Agg')
from six import BytesIO
from .encoder_decoder import EncoderDecoderModel
from matplotlib import pyplot as plt
from .text2speech import plot_spectrograms, save_audio

def plot_alignment(alignments, gs, logdir, save_to_tensorboard=False):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    """
    directory = '/home/rnie/Desktop/rnie/OpenSeq2Seq/plots/{}'.format(logdir)
    if not os.path.exists(directory):
      os.makedirs(directory)

    fig, axes = plt.subplots(nrows=len(alignments), ncols=1, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        # i=0
        im = ax.imshow(alignments[i])
        ax.axis('off')
        ax.set_title("Layer {}".format(i))

    fig.subplots_adjust(right=0.8, hspace=0.4)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle('{} Steps'.format(gs))
    
    if save_to_tensorboard:
      tag = "{}_image".format(i)
      iostream = BytesIO()
      fig.savefig(iostream, dpi=300)
      summary = tf.Summary.Image(
          encoded_image_string=iostream.getvalue(),
          height=int(fig.get_figheight() * 300),
          width=int(fig.get_figwidth() * 300)
      )
      summary = tf.Summary.Value(tag=tag, image=summary)
      plt.close(fig)
      return summary
    else:
      plt.savefig('{}/alignment_{}.png'.format(directory, gs), format='png')
      plt.close(fig)
      return None

class DeepVoice(EncoderDecoderModel):
  @staticmethod
  def get_required_params():
    return dict(
        EncoderDecoderModel.get_required_params(), **{}
    )

  @staticmethod
  def get_optional_params():
    return dict(
        EncoderDecoderModel.get_optional_params(), **{
            'save_to_tensorboard': bool,
        }
    )

  def __init__(self, params, mode="train", hvd=None):
    super(DeepVoice, self).__init__(params, mode=mode, hvd=hvd)
    self.save_to_tensorboard = self.params['save_to_tensorboard']
    self.reduction_factor = self.params['data_layer_params']['reduction_factor']
    self.both = "both" in self.params['data_layer_params']['output_type']
    if self.both:
      self.mel_feats = self.params['data_layer_params']['num_audio_features']['mel']
      self.mag_feats = self.params['data_layer_params']['num_audio_features']['magnitude']
    else:
      self.mel_feats = self.params['data_layer_params']['num_audio_features']

  def maybe_print_logs(self, input_values, output_values, training_step):
    predicted_mel_output = output_values[0]
    stop_prediction = output_values[1]
    target_mel_output = input_values['target_tensors'][0]
    stop_target = input_values['target_tensors'][1]

    predicted_mel_sample = predicted_mel_output[0]
    stop_prediction_sample = stop_prediction[0]
    target_mel_sample = target_mel_output[0]
    stop_target_sample = stop_target[0]
    spec_len = output_values[4][0]

    key_len_sample = output_values[3][1]


    alignment_list = output_values[2]
    dict_to_log = {}

    pe = output_values[3]

    specs = [
      target_mel_sample,
      predicted_mel_sample
    ]

    for alignment_layer_plot in alignment_list:
      specs.append(alignment_layer_plot)

    titles = [
        "target output",
        "predicted output",
        "alignment_1",
        "alignment_2",
        "alignment_3",
        "alignment_4"
    ]

    directory = self.params['logdir']
    if not os.path.exists(directory):
      os.makedirs(directory)

    save_format = "tensorboard" if self.save_to_tensorboard == True else "disk"

    # Plot spectrograms
    im_spec_summary = plot_spectrograms(
        specs,
        titles,
        stop_prediction_sample,
        0,
        self.params['logdir'],
        training_step,
        save_to_tensorboard=self.save_to_tensorboard,
        append="train"
    )
    dict_to_log['image'] = im_spec_summary

    # Save audio for mel spectrogram
    if self.reduction_factor != 1:
      predicted_mel_sample = np.reshape(predicted_mel_sample, (-1, self.mel_feats))

    predicted_mel_sample = predicted_mel_sample[:spec_len * self.reduction_factor - 1, :]
    predicted_mel_sample = self.get_data_layer().get_magnitude_spec(predicted_mel_sample, is_mel=True)

    wav_summary = save_audio(
        predicted_mel_sample,
        self.params["logdir"],
        training_step,
        n_fft=self.get_data_layer().n_fft,
        sampling_rate=self.get_data_layer().sampling_rate,
        save_format=save_format
    )
    dict_to_log['audio'] = wav_summary

    # Save audio for mag spectrogram
    if "both" in self.get_data_layer().params['output_type']:
      predicted_mag_spec = output_values[6][0]

      if self.reduction_factor != 1:
        predicted_mag_spec = np.reshape(predicted_mag_spec, (-1, self.mag_feats))

      predicted_mag_spec = predicted_mag_spec[:spec_len * self.reduction_factor - 1, :]
      predicted_mag_spec = self.get_data_layer().get_magnitude_spec(predicted_mag_spec)
      wav_summary = save_audio(
          predicted_mag_spec,
          self.params["logdir"],
          training_step,
          n_fft=self.get_data_layer().n_fft,
          sampling_rate=self.get_data_layer().sampling_rate,
          mode="mag",
          save_format=save_format,
      )

      dict_to_log['audio_mag'] = wav_summary

    return dict_to_log

  def evaluate(self, input_values, output_values):
    return output_values

  def finalize_evaluation(self, results_per_batch, training_step=None):
    return {}

  def infer(self, input_values, output_values):
    mel_output = output_values[0]
    max_attentions_list = output_values[5]
    alignment_list = output_values[2]
    stop_prediction = output_values[1]
    if self.both:
      predicted_mag_spec = output_values[6]
    else:
      predicted_mag_spec = np.zeros((5,5))

    return mel_output, max_attentions_list, alignment_list, stop_prediction, predicted_mag_spec

  def finalize_inference(self, results_per_batch, output_file):
    return {}