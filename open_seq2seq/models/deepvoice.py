# Copyright (c) 2018 NVIDIA Corporation
import numpy as np
import tensorflow as tf
import os
import matplotlib as mpl
mpl.use('Agg')
from six import BytesIO
from .encoder_decoder import EncoderDecoderModel
from matplotlib import pyplot as plt
from .text2speech import plot_spectrograms

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
            'save_to_tensorboard': bool
        }
    )

  def __init__(self, params, mode="train", hvd=None):
    super(DeepVoice, self).__init__(params, mode=mode, hvd=hvd)

  def maybe_print_logs(self, input_values, output_values, training_step):
    predicted_mel_output = output_values[0]
    stop_prediction = output_values[1]
    target_mel_output, stop_target, _ = input_values['target_tensors']

    predicted_mel_sample = predicted_mel_output[0]
    stop_prediction_sample = stop_prediction[0]
    target_mel_sample = target_mel_output[0]
    stop_target_sample = stop_target[0]

    key_len_sample = output_values[3][1]

    # print(key_len_sample)

    alignment_list = output_values[2]
    dict_to_log = {}

    pe = output_values[3]

    # print(alignment_list[0])

    specs = [
      target_mel_sample,
      predicted_mel_sample,
      alignment_list[0],
      alignment_list[1],
      alignment_list[2],
      alignment_list[3]
    ]

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
    # # Plot alignments
    # im_summary = plot_alignment(alignment_list, training_step, self.params['logdir'], self.params['save_to_tensorboard'])
    # dict_to_log['image'] = im_summary

    # Plot spectrograms
    im_spec_summary = plot_spectrograms(
        specs,
        titles,
        stop_prediction_sample,
        0,
        self.params['logdir'],
        training_step
    )


    if self.params['save_to_tensorboard']:
      save_format = "tensorboard"
    
    return dict_to_log

  def evaluate(self, input_values, output_values):
    return output_values

  def finalize_evaluation(self, results_per_batch, training_step=None):
    return {}


  def finalize_inference(self, results_per_batch, output_file):
    return {}