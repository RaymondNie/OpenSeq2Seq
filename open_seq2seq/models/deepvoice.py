# Copyright (c) 2018 NVIDIA Corporation
import numpy as np
import tensorflow as tf
import os

from .encoder_decoder import EncoderDecoderModel
from matplotlib import pyplot as plt

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
    plt.savefig('{}/alignment_{}.png'.format(directory, gs), format='png')

class DeepVoice(EncoderDecoderModel):
  @staticmethod
  def get_required_params():
    return dict(
        EncoderDecoderModel.get_required_params(), **{}
    )

  def __init__(self, params, mode="train", hvd=None):
    super(DeepVoice, self).__init__(params, mode=mode, hvd=hvd)

  def maybe_print_logs(self, input_values, output_values, training_step):
    alignment_list = output_values[1]
    plot_alignment(alignment_list, training_step, self.params['logdir'])
    return {}

  def evaluate(self, input_values, output_values):
    return output_values

  def finalize_evaluation(self, results_per_batch, training_step=None):
    return {}


  def finalize_inference(self, results_per_batch, output_file):
    return {}