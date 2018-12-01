# Copyright (c) 2018 NVIDIA Corporation
import numpy as np
from .encoder_decoder import EncoderDecoderModel

def plot_alignments():
  pass

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
    
    return {}

  def evaluate(self, input_values, output_values):
    return output_values

  def finalize_evaluation(self, results_per_batch, training_step=None):
    return {}


  def finalize_inference(self, results_per_batch, output_file):
    return {}