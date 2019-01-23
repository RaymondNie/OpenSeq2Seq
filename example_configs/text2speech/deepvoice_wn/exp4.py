import os
from open_seq2seq.encoders import DeepVoiceEncoder
from open_seq2seq.decoders import DeepVoiceDecoder
from open_seq2seq.models import DeepVoice
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.losses import DeepVoiceLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr, transformer_policy, exp_decay

base_model = DeepVoice
dataset = "LJ"
dataset_location = "/home/rnie/Desktop/rnie/dataset/LJSpeech_mixed"
output_type = "mel"

if dataset == "MAILABS":
  trim = True
  mag_num_feats = 401
  train = "train.csv"
  val = "val.csv"
  batch_size = 32
elif dataset == "LJ":
  trim = False
  mag_num_feats = 513
  train = "train_32.csv"
  val = "val_32.csv"
  batch_size = 48
else:
  raise ValueError("Unknown dataset")

exp_mag = False
if output_type == "magnitude":
  num_audio_features = mag_num_feats
  data_min = 1e-5
elif output_type == "mel":
  num_audio_features = 80
  data_min = 1e-2
elif output_type == "both":
  num_audio_features = {
      "mel": 80,
      "magnitude": mag_num_feats
  }
  data_min = {
      "mel": 1e-2,
      "magnitude": 1e-5,
  }
  exp_mag = False
else:
  raise ValueError("Unknown param for output_type")

'''
batch_size == B
emb_size == e
encoder_channels == c
reduction_factor == r
'''
reduction_factor = 1
keep_prob = 0.95
base_params = {
  "use_horovod": True,
  # "num_gpus": 1,
  # "logdir": "/results/deepvoice3_fp32",
  "logdir": "/home/rnie/Desktop/rnie/OpenSeq2Seq/l2_wn_noam",
  "save_summaries_steps": 500,
  "print_loss_steps": 250,
  "print_samples_steps": 250,
  "save_checkpoint_steps": 500,
  "save_to_tensorboard": True,
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 1e-6
  },
  "optimizer": tf.contrib.opt.LazyAdamOptimizer,
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.997,
    "epsilon": 1e-09,
  },

  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "learning_rate": 2.0,
    "warmup_steps": 8000,
    "d_model": 256,
  },
  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  "batch_size_per_gpu": 64,
  "max_steps": 200000,
  "dtype": tf.float32,
  "max_grad_norm":1.,
  "reduction_factor": reduction_factor,
  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, "test.csv"),
    ],
    "dataset": dataset,
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "open_seq2seq/test_utils/vocab_tts.txt",
    'dataset_location':dataset_location,
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": True,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
    "data_min":data_min,
    "mel_type":'htk',
    "trim": trim,   
    "duration_max":1024,
    "duration_min":24,
    "exp_mag": exp_mag,
    "reduction_factor": reduction_factor,
    "mixed_phoneme_char_prob": 0.,
    "deepvoice": True
  },
  # Encoder params
  "encoder": DeepVoiceEncoder,
  "encoder_params": {
      "speaker_emb": None,
      "emb_size": 256,
      "channels": 256,
      "conv_layers": 7,
      "keep_prob": keep_prob, 
      "kernel_size": 5
  },
  # Decoder params
  "decoder": DeepVoiceDecoder,
  "decoder_params": {
      "speaker_emb": None,
      "emb_size": 256,
      "attention_size": 128,
      "prenet_layers": [128, 256],
      "channels": 256,
      "decoder_layers": 4,
      "kernel_size": 5,
      "keep_prob": keep_prob,
      "pos_rate": 5.54
  },
  # Loss params
  "loss": DeepVoiceLoss,
  "loss_params": {
    "l1_loss": False
  }
}