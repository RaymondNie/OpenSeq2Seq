import os
from open_seq2seq.encoders import DeepVoiceEncoder
from open_seq2seq.decoders import DeepVoiceDecoder
from open_seq2seq.models import DeepVoice
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.losses import DeepVoiceLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr, transformer_policy, exp_decay

# WN FULLY CONNECTED WITHOUT REDUCTION FACTOR

base_model = DeepVoice
dataset = "LJ"
# dataset_location = "/home/rnie/Desktop/rnie/dataset/LJSpeechPart"
dataset_location = "/data/LJSpeech"
output_type = "mel"
data_min = 1e-2
trim = False
exp_mag = False
num_audio_features = 80
'''
batch_size == B
emb_size == e
encoder_channels == c
reduction_factor == r
'''
reduction_factor = None

base_params = {
  "use_horovod": True,
  # "num_gpus": 1,
  "logdir": "/results/deepvoice3_fp32",
  # "logdir": "/home/rnie/Desktop/rnie/OpenSeq2Seq/deepvoice3_fp32_8",
  "print_loss_steps": 100,
  "print_samples_steps": 100,
  "save_checkpoint_steps": 500,
  "save_to_tensorboard": False,
  # "optimizer": tf.contrib.opt.LazyAdamOptimizer,
  # "optimizer_params": {
  #   "beta1": 0.9,
  #   "beta2": 0.997,
  #   "epsilon": 1e-09,
  # },

  # "lr_policy": transformer_policy,
  # "lr_policy_params": {
  #   "learning_rate": 2.0,
  #   "warmup_steps": 8000,
  #   "d_model": 256,
  # },
  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": fixed_lr,
  "lr_policy_params":{
    "learning_rate": 1e-4
  },
  "summaries": ['learning_rate'],  
  "batch_size_per_gpu": 16,
  "max_steps": 200000,
  "dtype": tf.float32,
  # Data Layer params
  # "data_layer": DeepVoiceDataLayer,
  # "data_layer_params": {
  #   'vocab_file':'/home/rnie/Desktop/rnie/OpenSeq2Seq/open_seq2seq/test_utils/vocab_tts.txt',
  #   'mixed_pronounciation_p':0.2,
  #   'num_audio_features':80,
  #   'dataset_files': ['/home/rnie/Desktop/rnie/dataset/LJSpeech-1.1/metadata_processed.csv'],
  #   'mode':'train',
  #   'shuffle':True,
  #   'dtype':tf.float32,
  #   'batch_size':2,
  #   'dataset_location':'/home/rnie/Desktop/rnie/dataset/LJSpeech-1.1/',
  #   'feature_normalize':True
  # },
  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, "train.csv"),
    ],
    "dataset": dataset,
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "open_seq2seq/test_utils/vocab_tts.txt",
    'dataset_location':dataset_location,
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": False,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
    "data_min":data_min,
    "mel_type":'htk',
    "trim": trim,   
    "duration_max":1024,
    "duration_min":24,
    "exp_mag": exp_mag,
    "reduction_factor": reduction_factor
  },
  # Encoder params
  "encoder": DeepVoiceEncoder,
  "encoder_params": {
      "speaker_emb": None,
      "emb_size": 256,
      "channels": 64,
      "conv_layers": 7,
      "keep_prob": 0.9, 
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
      "keep_prob": 0.9,
      "reduction_factor": reduction_factor
  },
  # Loss params
  "loss": DeepVoiceLoss
}