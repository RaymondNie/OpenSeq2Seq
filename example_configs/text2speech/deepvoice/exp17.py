import os
from open_seq2seq.encoders import DeepVoiceEncoder
from open_seq2seq.decoders import DeepVoiceDecoder
from open_seq2seq.models import DeepVoice
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.losses import DeepVoiceLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr, transformer_policy, exp_decay

# Tests 15+ are with extra padded 0 to the left

base_model = DeepVoice
dataset = "LJ"
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
reduction_factor = 4
keep_prob = 0.9
base_params = {
  "use_horovod": True,
  "logdir": "/results/deepvoice3_fp32",
  "save_summaries_steps": 500,
  "print_loss_steps": 500,
  "print_samples_steps": 500,
  "save_checkpoint_steps": 500,
  "save_to_tensorboard": False,
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 1e-6
  },
  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "decay_steps": 10000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 20000,
    "min_lr": 1e-5,
  },
  "summaries": ["learning_rate", "gradients", "gradient_norm", "global_gradient_norm"],  
  "batch_size_per_gpu": 64,
  "max_steps": 200000,
  "dtype": tf.float32,
  "max_grad_norm":1.,
  "reduction_factor": reduction_factor,
  "num_audio_features": num_audio_features,
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
    "feature_normalize": False,
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
      "channels": 64,
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
      "reduction_factor": reduction_factor
  },
  # Loss params
  "loss": DeepVoiceLoss
}