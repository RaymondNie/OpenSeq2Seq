# Replace the first box of Interactive_Infer_example.ipynb with this
import IPython
import librosa

import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf

from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
                                     create_logdir, create_model, get_interactive_infer_results
from open_seq2seq.models.text2speech import plot_spectrograms, save_audio

args = [
        "--config_file=example_configs/text2speech/deepvoice3_infer.py",
        "--mode=interactive_infer",
        "--logdir=bn_batch512/exp6/logs/",
]

# A simpler version of what run.py does. It returns the created model and its 
# saved checkpoint
def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(args, base_config, config_module, base_model, None)
    return base_config, model, checkpoint

config_params, model_T2S, checkpoint_T2S = get_model(args, "deepvoice")

# Create the session and load the checkpoints
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

vars_T2S = {}
for v in tf.get_collection(tf.GraphKeys.VARIABLES):
    if "deepvoice" in v.name:
        vars_T2S["/".join(v.op.name.split("/")[1:])] = v

saver_T2S = tf.train.Saver(vars_T2S)
saver_T2S.restore(sess, checkpoint_T2S)

# Inference params
n_fft = model_T2S.get_data_layer().n_fft
sampling_rate = model_T2S.get_data_layer().sampling_rate
Ty = 500 # max length of sound

reduction_factor = model_T2S.get_data_layer().params['reduction_factor']
both = "both" in model_T2S.get_data_layer().params['output_type']

if both:
  mel_feats = model_T2S.get_data_layer().params['num_audio_features']['mel']
  mag_feats =  model_T2S.get_data_layer().params['num_audio_features']['magnitude']
  predicted_mag = np.zeros((1, Ty, mag_feats * reduction_factor), np.float32)
  predicted_mel = np.zeros((1, Ty, mel_feats * reduction_factor), np.float32)
else:
  mel_feats = model_T2S.get_data_layer().params['num_audio_features']
  predicted_mel = np.zeros((1, Ty, mel_feats * reduction_factor), np.float32)


decoder_layers = config_params['decoder_params']['decoder_layers']
pad_to = model_T2S.get_data_layer().params.get('pad_to', 8)

# Initial text values with proper padding
text = "something went wrong"
text_length = len(text)
text_length += pad_to - ((text_length + 2) % pad_to) + 2

# Initial spectrogram value and arrays to hold predicted values
input_spec = np.zeros((1, Ty, mel_feats * reduction_factor), np.float32)
spec_lens = np.array([[Ty]], np.int32)
prev_max_attentions_li = np.zeros((decoder_layers, 1), np.int32)
alignments_li = np.zeros((decoder_layers, Ty, text_length), np.float32)
stop_predictions = []
stop_timestep = Ty
stop_predicted = False

# Teacher enforced
# test_spec = np.load("sample_spec.npy")
# test_spec = np.expand_dims(test_spec, 0)

for j in range(Ty // reduction_factor):
  _mel_output, _max_attentions_li, _alignments_li, _stop_predictions, _mag_output = get_interactive_infer_results(
      model_T2S, sess,
      model_in=(text, text_length, input_spec, spec_lens, prev_max_attentions_li)
  )

  prev_max_attentions_li = np.array(_max_attentions_li)[:,:,j]
  predicted_mel[:,j,:] = _mel_output[:,j,:]

  # Get alignments for first sample
  for layer in range(decoder_layers):
    alignments_li[layer,j,:] = np.array(_alignments_li[layer][0])[j,:]

  stop_predictions.append(_stop_predictions[:,j,:][0])

  if both:
    predicted_mag = _mag_output
  else:
    input_spec = predicted_mel

  # if stop_predictions[j] > 0.95 and stop_predicted == False:
  #   stop_timestep = j * reduction_factor
  #   stop_predicted = True
  #   break

# Plot alignments and save audio

if reduction_factor != 1:
  predicted_mel = predicted_mel[0]
  predicted_mel = np.reshape(predicted_mel, (-1, mel_feats))
  predicted_mel = predicted_mel[:stop_timestep, :]
  predicted_mel = model_T2S.get_data_layer().get_magnitude_spec(predicted_mel, is_mel=True)

  wav_summary = save_audio(
      predicted_mel,
      ".",
      1,
      n_fft=n_fft,
      sampling_rate=sampling_rate,
      save_format="disk"
  )

  if both:
    predicted_mag = predicted_mag[0]
    predicted_mag = np.reshape(predicted_mag, (-1, mag_feats))
    predicted_mag = predicted_mag[:stop_timestep, :]
    predicted_mag = model_T2S.get_data_layer().get_magnitude_spec(predicted_mag)

    wav_summary = save_audio(
        predicted_mag,
        ".",
        1,
        n_fft=n_fft,
        sampling_rate=sampling_rate,
        mode="mag",
        save_format="disk",
    )

plots = [
  predicted_mel,
]

for layer in range(decoder_layers):
  plots.append(alignments_li[layer][:Ty//reduction_factor,:])

titles = [
    "predicted output",
    "alignment_1",
    "alignment_2",
    "alignment_3",
    "alignment_4"
]

im_spec_summary = plot_spectrograms(
    plots,
    titles,
    np.array(stop_predictions),
    0,
    ".",
    1
)
