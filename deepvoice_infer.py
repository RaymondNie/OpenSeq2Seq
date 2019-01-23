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
        "--logdir=exp17-50dropout/deepvoice_fp32/logs/",
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
decoder_layers = config_params['decoder_params']['decoder_layers']
pad_to = config_params['data_layer_params'].get('pad_to', 8)
reduction_factor = config_params.get('reduction_factor', 1)
if "both" in model_T2S.get_data_layer().params.get('output_type', None):
  both = True
  num_audio_features = model_T2S.get_data_layer().params['num_audio_features']['mel']
  mag_audio_features = model_T2S.get_data_layer().params['num_audio_features']['magnitude']
else:
  num_audio_features = model_T2S.get_data_layer().params.get('num_audio_features', 80)
  mag_audio_features = 0
  both = False

# Starting values
# text = "test one two."
text = "testing random sentence made out of many words."
text_length = len(text)
text_length += pad_to - ((text_length + 2) % pad_to) + 2
input_spec = np.zeros((1, Ty, (num_audio_features + mag_audio_features) * reduction_factor), np.float32)
predicted_mel = np.zeros((1, Ty, num_audio_features * reduction_factor), np.float32)
predicted_mag = np.zeros((1, Ty, mag_audio_features * reduction_factor), np.float32)
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
  alignments_li[:,j,:] = np.array(_alignments_li)[:,j,:]
  stop_predictions.append(_stop_predictions[:,j,:][0])
  if both:
    predicted_mag[:,j,:] = _mag_output[:,j,:]
    input_spec = np.concatenate((predicted_mel, predicted_mag), axis=2)
  else:
    input_spec = predicted_mel

  # if stop_predictions[j] > 0.95 and stop_predicted == False:
  #   stop_timestep = j + 5
  #   stop_predicted = True

# Plot alignments and save audio
predicted_mel = predicted_mel[0]
predicted_mag = predicted_mag[0]

if reduction_factor != 1:
  predicted_mel = np.reshape(predicted_mel, (-1, num_audio_features))

predicted_mel = predicted_mel[:stop_timestep, :]


if both:
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

predicted_mel = model_T2S.get_data_layer().get_magnitude_spec(predicted_mel, is_mel=True)
wav_summary = save_audio(
    predicted_mel,
    ".",
    1,
    n_fft=n_fft,
    sampling_rate=sampling_rate,
    save_format="disk"
)

plots = [
  predicted_mel,
  alignments_li[0][:Ty//reduction_factor,:],
  alignments_li[1][:Ty//reduction_factor,:],
  alignments_li[2][:Ty//reduction_factor,:],
  alignments_li[3][:Ty//reduction_factor,:]
]

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
