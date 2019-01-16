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
        "--logdir=exp17_50dropout/logs/",
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
num_audio_features = config_params.get('num_audio_features', 80)
Ty = 350 # max length of sound
decoder_layers = config_params['decoder_params']['decoder_layers']
pad_to = config_params['data_layer_params'].get('pad_to', 8)
reduction_factor = config_params.get('reduction_factor', 1)

# Starting values
text = "test one two three."
text_length = len(text)
text_length += pad_to - ((text_length + 2) % pad_to) + 2
spec = np.zeros((1, Ty // reduction_factor, num_audio_features * reduction_factor), np.float32)
spec_lens = np.array([[Ty]], np.int32)
prev_max_attentions_li = np.zeros((decoder_layers, 1), np.int32)
alignments_li = np.zeros((decoder_layers, Ty, text_length), np.float32)
stop_predictions = []
stop_timestep = Ty // reduction_factor
stop_predicted = False
# Teacher enforced
# test_spec = np.load("sample_spec.npy")
# test_spec = np.expand_dims(test_spec, 0)

for j in range(Ty // reduction_factor):
  _mel_output, _max_attentions_li, _alignments_li, _stop_predictions = get_interactive_infer_results(
      model_T2S, sess,
      model_in=(text, text_length, spec, spec_lens, prev_max_attentions_li)
  )

  prev_max_attentions_li = np.array(_max_attentions_li)[:,:,j]
  spec[:,j,:] = _mel_output[:,j,:]
  alignments_li[:,j,:] = np.array(_alignments_li)[:,j,:]
  stop_predictions.append(_stop_predictions[:,j,:][0])

  if stop_predictions[j] > 0.9 and stop_predicted == False:
    stop_timestep = j + 5
    stop_predicted = True

# Plot alignments and save audio
spec = spec[0]
if reduction_factor != 1:
  spec = np.reshape(spec, (-1, num_audio_features))

spec = spec[:stop_timestep, :]

spec = model_T2S.get_data_layer().get_magnitude_spec(spec, is_mel=True)
wav_summary = save_audio(
    spec,
    ".",
    1,
    n_fft=n_fft,
    sampling_rate=sampling_rate,
    save_format="disk"
)

plots = [
  spec,
  alignments_li[0],
  alignments_li[1],
  alignments_li[2],
  alignments_li[3]
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
