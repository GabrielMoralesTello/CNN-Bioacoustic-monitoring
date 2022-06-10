import json
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np

def divide_frames(im, w, s): 
    for i in range(0, im.shape[1], s):  
        yield im[:, i:i + w]
    
def fig2data (fig):
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    return buf

# CNN model
model_path = '../model/ResNet50_test'

# Recording directory
recording_dir = '/media/gabsoni/GABOWET/Angachilla/' # '../test_recordings_wav/'
test_recordings = [f for f in sorted(os.listdir(recording_dir)) if f.startswith('MON') and f.endswith('.wav')]

# Load CNN model

model = model_from_json(open(model_path+'.json', 'r').read())
model.load_weights(model_path+'.h5')
class_dict = json.load(open(model_path+'_classes.json', 'r'))
class_dict_rev = {(str(v[0])): k for k, v in class_dict.items()}

print('Loaded model ')

model_input_shape = model.get_layer(index=0).input_shape[1:]
n_classes = model.get_layer(index=-1).output_shape[1:][0]

# CNN input sample rate
model_sample_rate = 44100

pixLen = 172 # 172 spectrogram pixels is ~2 seconds
shft = 86 # %50 overlap between 172-length windows 

for n, j in enumerate(test_recordings): # loop over recordings
    if not os.path.exists('../report/predictions/' + j.split('.')[0] + '.csv'):
        try:
            audio_data, sampling_rate = librosa.load(recording_dir+j, sr=model_sample_rate)
            pxx = librosa.feature.melspectrogram(y = audio_data, 
                                sr = sampling_rate,
                                n_fft=2048,
                                n_mels=128,
                                hop_length=512, 
                                win_length=1024)
            del audio_data
            pxx = librosa.power_to_db(pxx, ref=np.max)
            X = []
            for c, jj in enumerate(divide_frames(pxx, pixLen, shft)): # loop over frames
                if jj.shape[1] != pixLen:
                    continue
                dpi=100
                fig = plt.figure(num=None, figsize=(224/dpi, 224/dpi), dpi=dpi)
                ax = plt.axes()
                ax.set_axis_off()
                librosa.display.specshow(jj)
                img = fig2data(fig)
                del fig
                plt.close()
                X.append(img/255.0)
            X = np.stack(X)
            p = model.predict(X)
            del X
            p = pd.DataFrame(p)
            p.to_csv('../report/predictions/' + j.split('.')[0] + '.csv', index=False)
            print('Predicted', j)
        except:
            print('OVERLOAD')
            continue