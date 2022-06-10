from tensorflow.keras.models import model_from_json
import json
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import librosa
import librosa.display
import pandas as pd
import tensorflow as tf
import time

# CNN model
model_path = '../model/ResNet50_test'

# Spectrograms directory
spectrograms_dir_p = '../dataset_test/STFT/presences/'
spectrograms_dir_a = '../dataset_test/STFT/absences/'

batch_size = 32

# Load CNN model
model = model_from_json(open(model_path+'.json', 'r').read())
model.load_weights(model_path+'.h5')
class_dict = json.load(open(model_path+'_classes.json', 'r'))
class_dict_rev = {(str(v[0])): k for k, v in class_dict.items()}

print(model_path)
print('Loaded model ')

model_input_shape = model.get_layer(index=0).input_shape[1:]
n_classes = model.get_layer(index=-1).output_shape[1:][0]

files = []
target = []
target_size_p = []
target_size_a = []
columns_df = ['filename']

for c, i in enumerate(sorted(os.listdir(spectrograms_dir_p))):
    columns_df.append(i)
    for cc, j in enumerate(sorted(os.listdir(spectrograms_dir_p+i))):
        files.append(spectrograms_dir_p+i+'/'+j)
        tmp = np.empty(n_classes)
        tmp[:] = np.nan
        tmp[c] = int(1)
        target.append(tmp)
    target_size_p.append(cc+1)
        
for c, i in enumerate(sorted(os.listdir(spectrograms_dir_a))):
    for cc, j in enumerate(sorted(os.listdir(spectrograms_dir_a+i))):
        files.append(spectrograms_dir_a+i+'/'+j)
        tmp = np.empty(n_classes)
        tmp[:] = np.nan
        tmp[c] = int(0)
        target.append(tmp)
    target_size_a.append(cc+1)
        
df_test = pd.concat([pd.DataFrame({'filename':files}),pd.DataFrame(np.asarray(target))],axis=1)
df_test.columns = columns_df

print('Presences by class: ', target_size_p)
print('Total presences: ', sum(target_size_p))
print('Absences by class: ', target_size_a)
print('Total absences: ', sum(target_size_a))
Total_test_set = sum(target_size_p)+sum(target_size_a)
print('Total test-set: ', Total_test_set)

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D np array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a np 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( w, h, 3 )
    
    return buf

def to_predict(col_class, presence):
    if presence == 1:
        info = ' presence'
    elif presence == 0:
        info = ' absence'
    else:
        print('enter 1 for presence and 0 for absence')
    print('... Loading spectrograms with ' + class_dict[str(col_class)] + info + '...' )
    index = df_test[df_test.iloc[:,col_class+1] == presence].index
    X = [] 
    for c, jj in enumerate(df_test.filename[index[0]:index[-1]+1]): # loop over spectrograms with presences
        #with tf.device('/device:GPU:0'):
        #    img_tmp = tf.Variable(tf.zeros([224,224,3]))
        dpi=100
        fig = plt.figure(num=None, figsize=(224/dpi, 224/dpi), dpi=dpi)
        ax = plt.axes()
        ax.set_axis_off()
        spectrogram = mpimg.imread(jj, format('png'))
        #spectrogram_edit = spectrogram[7:231,0:224,0:-1]
        ax.imshow(spectrogram)
        plt.tight_layout()
        plt.close()
        img = fig2data(fig)
        #img_tmp.assign_add(img/255.0)
        X.append(img/255.0) 
    X = np.stack(X)
    print('... Generating prediction matrix')
    p = model.predict(X, batch_size = batch_size)
    print('... Prediction finished')
    return(p)

P = []
for i in range(len(class_dict)):
    prediction_on_presences = to_predict(col_class=i, presence=1)
    P.append(prediction_on_presences)
for j in range(len(class_dict)):
    prediction_on_absences = to_predict(col_class=j, presence=0)
    P.append(prediction_on_absences)

x_true = [] #GroundTruth by class (presences and absences)
x_predict = [] #Prediction by class (presences and absences)
for i in range(n_classes): 
    x_p = df_test[df_test.loc[:,class_dict[str(i)]] == 1]
    x_a = df_test[df_test.loc[:,class_dict[str(i)]] == 0]
    x_true.append(np.concatenate((np.array(x_p[class_dict[str(i)]]), np.array(x_a[class_dict[str(i)]])), axis=0))
    x_predict.append(np.concatenate((P[i][:,i], P[i+n_classes][:,i]), axis=0))

#Etiquetas para el plot:
labels_to_plot = []
for i in class_dict:
    labels_to_plot.append(class_dict[str(i)].split(' ')[0][0]+'. '+class_dict[str(i)].split(' ')[1])


#Store: x_true, x_predict, labels_to_plot

x_true_store = open('x_true.pkl', 'wb') #write binary
pickle.dump(x_true, x_true_store)
x_true_store.close()

x_predict_store = open('x_predict.pkl', 'wb') #write binary
pickle.dump(x_predict, x_predict_store)
x_predict_store.close()

labels_to_plot_store = open('labels_to_plot.pkl', 'wb') #write binary
pickle.dump(labels_to_plot, labels_to_plot_store)
labels_to_plot_store.close()

print('Predictions saved')