#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)


# # New Section

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


dfFall = pd.read_csv('/content/drive/MyDrive/fall.csv')
uploaded = open('/content/WISDM_ar_v1.1_raw.txt', "r")


# In[ ]:


dfFall= dfFall[dfFall['Sensor Type'] == 0]
dfFall


# In[ ]:


df = dfFall.drop(['Unnamed: 0', 'Sample No','Sensor Type','Sensor ID','Fall'], axis = 1)
df['user']=40
df['activity']='Fall'
df.columns=['time','x','y','z','user','activity']
df_new = df[['user', 'activity', 'time','x','y','z']]


# In[ ]:


df_new


# In[ ]:


# file = "WISDM_ar_v1.1_raw.txt"

# for f in uploaded.keys():
#     file = open(f, 'r')
lines = uploaded.readlines()
    


processedList = []

for i, line in enumerate(lines):
    try:
        line = line.split(',')
        last = line[5].split(';')[0]
        last = last.strip()
        if last == '':
            break;
        temp = [line[0], line[1], line[2], line[3], line[4], last]
        processedList.append(temp)
    except:
        print('Error at line number: ', i)


# In[ ]:


processedList[:10]


# In[ ]:


columns = ['user', 'activity', 'time', 'x', 'y', 'z']
data = pd.DataFrame(data = processedList, columns = columns)
data = pd.concat([data,df_new])
data


# In[ ]:


data= data[data['activity'] != 'Jogging']
data


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data['activity'].value_counts()


# In[ ]:


data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')


# In[ ]:


data.info()


# In[ ]:


Fs = 20#sampling rate in Hz
activities = data['activity'].value_counts().index
activities


# In[ ]:


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, data['time'], data['x'], 'X-Axis')
    plot_axis(ax1, data['time'], data['y'], 'Y-Axis')
    plot_axis(ax2, data['time'], data['z'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in activities:
    data_for_plot = data[(data['activity'] == activity)][:Fs*10]
    plot_activity(activity, data_for_plot)


# In[ ]:


df = data.drop(['user', 'time'], axis = 1).copy()
df.head()


# In[ ]:


df['activity'].value_counts()


# In[ ]:


Walking = df[df['activity']=='Walking'].head(1630).copy()
#Jogging = df[df['activity']=='Jogging'].head(1630).copy()
Upstairs = df[df['activity']=='Upstairs'].head(1630).copy()
Downstairs = df[df['activity']=='Downstairs'].head(1630).copy()
Sitting = df[df['activity']=='Sitting'].head(1630).copy()
Fall = df[df['activity']=='Fall'].head(1630).copy()
Standing = df[df['activity']=='Standing'].copy()

balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([Walking, Upstairs, Downstairs, Sitting,Fall, Standing])
balanced_data.shape


# In[ ]:


balanced_data['activity'].value_counts()


# In[ ]:


balanced_data.head()


# In[ ]:


label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['activity'])
balanced_data.head()


# In[ ]:


label.classes_


# In[ ]:


X = balanced_data[['x', 'y', 'z']]
y = balanced_data['label']


# In[ ]:


scaler = StandardScaler()
X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
scaled_X['label'] = y.values

scaled_X.head()


# In[ ]:


import scipy.stats as stats


# In[ ]:


Fs = 20
frame_size = Fs*4 
hop_size = Fs*2 


# In[ ]:


def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

X, y = get_frames(scaled_X, frame_size, hop_size)

X.shape, y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:


X_train[0].shape, X_test[0].shape


# In[ ]:


X_train = X_train.reshape(194, 80, 3, 1)
X_test = X_test.reshape(49, 80, 3, 1)


# In[ ]:


X_train[0].shape, X_test[0].shape


# In[ ]:


import sys
sys.path
sys.executable


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D  #this is to perform the convolution operation

from keras.layers import MaxPooling2D  #used for pooling operation

from keras.layers import Flatten  #used for Flattening. Flattening is the process of converting all the resultant 2 dimensional arrays into a single long continuous linear vector.
from keras.layers import Dropout
from keras.layers import Dense  #used to perform the full connection of the neural network
model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))


# In[ ]:


model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = 15, validation_data= (X_test, y_test), verbose=1)


# In[ ]:


model.summary()


# In[ ]:


def plot_learningCurve(history, epochs):
  # Plot training & validation accuracy values
  sns.set(font_scale=1)
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()


# In[ ]:


plot_learningCurve(history, 15)


# In[ ]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[ ]:


y_pred = np.argmax(model.predict(X_test), axis=-1)


# In[ ]:


import matplotlib.pyplot as plt

plt.imshow(conf, interpolation='nearest', cmap=plt.cm.Greens)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = conf.max() / 2.
for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
    plt.text(j, i, format(conf[i, j], fmt),
             horizontalalignment="center",
             color="white" if conf[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)

ax = plt.subplot()
sns.set(font_scale=1.5) # Adjust to fit
sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g");  

# Labels, title and ticks
label_font = {'size':'12'}  # Adjust to fit
ax.set_xlabel('Predicted labels', fontdict=label_font,fontweight='bold');
ax.set_ylabel('True labels', fontdict=label_font,fontweight='bold');

title_font = {'size':12}  # Adjust to fit
ax.set_title('Confusion Matrix', fontdict=title_font,fontweight='bold');

ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust to fit
ax.xaxis.set_ticklabels(['Walking', 'Upstairs', 'Downstairs', 'Sitting','Fall', 'Standing'],fontweight='bold',rotation='vertical');
ax.yaxis.set_ticklabels(['Walking', 'Upstairs', 'Downstairs', 'Sitting','Fall', 'Standing'],fontweight='bold',rotation='horizontal');
plt.show()


# In[ ]:


def get_confusion_matrix(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    conf = np.zeros((n_classes, n_classes))
    for actual, pred in zip(y_true, y_pred):
        conf[int(actual)][int(pred)] += 1
    return conf.astype('int')
    
mat = get_confusion_matrix(y_test, y_pred)
ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust to fit
ax.xaxis.set_ticklabels(['Walking', 'Upstairs', 'Downstairs', 'Sitting','Fall', 'Standing'],fontweight='bold',rotation='vertical');
ax.yaxis.set_ticklabels(['Walking', 'Upstairs', 'Downstairs', 'Sitting','Fall', 'Standing'],fontweight='bold',rotation='horizontal');
plot_confusion_matrix(conf_mat=mat,hide_ticks=False,show_normed=True, figsize=(7,7))


# In[ ]:




