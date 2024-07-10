import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as p
import PIL as pil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import seaborn as sns

#Visualising Dataset
meta=pd.read_csv("C://Users//harsh//Downloads//HAM10000//HAM10000_metadata.csv")
meta.info()
meta.head()
g = sns.catplot(x="dx", kind="count", palette='bright', data=meta)
g.fig.set_size_inches(16, 5)

g.ax.set_title('Skin Cancer by Class', fontsize=20)
g.set_xlabels('Skin Cancer Class', fontsize=14)
g.set_ylabels('Number of Data Points', fontsize=14)

g = sns.catplot(x="dx", kind="count", hue="sex", palette='coolwarm', data=meta)
g.fig.set_size_inches(16, 5)

g.ax.set_title('Skin Cancer by Sex', fontsize=20)
g.set_xlabels('Skin Cancer Class', fontsize=14)
g.set_ylabels('Number of Data Points', fontsize=14)
g._legend.set_title('Sex')

g = sns.catplot(x="dx", kind="count", hue="age", palette='bright', data=meta)
g.fig.set_size_inches(16, 9)

g.ax.set_title('Skin Cancer by Age', fontsize=20)
g.set_xlabels('Skin Cancer Class', fontsize=14)
g.set_ylabels('Number of Data Points', fontsize=14)
g._legend.set_title('Age')



#Extracting x and y from csv file
df=pd.read_csv("C://Users//harsh//Downloads//HAM10000//hmnist_28_28_RGB.csv")
x=df.drop('label',axis=1)
y=df['label']
x=x.to_numpy()
x=x/255
y=to_categorical(y)
df['label'].value_counts()
label={
    ' Actinic keratoses':0,
    'Basal cell carcinoma':1,
    'Benign keratosis-like lesions':2,
    'Dermatofibroma':3,
    'Melanocytic nevi':4,
    'Melanoma':5,
    'Vascular lesions':6
}
x=x.reshape(-1,28,28,3)


#Spliiting into train test
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=13,stratify=df['label'])

#Image augmentation
datagen=ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='nearest')
datagen=ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='nearest')
datagen.fit(xtrain)

#Model
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
def accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall
from tensorflow.keras.optimizers import RMSprop

model=Sequential()

model.add(Conv2D(64,(2,2),input_shape=(28,28,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512,(2,2),input_shape=(28,28,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Conv2D(1024,(2,2),input_shape=(28,28,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Conv2D(1024,(1,1),input_shape=(28,28,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[accuracy])

model.summary()
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

early=EarlyStopping(monitor='accuracy',patience=3)
reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, verbose=1, mode='min', min_lr=0.0001)



#Training
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
def accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
class_weights={0:1,1:1,2:1,3:1,4:0.5,5:1,6:1}
model.fit(xtrain,ytrain,epochs=30,validation_data=(xtest,ytest),callbacks=[reduce_lr,early],class_weight=class_weights)


#Evaluation
p.figure(figsize=(15,10))
loss=pd.DataFrame(model.history.history)
loss[['accuracy','val_accuracy']].plot

p.figure(figsize=(15,10))
loss[['loss','val_loss']].plot()
decode={
    0:'Actinic keratosis',
    1:'Basal cell carcinoma',
    2:'Benign keratosis-like lesions',
    3:'Dermatofibroma',
    4:'Melanocytic nevi',
    5:'Melanoma',
    6:'Vascular lesion'  
}
p.figure(figsize=(10,8))

pred=model.predict(xtest)

from sklearn.metrics import roc_curve,auc
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(7):
    fpr[i], tpr[i], _ = roc_curve(ytest[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(7):
    p.plot(fpr[i],tpr[i],label=decode[i],linewidth=2)
p.plot([0, 1], [0, 1], 'k--', lw=2,label='random guess')
p.legend(loc="lower right")
from sklearn.metrics import classification_report,confusion_matrix

# predictions=model.predict_classes(xtest)
#predictions = (model.predict(x_test) > 0.5).astype("int32")
predict_x=model.predict(xtest) 
predictions=np.argmax(predict_x,axis=1)

check=[]
for i in range(len(ytest)):
  for j in range(7):
    if(ytest[i][j]==1):
      check.append(j)
check=np.asarray(check)
print(classification_report(check,predictions))



#image testing




# from PIL import Image
# import numpy as np
# import cv2

# image=Image.open('D://Skin-Disease-Prediction-Web-Application-master//Skin-Disease-Prediction-Web-Application-master//media//test//ISIC_0024332.jpg')
# image=np.array(image)
# image=cv2.resize(image,(28,28))/255
# image=np.expand_dims(image,axis=0)
# pred=np.argmax(model.predict(image))
# label = {
#           0:'Actinic keratoses',
#           1:'Basal cell carcinoma',
#           2:'Seborrhoeic Keratosis',
#           3:'Dermatofibroma',
#           4:'Melanocytic nevi',
#           6:'Melanoma',
#           5:'Vascular lesions'
# }
# disease=label[pred]
# params={'disease':disease,'disp':True,'val':pred}
# print(params)
# import pickle
# pickle.dump(model,open("skinD.pkl","wb"))
# save_model(model, 'Model.h5')
model.save('Model.h5')