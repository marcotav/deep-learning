import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential  
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
from keras import applications
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
import math, cv2 

folder_train = './train_toy_3/'
folder_test = './test_toy_3/'

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import EarlyStopping, Callback
K.image_data_format()  # this means that "backend": "tensorflow". Channels are RGB
from keras import applications  
from keras.utils.np_utils import to_categorical  
import math, cv2  

## Defining the new size of the image

img_width, img_height = 120,120

if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height)
    print('Theano Backend')
else:
    input_shape = (img_width, img_height, 3)
    print('TensorFlow Backend')
    
input_shape

nb_train_samples = 0
for p in range(len(os.listdir(os.path.abspath(folder_train)))):
    nb_train_samples += len(os.listdir(os.path.abspath(folder_train) +'/'+ os.listdir(
                                os.path.abspath(folder_train))[p]))
nb_train_samples

nb_test_samples = 0
for p in range(len(os.listdir(os.path.abspath(folder_test)))):
    nb_test_samples += len(os.listdir(os.path.abspath(folder_test) +'/'+ os.listdir(
                                os.path.abspath(folder_test))[p]))

train_data_dir = os.path.abspath(folder_train) # folder containing training set already subdivided
validation_data_dir = os.path.abspath(folder_test) # folder containing test set already subdivided
nb_train_samples = nb_train_samples
nb_validation_samples = nb_test_samples
epochs = 100
batch_size = 16   # batch_size = 16
num_classes = len(os.listdir(os.path.abspath(folder_train)))
print('The painters are',os.listdir(os.path.abspath(folder_train)))

### Class for early stopping

# rdcolema
class EarlyStoppingByLossVal(Callback):
    """Custom class to set a val loss target for early stopping"""
    def __init__(self, monitor='val_loss', value=0.45, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto') #

top_model_weights_path = 'bottleneck_fc_model.h5'  

### Creating InceptionV3 model

from keras.applications.inception_v3 import InceptionV3
model = applications.InceptionV3(include_top=False, weights='imagenet')  

applications.InceptionV3(include_top=False, weights='imagenet').summary()

type(applications.InceptionV3(include_top=False, weights='imagenet').summary())

### Training and running images on InceptionV3

datagen = ImageDataGenerator(rescale=1. / 255)  
   
generator = datagen.flow_from_directory(
    train_data_dir,  
    target_size=(img_width, img_height),  
    batch_size=batch_size,  
    class_mode=None,  
    shuffle=False) 
   
nb_train_samples = len(generator.filenames)  
num_classes = len(generator.class_indices)  
predict_size_train = int(math.ceil(nb_train_samples / batch_size))  
print('Number of training samples:',nb_train_samples)
print('Number of classes:',num_classes)


bottleneck_features_train = model.predict_generator(generator, predict_size_train)  # these are numpy arrays

bottleneck_features_train[0].shape
bottleneck_features_train.shape


np.save('bottleneck_features_train.npy', bottleneck_features_train)  


generator = datagen.flow_from_directory(  
     validation_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_validation_samples = len(generator.filenames)  
   
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size)) 
print('Number of testing samples:',nb_validation_samples)

bottleneck_features_validation = model.predict_generator(  
     generator, predict_size_validation)  
   
np.save('bottleneck_features_validation.npy', bottleneck_features_validation) 

### Training the fully-connected network (the top-model)


datagen_top = ImageDataGenerator(rescale=1./255)  
generator_top = datagen_top.flow_from_directory(
    train_data_dir,  
    target_size=(img_width, img_height),  
    batch_size=batch_size,  
    class_mode='categorical',  
    shuffle=False)  
   
nb_train_samples = len(generator_top.filenames)  
num_classes = len(generator_top.class_indices)  


train_data = np.load('bottleneck_features_train.npy') 

## Converting training data into vectors of categories:

train_labels = generator_top.classes  
print('Classes before dummification:',train_labels)  
train_labels = to_categorical(train_labels, num_classes=num_classes) 
print('Classes after dummification:\n\n',train_labels)  

## Again repeating the process with the validation data:

generator_top = datagen_top.flow_from_directory(
    validation_data_dir,  
    target_size=(img_width, img_height),  
    batch_size=batch_size,  
    class_mode=None,  
    shuffle=False)  
   
nb_validation_samples = len(generator_top.filenames) 

validation_data = np.load('bottleneck_features_validation.npy')  

validation_labels = generator_top.classes  
validation_labels = to_categorical(validation_labels, num_classes=num_classes)  

### Building the small FL model using bottleneck features as input

model = Sequential()  
model.add(Flatten(input_shape=train_data.shape[1:])) 
# model.add(Dense(1024, activation='relu'))  
# model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(16, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(8, activation='relu')) # Not valid for minimum = 500
model.add(Dropout(0.5)) 
# model.add(Dense(4, activation='relu')) # Not valid for minimum = 500
# model.add(Dropout(0.5)) 
model.add(Dense(num_classes, activation='sigmoid'))  
   
model.compile(optimizer='Adam',  
              loss='binary_crossentropy', metrics=['accuracy'])   
   
history = model.fit(train_data, train_labels,  
          epochs=epochs,  
          batch_size=batch_size,  
          validation_data=(validation_data, validation_labels))  
   
model.save_weights(top_model_weights_path)  
   
(eval_loss, eval_accuracy) = model.evaluate(
    validation_data, validation_labels, 
    batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("[INFO] Loss: {}".format(eval_loss))  

train_data.shape[1:]

plt.figure(1)  
   
# summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch') 
#pylab.ylim([0.4,0.68])
plt.legend(['train', 'test'], loc='upper left')  

### Plotting the loss history

import pylab
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
pylab.xlim([0,60])
# pylab.ylim([0,1000])
plt.show()  

import matplotlib.pyplot as plt
import pylab
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Classification Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
pylab.xlim([0,60])
plt.legend(['Test', 'Validation'], loc='upper right')
fig.savefig('loss.png')
plt.show();

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(figsize=(15,15))
plt.title('Classification Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
pylab.xlim([0,100])
plt.legend(['Test', 'Validation', 'Success Metric'], loc='lower right')
fig.savefig('acc.png')
plt.show();


### Predictions

os.listdir(os.path.abspath('train_toy_3/Pierre-Auguste_Renoir))

image_path = os.path.abspath('test_toy_3/Pierre-Auguste_Renoir/91485.jpg')
orig = cv2.imread(image_path) 
image = load_img(image_path, target_size=(120,120))  
image
image = img_to_array(image)  
image

image = image / 255.  
image = np.expand_dims(image, axis=0)  
image

# build the VGG16 network  
#model = applications.VGG16(include_top=False, weights='imagenet')  
model = applications.InceptionV3(include_top=False, weights='imagenet')   
# get the bottleneck prediction from the pre-trained VGG16 model  
bottleneck_prediction = model.predict(image)

# build top model  
model = Sequential()  
model.add(Flatten(input_shape=train_data.shape[1:])) 
# model.add(Dense(1024, activation='relu'))  
# model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(16, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(8, activation='relu')) # Not valid for minimum = 500
model.add(Dropout(0.5)) 
# model.add(Dense(4, activation='relu')) # Not valid for minimum = 500
# model.add(Dropout(0.5)) 
model.add(Dense(num_classes, activation='sigmoid')) 

model.load_weights(top_model_weights_path)  
   
# use the bottleneck prediction on the top model to get the final classification  
class_predicted = model.predict_classes(bottleneck_prediction)

inID = class_predicted[0]  
   
class_dictionary = generator_top.class_indices  
   
inv_map = {v: k for k, v in class_dictionary.items()}  
   
label = inv_map[inID]  
   
# get the prediction label  
print("Image ID: {}, Label: {}".format(inID, label))  
   
# display the predictions with the image  
cv2.putText(orig, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)  
   
cv2.imshow("Classification", orig)  
cv2.waitKey(0)  
cv2.destroyAllWindows()