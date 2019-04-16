# Transfer Learning with pre-trained models

import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions

model = keras.applications.inception_v3.InceptionV3()

img = image.load_img("lionNN.jpg", target_size=(299, 299))


### Converting to an array and changing the dimension:

x = image.img_to_array(img)
print(x.shape)
x = np.expand_dims(x, axis=0)
print(x.shape)

### Preprocessing and making predictions

x = preprocess_input(x)

predictions = model.predict(x)


predicted_classes = decode_predictions(predictions, top=9)

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(name,':', likelihood)

