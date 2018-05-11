# Transfer Learning ![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/keras-v2.1.5-red.svg) ![Image title](https://img.shields.io/badge/numpy-1.14.2-green.svg)


Keras has many well-known pre-trained image recognition models already built in. These models are trained using images from the Imagenet data set, a collection of millions of pictures of labeled objects. Transfer learning is when we reuse these pre-trained models in consort with our own models. If you need to recognize an object not included in the Imagenet set, you can start with the pre-trained model and fine-tune it if need be, and that makes things much easier and faster than starting from scratch. 

I will illustrate this using the Inception V3 deep neural network model. We first load the Keras' Inception V3 model creating a new object `model`. 
```
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
```
Note that the size of the image needs and the number of input nodes must match. For Inception V3, images need to be 299 pixels by 299 pixels so depending on the size of your input images, you may need to resize them.
```
img = image.load_img("lionNN.jpg", target_size=(224, 224))
```
The next step is to convert `img` into a `numpy` array and since Keras expects a list of images we must add a forth component:

```
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
```

<br>
<p align="center">
  <img src="images/lionNN.jpg",width="150" height="200">
</p>
<br>

Pixel values vary from zero to 25 but with neural nets, smaller numbers perform best. We therefore must rescale the data before inputing it. More specifically, we must normalize to the range of the images used by the trained network. We do this using `preprocess_input`:
```
x = InceptionV3.preprocess_input(x)
```
Now we run the sclaed data through the network. To make a prediction we call `model.predict` passing the image x above:
```
predictions = model.predict(x)
```
This cell returns a `predictions` object which is an array of 1,000 floats. There floats represent the likelihood that our image contains each of the 1,000 objects recognized by the pre-trained model. The last stepo is to look up the names of the classes:
```
predicted_classes = InceptionV3.decode_predictions(predictions, top=9)
```
The output of the predictions is:
```
 lion : 0.9088954
collie : 0.0037420776
chow : 0.0013897745
leopard : 0.0013692076
stopwatch : 0.00096159696
cheetah : 0.0008766686
Arabian_camel : 0.0006716717
tiger : 0.00063297455
hyena : 0.00061666104
 ```
 
