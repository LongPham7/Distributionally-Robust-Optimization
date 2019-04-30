"""
The code is attributed to the GitHub page of foolbox:
https://github.com/bethgelab/foolbox
"""

import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt

img_rows, img_cols = 224, 224
nb_channels = 3
img_shape = (img_rows, img_cols, nb_channels)

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

img_path = './sample_image.jpg' # An image of a yellow cab/taxi
x = image.load_img(img_path, color_mode='rgb', target_size=(img_rows, img_cols))
img = image.img_to_array(x)
img = np.expand_dims(img, axis=0)
img = img.reshape(img_shape)
label = 468 # For the class of taxi and cab

prediction = kmodel.predict(preprocess_input(img).reshape((1, img_rows, img_cols, nb_channels)))
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Prediction on the adversarial example:', decode_predictions(prediction, top=3)[0])
# The original image is correctly classified as a cab with the confidence of 0.996. 

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(img[:, :, ::-1], label)
# if the attack fails, adversarial will be None and a warning will be printed

adversarial_prediction = kmodel.predict(adversarial.reshape((1, img_rows, img_cols, nb_channels)))
print('Prediction on the adversarial example:', decode_predictions(adversarial_prediction, top=3)[0])
# The adversarial example is incorrectly classified as an ear with the confidence of 0.302. 

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(img / 255)  # division by 255 to convert [0, 255] to [0, 1]
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Difference')
difference = adversarial[:, :, ::-1] - img
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Adversarial')
plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
plt.axis('off')

plt.tight_layout()
#plt.show()

folderpath = "./images/"
filepath = folderpath + "sample_adversarial_example.png"
plt.savefig(filepath, dpi=300)
plt.close()
