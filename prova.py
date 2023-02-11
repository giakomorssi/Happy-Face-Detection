
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = 'Image.open(upload)'
img = np.array(img)
resize = tf.image.resize(img, (256, 256))
model = tf.keras.models.load_model('/Users/giacomorossi/Desktop/progetti/Neural Network/Happy/HappyImageClass')
yhat = model.predict(np.expand_dims(resize/255, 0))

if yhat.round() == 1:
    st.image(img)
    st.subheader('sad person')
else: 
    st.image(img)
    st.write('happy person')



