import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc

import tensorflow as tf
from keras.losses import binary_crossentropy
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Input, BatchNormalization, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.saving import register_keras_serializable

from tensorflow.keras import optimizers
from tensorflow.keras.metrics import Precision, Recall

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image   

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * tf.keras.backend.sum(intersection) + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return 1. - score

### bce_dice_loss = binary_crossentropy_loss + dice_loss
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def focal_loss(y_true, y_pred):
    gamma=2.
    alpha=0.25

    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())

    # Cross entropy
    cross_entropy = -y_true * tf.keras.backend.log(y_pred)

    # Compute focal loss
    loss = alpha * tf.keras.backend.pow(1 - y_pred, gamma) * cross_entropy
    return tf.keras.backend.sum(loss, axis=-1)


precision = Precision()
recall = Recall()

# Define F1-Score Metric
@register_keras_serializable()
def f1_metric(y_true, y_pred):
    # Calculate Precision and Recall using the already instantiated objects
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    
    # F1 Score calculation
    return 2 * (precision_value * recall_value) / (precision_value + recall_value + tf.keras.backend.epsilon())

@register_keras_serializable()
def rmse_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Ensure that y_true is of type float32
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


def iou_metric(y_true, y_pred):
    # Flatten tensors and cast them to float32 to ensure matching data types
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))  # Cast y_true to float32
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))  # Cast y_pred to float32

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    # Avoid division by zero and compute IOU
    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou


# Load the models
segmentation_model = load_model('model_best_checkpoint.keras', custom_objects={'bce_dice_loss': bce_dice_loss,'iou_metric':iou_metric, 'f1_metric': f1_metric, 'rmse_metric': rmse_metric})
classification_model = load_model('classification_best_checkpoint.keras', custom_objects={'focal_loss': focal_loss, 'f1_metric': f1_metric})

# Set threshold for segmentation
THRESHOLD = 0.2

# Define function to preprocess image
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize the image to 128x128
    img = img.convert("L")  # Convert image to grayscale for a single channel
    img_array = img_to_array(img)  # Convert to array, resulting in shape (128, 128, 1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension, resulting in shape (None, 128, 128, 1)
    img_array = img_array / 255.0  # Normalize image to [0, 1]
    print(img_array.shape)
    return img_array



# Define function to make predictions
def predict_image(img):
    # Preprocess the image
    img_array = preprocess_image(img)

    # Get the predicted segmentation mask
    predicted_mask = segmentation_model.predict(img_array)
    predicted_mask = (predicted_mask > THRESHOLD).astype(np.uint8)  # Binarize the output

    # Get the predicted class for the tumor
    predicted_class = np.argmax(classification_model.predict(img_array), axis=1)

    # Convert predicted class to tumor type
    tumor_types = ['Glioma', 'Meningioma', 'Pituitary']
    predicted_tumor_type = tumor_types[predicted_class[0]]

    return predicted_mask[0], predicted_tumor_type

# Streamlit interface
st.title("MRI Tumor Segmentation and Classification")

st.write("Upload an MRI image, and the app will segment the tumor and predict its type.")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file, target_size=(128, 128))  # Resize to match model input size
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Make predictions
    predicted_mask, predicted_tumor_type = predict_image(img)

    # Display the predicted tumor type
    st.write(f"Predicted Tumor Type: {predicted_tumor_type}")

    # Plot the MRI image with the predicted mask overlaid
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original MRI Image
    ax[0].imshow(np.array(img), cmap='gray')
    ax[0].set_title("MRI Image")
    ax[0].axis('off')

    # Predicted Mask
    ax[1].imshow(np.array(img), cmap='gray')
    ax[1].imshow(predicted_mask, alpha=0.5, cmap='jet')  # Overlay mask with alpha blending
    ax[1].set_title("Segmented Tumor Mask")
    ax[1].axis('off')

    st.pyplot(fig)

