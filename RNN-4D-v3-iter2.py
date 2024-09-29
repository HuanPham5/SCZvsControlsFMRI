#!/usr/bin/env python
# coding: utf-8

# # RNN 4D - no collapsing
# # V3 - 3d_lstm - iter 2

# In[1]:


import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import glob
import nibabel as nib
import os
import matplotlib.pyplot as plt
import scipy.ndimage
import random
from tensorflow.keras.layers import Dropout, Dense, Reshape, Flatten, Conv3D, Conv3DTranspose, LeakyReLU, Input, Embedding, multiply, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Bidirectional, AdditiveAttention, LayerNormalization
from functools import partial
from tensorflow.keras import models, layers


# In[2]:


#pip install nibabel


# In[3]:


#pip install numpy==1.24.0


# In[4]:


full_schizophrenia_ids = [
    'A00009280', 'A00028806', 'A00023132', 'A00014804', 'A00016859', 'A00021598', 'A00001181', 'A00023158',
    'A00024568', 'A00028405', 'A00001251', 'A00000456', 'A00015648', 'A00002405', 'A00027391', 'A00016720',
    'A00018434', 'A00016197', 'A00027119', 'A00006754', 'A00009656', 'A00038441', 'A00012767', 'A00034273',
    'A00028404', 'A00035485', 'A00024684', 'A00018979', 'A00027537', 'A00004507', 'A00001452', 'A00023246',
    'A00027410', 'A00014719', 'A00024510', 'A00000368', 'A00019293', 'A00014830', 'A00015201', 'A00018403',
    'A00037854', 'A00024198', 'A00001243', 'A00014590', 'A00002337', 'A00024953', 'A00037224', 'A00027616',
    'A00001856', 'A00037619', 'A00024228', 'A00038624', 'A00037034', 'A00037649', 'A00022500', 'A00013216',
    'A00020787', 'A00028410', 'A00002480', 'A00028303', 'A00020602', 'A00024959', 'A00018598', 'A00014636',
    'A00019349', 'A00017147', 'A00023590', 'A00023750', 'A00031597', 'A00015518', 'A00018317', 'A00016723',
    'A00021591', 'A00023243', 'A00017943', 'A00023366', 'A00014607', 'A00020414', 'A00035003', 'A00028805',
    'A00029486', 'A00000541', 'A00028408', 'A00000909', 'A00031186', 'A00000838' ]

# schizohrenia_id that satisfy t>90, 59 in total
met_requirement_schizophrenia_ids = [
    'A00000368', 'A00000456', 'A00000541', 'A00000838', 'A00001251', 'A00001452', 'A00004507',
    'A00006754', 'A00009280', 'A00012767', 'A00013216', 'A00014607', 'A00014719', 'A00014804',
    'A00014830', 'A00015201', 'A00015648', 'A00016197', 'A00016720', 'A00016723', 'A00017147',
    'A00018317', 'A00018403', 'A00018434', 'A00018979', 'A00019293', 'A00020414', 'A00020602', 
    'A00020787', 'A00021591', 'A00021598', 'A00023158', 'A00023246', 'A00023590', 'A00023750', 
    'A00024198', 'A00024228', 'A00024568', 'A00024684', 'A00024953', 'A00024959', 'A00027410', 
    'A00027537', 'A00028303', 'A00028404', 'A00028408', 'A00028805', 'A00028806', 'A00031186', 
    'A00031597', 'A00034273', 'A00035003', 'A00035485', 'A00037034', 'A00037224', 'A00037619', 
    'A00037649', 'A00038441', 'A00038624']

full_control_ids = [
    'A00007409', 'A00013140', 'A00021145', 'A00036049', 'A00022810', 'A00002198', 'A00020895', 'A00004667',
    'A00015826', 'A00023120', 'A00022837', 'A00010684', 'A00009946', 'A00037318', 'A00033214', 'A00022490',
    'A00023848', 'A00029452', 'A00037564', 'A00036555', 'A00023095', 'A00022729', 'A00024955', 'A00024160',
    'A00011725', 'A00027487', 'A00024446', 'A00014898', 'A00015759', 'A00028409', 'A00017294', 'A00014522',
    'A00012995', 'A00031764', 'A00025969', 'A00033147', 'A00018553', 'A00023143', 'A00036916', 'A00028052',
    'A00023337', 'A00023730', 'A00020805', 'A00020984', 'A00000300', 'A00010150', 'A00024932', 'A00035537',
    'A00022509', 'A00028406', 'A00004087', 'A00035751', 'A00023800', 'A00027787', 'A00022687', 'A00023866',
    'A00021085', 'A00022619', 'A00036897', 'A00019888', 'A00021058', 'A00022835', 'A00037495', 'A00026945',
    'A00018716', 'A00026907', 'A00023330', 'A00016199', 'A00037238', 'A00023131', 'A00014120', 'A00021072',
    'A00037665', 'A00022400', 'A00003150', 'A00024372', 'A00021081', 'A00022592', 'A00022653', 'A00013816',
    'A00014839', 'A00031478', 'A00014225', 'A00013363', 'A00037007', 'A00020968', 'A00024301', 'A00024820',
    'A00035469', 'A00029226', 'A00022915', 'A00022773', 'A00024663', 'A00036844', 'A00009207', 'A00024535',
    'A00022727', 'A00011265', 'A00024546'
]

 # 82 controls that met requirement
met_requirement_control_ids = [
    'A00000300', 'A00002198', 'A00003150', 'A00004087', 'A00007409', 'A00010684', 'A00011265', 'A00011725',
    'A00012995', 'A00013140', 'A00013816', 'A00014839', 'A00014898', 'A00015759', 'A00015826', 'A00018553',
    'A00018716', 'A00019888', 'A00020805', 'A00020895', 'A00020968', 'A00020984', 'A00021058', 'A00021072',
    'A00021081', 'A00021085', 'A00022400', 'A00022490', 'A00022509', 'A00022592', 'A00022619', 'A00022653',
    'A00022687', 'A00022727', 'A00022729', 'A00022773', 'A00022810', 'A00022835', 'A00022837', 'A00022915',
    'A00023095', 'A00023120', 'A00023131', 'A00023143', 'A00023330', 'A00023337', 'A00023730', 'A00023800',
    'A00023848', 'A00023866', 'A00024160', 'A00024301', 'A00024372', 'A00024446', 'A00024535', 'A00024546', 
    'A00024663', 'A00024820', 'A00024932', 'A00024955', 'A00025969', 'A00026945', 'A00027487', 'A00027787', 
    'A00028052', 'A00028406', 'A00028409', 'A00029226', 'A00029452', 'A00031478', 'A00031764', 'A00033214', 
    'A00035751', 'A00036049', 'A00036555', 'A00036844', 'A00037007', 'A00037238', 'A00037318', 'A00037495', 
    'A00037564', 'A00037665'
]


# In[5]:


# import nibabel as nib
# import os
# import shutil

# # Directory containing your .nii.gz files
# directory_path = '4D/'
# file_pattern = '*.nii.gz'  # Adjust as needed for your file pattern

# # Directory to move corrupt files, create if doesn't exist
# corrupt_files_dir = os.path.join(directory_path, 'corrupt_files')
# os.makedirs(corrupt_files_dir, exist_ok=True)

# # List to store paths of corrupt files
# corrupt_files = []

# # Iterate over all files in the directory
# for root, _, files in os.walk(directory_path):
#     for file in files:
#         if file.endswith('.nii.gz'):
#             file_path = os.path.join(root, file)
#             try:
#                 # Attempt to load the file
#                 t1_img = nib.load(file_path)
#                 # Attempt to read the data to ensure it's not corrupt
#                 t1_data = t1_img.get_fdata()

#             except (EOFError, OSError, nib.filebasedimages.ImageFileError) as e:
#                 # Log the corrupt file and the error message
#                 print(f"Corrupt file detected: {file_path} | Error: {e}")
#                 corrupt_files.append(file_path)

#                 # Optionally, move the corrupt file to a separate directory
#                 shutil.move(file_path, os.path.join(corrupt_files_dir, file))
#                 continue

# # Output the list of corrupt files
# if corrupt_files:
#     print(f"\nTotal corrupt files found: {len(corrupt_files)}")
#     for corrupt_file in corrupt_files:
#         print(corrupt_file)
# else:
#     print("No corrupt files found.")


# In[6]:


# GAN Training Data Selection
gan_train_ids_schiz = random.sample(met_requirement_schizophrenia_ids, 50)
gan_test_ids_schiz = [id for id in met_requirement_schizophrenia_ids if id not in gan_train_ids_schiz]

gan_train_ids_control = random.sample(met_requirement_control_ids, 50)
gan_test_ids_control = [id for id in met_requirement_control_ids if id not in gan_train_ids_control]
gan_test_ids_control = random.sample(gan_test_ids_control,9)

''' data training for classifier '''
''' just use the same train set as GAN above '''

# Classifier Test Data Selection
classifier_test_ids = gan_test_ids_schiz + gan_test_ids_control

''' File loading '''
# Specify the directory and file pattern
directory_path = '4D/'
file_pattern = 'A*_????_func_FL_FD_RPI_DSP_MCF_SS_SM_Nui_CS_InStandard.nii.gz'

# Construct the full path pattern
path_pattern = f'{directory_path}/{file_pattern}'

# Use glob to find all matching files
matching_files = glob.glob(path_pattern)

''' File loading for GAN Training and classifer '''
''' But this time we have 2 separate GANs, 1 train on schizoprenia and 1 train on control'''

#classifier_image_data = []
#classifier_labels = []  # 1 for schizophrenia, 0 for non-schizophrenia
gan_image_data_schiz = []
gan_image_data_control = []

for file_path in matching_files:
    filename = os.path.basename(file_path)
    file_id = filename.split('_')[0]
    
    if file_id in gan_train_ids_schiz:
        t1_img = nib.load(file_path)
        t1_data = t1_img.get_fdata()
        

        if t1_data.shape[3] < 90:
            continue

        #t1_data = np.sum(t1_data, axis=1)
        #print('shape of image: ', t1_data.shape)
        gan_image_data_schiz.append(t1_data)

    if file_id in gan_train_ids_control:
        t1_img = nib.load(file_path)
        t1_data = t1_img.get_fdata()

        if t1_data.shape[3] < 90:
            continue

        #t1_data = np.sum(t1_data, axis=1)
        gan_image_data_control.append(t1_data)


print(f"Total GAN control loaded: {len(gan_image_data_control)}")
print(f"Total GAN schiz loaded: {len(gan_image_data_schiz)}")



'''Determine the maximum time-dimension size '''
max_z_size_schiz = max(img.shape[3] for img in gan_image_data_schiz)
max_z_size_control = max(img.shape[3] for img in gan_image_data_control)
max_t_size = max(max_z_size_schiz,max_z_size_control)


# Normalize and pad the data
def normalize_and_pad(data, max_t):
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1
    padded = np.pad(normalized, ((0, 0), (0, 0), (0, 0), (0, max_t - data.shape[3])), mode='constant')
    return padded

padded_data_schiz = [normalize_and_pad(img, max_t_size) for img in gan_image_data_schiz]
padded_data_control = [normalize_and_pad(img, max_t_size) for img in gan_image_data_control]

padded_data_array_schiz = padded_data_schiz
padded_data_array_control = padded_data_control
print("shape after normalization and padding", padded_data_array_control[0].shape)


# In[7]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth for GPUs.")
    except RuntimeError as e:
        print(e)


# In[8]:


import numpy as np
import tensorflow as tf

# Batch size
batch_size = 2

# Create labels
labels_schiz = np.ones(len(padded_data_array_schiz))
labels_control = np.zeros(len(padded_data_array_control))

# Combine images and labels
train_images = padded_data_array_schiz + padded_data_array_control
train_labels = np.concatenate((labels_schiz, labels_control), axis=0)

# Shuffle indices
indices = np.arange(len(train_images))
np.random.shuffle(indices)

# Shuffle data based on indices
train_images = [train_images[i] for i in indices]
train_labels = train_labels[indices]

# Define a generator function to yield data batches
def data_generator(images, labels, batch_size):
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        # Reshape to (batch_size, time, x, y, z, 1) to add a channel dimension
        batch_images = np.array(batch_images).transpose(0, 4, 1, 2, 3)[..., np.newaxis]  
        yield batch_images, np.array(batch_labels)

# Create TensorFlow Dataset from the generator
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_images, train_labels, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, max_t_size, 91, 109, 91, 1), dtype=tf.float32),  # Adjust shape for ConvLSTM3D
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

# Prefetch for performance improvement
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Debug: Test the generator
for images, labels in train_dataset.take(1):
    print(f"Batch image shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")


# In[10]:


# Parallelize
# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Available GPUs: {[gpu.name for gpu in gpus]}")
else:
    print("No GPUs detected.")

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Initialize MirroredStrategy to utilize all available GPUs
strategy = tf.distribute.MirroredStrategy()  # Automatically detects and uses all GPUs

# Wrap model creation and training code within the strategy scope
with strategy.scope():
    def build_rnn_model():
        # Define input shape: (time_steps, x, y, z, channels)
        time_steps = max_t_size  # Number of time points
        input_shape = (time_steps, 91, 109, 91, 1)  # Assuming a single channel for each volume
    
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # ConvLSTM3D layer to process 3D spatial-temporal data
        convlstm_out = layers.ConvLSTM3D(filters=2, kernel_size=(3, 3, 3), padding='same', return_sequences=True, data_format='channels_last')(inputs)
    
        # Use TimeDistributed to apply pooling to each time step
        pooled_out = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling3D())(convlstm_out)
        # Dense layer with reduced size
        dense_out = tf.keras.layers.Dense(8, activation='relu')(pooled_out)
        
        # Output layer for binary classification
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense_out)
    
        # Compile model
        model = tf.keras.models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    
    rnn_model = build_rnn_model()
    rnn_model.summary()

    # Metrics for training
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    # Define number of epochs
    epochs = 100

    # Define a step function for distributed training
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = rnn_model(images, training=True)
            # Remove squeeze and directly compute loss
            loss = tf.keras.losses.binary_crossentropy(labels, predictions[:, -1, 0])
    
        gradients = tape.gradient(loss, rnn_model.trainable_variables)
        rnn_model.optimizer.apply_gradients(zip(gradients, rnn_model.trainable_variables))
    
        train_loss.update_state(loss)
        train_accuracy.update_state(labels, predictions[:, -1, 0])

    # Training loop using the existing train_dataset
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        # Iterate through the training dataset using strategy.run()
        for images, labels in train_dataset:
            # Use strategy.run to ensure that each step runs in the correct replica context
            strategy.run(train_step, args=(images, labels))

        # Print the loss and accuracy for the current epoch
        print(f"Epoch {epoch + 1}, Loss: {train_loss.result().numpy()}, Training Accuracy: {train_accuracy.result().numpy()}")


# In[11]:


# Save the model
rnn_model.save('3d_conv_lstm_iter2.h5')


# In[12]:


import numpy as np
import tensorflow as tf

test_image_data = []
test_labels = []

test_ids = classifier_test_ids  # List of IDs to filter test data

for file_path in matching_files:
    filename = os.path.basename(file_path)
    file_id = filename.split('_')[0]
    
    if file_id in test_ids:
        t1_img = nib.load(file_path)
        t1_data = t1_img.get_fdata()
        
        if t1_data.shape[3] < 90:
            continue
    
        # Normalize the processed image
        processed_image_normalized = (t1_data - np.min(t1_data)) / (np.max(t1_data) - np.min(t1_data)) * 2 - 1

        # Pad or truncate the time dimension to match the expected size (max_t_size)
        current_t_size = processed_image_normalized.shape[3]
        if current_t_size < max_t_size:
            pad_size = max_t_size - current_t_size
            processed_image_padded = np.pad(
                processed_image_normalized, 
                ((0, 0), (0, 0), (0, 0), (0, pad_size)), 
                mode='constant'
            )
        elif current_t_size > max_t_size:
            processed_image_padded = processed_image_normalized[:, :, :, :max_t_size]
        else:
            processed_image_padded = processed_image_normalized

        # Reshape to add channel dimension
        processed_image_padded = np.expand_dims(processed_image_padded, axis=-1)  # Shape: (91, 109, 91, t, 1)
        processed_image_padded = np.transpose(processed_image_padded, (3, 0, 1, 2, 4))
        
        test_image_data.append(processed_image_padded)
        
        label = 1 if file_id in met_requirement_schizophrenia_ids else 0
        test_labels.append(label)

# Convert to numpy arrays for easier handling in TensorFlow
test_images_array = np.array(test_image_data)

# Reshape each label to match (146, 1) if applicable.
reshaped_labels = []
for label in test_labels:
    # If label corresponds to a full sequence, adjust shape to (146, 1).
    reshaped_labels.append(np.full((146, 1), label))

# Convert the reshaped labels list into a numpy array.
test_labels_array = np.array(reshaped_labels)

# Create a TensorFlow dataset from the numpy arrays
batch_size = 1
test_dataset = tf.data.Dataset.from_tensor_slices((test_images_array, test_labels_array)).batch(batch_size)


# In[13]:


# Evaluate the model
# Define evaluation step without using tf.function
def evaluation_step(images, labels):
    predictions = rnn_model(images, training=False)
    loss = tf.keras.losses.binary_crossentropy(labels, predictions[:, -1, 0])
    test_loss.update_state(loss)
    test_accuracy.update_state(labels, predictions[:, -1, 0])

# Initialize metrics for evaluation
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

# Reset states before evaluating
test_loss.reset_states()
test_accuracy.reset_states()

# Evaluate on test_dataset using strategy.run() without tf.function
for images, labels in test_dataset:
    strategy.run(evaluation_step, args=(images, labels))

# Print evaluation results
print(f"Test Loss: {test_loss.result().numpy()}, Test Accuracy: {test_accuracy.result().numpy()}")


# In[15]:


# Predict the probabilities
predictions = rnn_model.predict(test_dataset)

# Convert probabilities to class labels
predicted_labels = (predictions > 0.5).astype(int)

# Extract the last time step prediction for each sequence
predicted_labels_last = predicted_labels[:, -1].flatten()


actual_labels = test_labels_array
# Flatten the actual labels to match the shape
actual_labels_flat = actual_labels.flatten()

# Extract one label per sequence from actual labels to match predicted labels
actual_labels_per_sequence = actual_labels_flat[::146]  # Assuming each sequence has 146 time steps

# Check if the lengths match and print the confusion matrix and classification report
if len(actual_labels_per_sequence) == len(predicted_labels_last):
    from sklearn.metrics import classification_report, confusion_matrix

    print(confusion_matrix(actual_labels_per_sequence, predicted_labels_last))
    print(classification_report(actual_labels_per_sequence, predicted_labels_last))
else:
    print("The lengths of actual labels per sequence and predicted labels still don't match.")


# In[16]:


import pandas as pd

# Create a DataFrame to compare predicted vs. actual labels
comparison_df = pd.DataFrame({
    'Predicted Labels': predicted_labels_last,
    'Actual Labels': actual_labels_per_sequence
})

# Display the first few rows of the comparison
print(comparison_df.head())

# Print the full comparison DataFrame
print(comparison_df)


# In[ ]:




