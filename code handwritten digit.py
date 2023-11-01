#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.datasets.mnist import load_data


# In[2]:


data = load_data()


# In[3]:


np.shape(data)


# In[4]:


(train_images,train_labels),(test_images,test_labels) = data


# In[5]:


train_images[0]


# In[6]:


train_labels[0]


# In[7]:


set(train_labels)


# In[8]:


print('Max pixel value : ', train_images[0].max()) 
print('Min pixel value : ', train_images[0].min())
print()
print('Shape of training data : ', train_images.shape)
print('Shape of each training example : ', train_images[0].shape)
print()
print('Shape of testing data : ',test_images.shape)
print('Shape of each testing example :',test_images[0].shape)


# In[9]:


print('Label : ', train_labels[0])
print(train_images[0, 5:23, 5:23])


# In[11]:


plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])


# In[12]:


np.unique(train_images[0])


# In[13]:


train_images = train_images / 255.0
test_images = test_images /255.0


# In[14]:


np.unique(train_images[0])


# In[15]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax')
])


# In[16]:


hidden_layer = model.layers[1]
weights = hidden_layer.get_weights()
print('Shape of weights : ', np.shape(weights[0]))
print('Shape of biases : ', np.shape(weights[1]))


# In[18]:


output_layer = model.layers[2]
weights = output_layer.get_weights()
print('Shape of weights : ', np.shape(weights[0]))
print('Shape of biases : ', np.shape(weights[1]))


# In[20]:


sgd = keras.optimizers.SGD(lr=0.5, decay=1e-6, momentum=0.5)
model.compile(optimizer=sgd,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[22]:


history = model.fit(train_images, train_labels, epochs=10, batch_size=100, validation_split=0.1)


# In[24]:


val_losses = history.history['val_loss']
losses = history.history['loss']
indices = range(len(losses))

plt.figure(figsize=(10, 5))
plt.plot(indices, val_losses, color='r')
plt.plot(indices, losses, color='g')
plt.legend(['Validation loss', 'Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')


# In[25]:


test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# In[26]:


predictions = model.predict(test_images)


# In[38]:


def plot_confidence(images, labels, predictions):
    plt.figure(figsize=(15,30))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)
    plot_index = 0
    for i in range(len(images)):
        plot_index += 1
        plt.subplot(len(images), 2, plot_index)
        plt.imshow(images[i], cmap=plt.cm.binary)
        correct_label = str(labels[i])
        predicted_label = str(np.argmax(predictions[i]))
        title = 'Correct Label : ' + str(labels[i]) + '\n' + 'Predicted Label : ' + str(np.argmax(predictions[i]))
        if predicted_label != correct_label:
            plt.title(title, backgroundcolor='r', color='w')
        else:
                plt.title(title, backgroundcolor='g', color='w')
        plt.xticks([])
        plt.yticks([])
                
        plot_index += 1
        plt.subplot(len(images), 2, plot_index)
        plt.bar(range(10), predictions[i])
        plt.xticks(range(10))
        plt.ylim(0, 1)


# In[39]:


images = test_images[:10]
labels = test_labels[:10]
test_predictions = predictions[:10]
plot_confidence(images, labels, test_predictions)


# In[43]:


incorrect_indices = list()

for i in range(len(predictions)):
    predicted_label = np.argmax(predictions[i])
    if predicted_label != test_labels[i]:
        incorrect_indices.append(i)
print('Number of incorrectly classified images : ', len(incorrect_indices))

incorrect_indices = incorrect_indices[:10]

incorrect_images = [test_images[i] for i in incorrect_indices]
incorrect_labels = [test_labels[i] for i in incorrect_indices]
incorrect_predictions = [predictions[i] for i in incorrect_indices]

plot_confidence(incorrect_images, incorrect_labels, incorrect_predictions)


# In[ ]:




