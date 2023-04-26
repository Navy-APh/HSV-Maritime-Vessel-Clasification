# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.metrics import Precision

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


# Compute class weights
from sklearn.utils import class_weight
import numpy as np

import csv

import pandas as pd

# Define the learning rate scheduler function
def lr_scheduler(epoch):
    lr = 0.1
    if epoch > 3:
        lr = lr / 10
    if epoch > 6:
        lr = lr / 10
    if epoch > 9:
        lr = lr / 10
        
    return lr

batches_size = 16

data_types = ['Ships_Main', 'Ships_Class', 'Ships_SubClass']
no_of_classes = [98,26,12]
train_size = [32247,32247,32213]
test_size = [7946,7946,7938]

for i in range (0,3):
    # Generating images for the Training set
    train_datagen = ImageDataGenerator(rescale=1./255)

    # Generating images for the Test set
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Creating the Training set
    training_set = train_datagen.flow_from_directory(f'E:/APhD/_Projects/3) Handwritting signature/Data/{data_types[i]}/Train',
                                                     target_size=(224, 224),
                                                     batch_size=batches_size,
                                                     class_mode='categorical',shuffle=True)

    # Creating the Test set
    test_set = test_datagen.flow_from_directory(f'E:/APhD/_Projects/3) Handwritting signature/Data/{data_types[i]}/Test',
                                                 target_size=(224, 224),
                                                 batch_size=batches_size,
                                                 class_mode='categorical',shuffle=True)

    # Get the class labels for the training set
    y_train = training_set.classes

    # Compute class weights
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    # Convert class weights to dictionary
    class_weights_dict = dict(enumerate(class_weights))


    # Get the class indices
    class_indices = training_set.class_indices

    # Print the class indices
    #print(class_indices)


    with open('E:/APhD/_Projects/3) Handwritting signature/class_indices.csv', 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        
        # Write the header row
        writer.writerow(['class_name', 'class_index'])
        
        # Write each row of data
        for class_name, class_index in class_indices.items():
            writer.writerow([class_name, class_index])

    '''# Building the CNN
    model = tf.keras.models.Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(no_of_classes[i], activation='softmax'))

    # Define the learning rate scheduler callback
    lr_callback = LearningRateScheduler(lr_scheduler)

    # Compile the CNN with an initial learning rate
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the CNN on the Training set and evaluate it on the Test set
    # Train the CNN with class weighting
    model.fit(training_set,
              steps_per_epoch=int(train_size[i]/batches_size),
              epochs=12,
              validation_data=test_set,
              validation_steps=int(test_size[i]/batches_size),
              callbacks=[lr_callback])
              #,class_weight=class_weights_dict)
              
    # Save the model to disk
    model.save(f"E:/APhD/_Projects/3) Handwritting signature/Models/CNN/CNN_{data_types[i]}_model.h5")'''
    
    
    # Load the CNN
    model = load_model(f"E:/APhD/_Projects/3) Handwritting signature/Models/CNN/CNN_{data_types[i]}_model.h5")

    try:
        print("Testing the model")
        
        # Make predictions on the test set
        y_pred = model.predict(test_set)
        
        y_indices = np.argmax(y_pred, axis=1).astype(int)

        # Get the class labels for the test set
        y_true = test_set.classes.astype(int)

        # Get the class indices
        class_indices = test_set.class_indices
        
        labels_pred = [key for value in y_indices for key, code in class_indices.items() if code == value]
        y_indices = labels_pred
        
        labels_true = [key for value in y_true for key, code in class_indices.items() if code == value]
        y_true = labels_true
        
        #print(class_indices.values())
        
        
        from sklearn.metrics import confusion_matrix
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_indices)

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true, y_indices)
        print(f"Accuracy: {accuracy}")

        # create a pandas dataframe from the confusion matrix array
        cm_df = pd.DataFrame(cm, index = [i for i in range(cm.shape[0])], columns = [i for i in range(cm.shape[1])])

        # save the dataframe as a CSV file
        cm_df.to_csv(f'E:/APhD/_Projects/3) Handwritting signature/Models/CNN/CNN_{data_types[i]}_cm.csv', index=False)
    except:
        print("Error")
    


