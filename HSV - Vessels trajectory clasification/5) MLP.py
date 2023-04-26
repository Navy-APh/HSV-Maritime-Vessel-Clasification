# Importing the libraries
import numpy as np
import pandas as pd

data_types = ['Ships_SubClass', 'Ships_Class','Ships_Main']
no_of_classes = [98,26,12]


for i in range (0,3):
    # Importing the dataset
    dataset1 = pd.read_csv(f"E:/APhD/_Projects/3) Handwritting signature/CSV_Data/{data_types[i]}_Test.csv")
    dataset2 = pd.read_csv(f"E:/APhD/_Projects/3) Handwritting signature/CSV_Data/{data_types[i]}_Train.csv")

    # Merging the datasets
    dataset = pd.concat([dataset1, dataset2], ignore_index=True)

    X = dataset.iloc[:, 0:784].values
    y = dataset.iloc[:, 785].values

    from sklearn.preprocessing import LabelEncoder

    # Encoding the labels with LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Define the learning rate scheduler function
    from tensorflow.keras.callbacks import LearningRateScheduler

    # Define the learning rate scheduler function
    def lr_scheduler(epoch, lr):
        lr = 0.01
        if epoch > 5:
            lr = lr / 10
        if epoch > 10:
            lr = lr / 10
        if epoch > 15:
            lr = lr / 10
        if epoch > 20:
            lr = lr / 10
        return lr

    lr_callback = LearningRateScheduler(lr_scheduler)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # Defining the neural network model
    model = Sequential()
    model.add(Dense(4096, activation='relu', input_shape=(784,)))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(no_of_classes[i], activation='softmax'))

    # Compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_callback], shuffle=True)

    # Predicting the Test set results
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)  # convert to categorical labels

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the model to disk
    model.save(f"E:/APhD/_Projects/3) Handwritting signature/Models/MLP/MLP_{data_types[i]}_model.h5")

    # create a pandas dataframe from the confusion matrix array
    cm_df = pd.DataFrame(cm, index = [i for i in range(cm.shape[0])], columns = [i for i in range(cm.shape[1])])

    # save the dataframe as a CSV file
    cm_df.to_csv(f'E:/APhD/_Projects/3) Handwritting signature/Models/MLP/MLP_{data_types[i]}_cm.csv', index=False)