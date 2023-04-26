# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import pandas as pd
import joblib

#data_types = ['Ships_SubClass', 'Ships_Class','Ships_Main']

data_types = ['Ships_Class','Ships_Main']

for data_type in data_types:
    
    # Importing the dataset
    dataset1 = pd.read_csv(f"E:/APhD/_Projects/3) Handwritting signature/CSV_Data/{data_type}_Test.csv")
    dataset2 = pd.read_csv(f"E:/APhD/_Projects/3) Handwritting signature/CSV_Data/{data_type}_Train.csv")

    # Merging the datasets
    dataset = pd.concat([dataset1, dataset2], ignore_index=True)

    X = dataset.iloc[:, 0:784].values
    y = dataset.iloc[:, 785].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the model to disk
    filename = f'E:/APhD/_Projects/3) Handwritting signature/Models/SVM/SVM_{data_type}_model.sav'
    joblib.dump(classifier, filename)


    # create a pandas dataframe from the confusion matrix array
    cm_df = pd.DataFrame(cm, index = [i for i in range(cm.shape[0])], columns = [i for i in range(cm.shape[1])])

    # save the dataframe as a CSV file
    cm_df.to_csv(f'E:/APhD/_Projects/3) Handwritting signature/Models/SVM/SVM_{data_type}_cm.csv', index=False)

