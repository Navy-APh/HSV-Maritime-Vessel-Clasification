import pandas as pd

datasets = ['Ships_Main', 'Ships_Class', 'Ships_SubClass']
#models = ['KNN', 'SVM', 'MLP','CNN']
models = ['KNN','SVM']
classes_IDs = {
    # fishing, law, SAR, cargo
    'Ships_Main': [2, 3, 9, 0],
    'Ships_Class': [8, 10, 18, 1],
    'Ships_SubClass': [18, 52, 71, 5]
}

for dataset in datasets:
    
    classes = classes_IDs[dataset]
    print('-------------------\n' + dataset + '\n')
    
    for model in models:
        
        print(model + '\n')
        
        # Read the confusion matrix from a CSV file
        cm = pd.read_csv(f"E:/APhD/_Projects/3) Handwritting signature/Models/{model}/{model}_{dataset}_cm.csv")
        
        for class_id in classes:
            # Calculate FNR for current class
            TP = cm.iloc[class_id, class_id]
            FN = cm.iloc[class_id].sum() - TP
            FNR = FN / (TP + FN)
            
            if class_id != classes[-1]:
                print(f"Class {class_id} FNR: {FNR:.4f}")
            
            # Calculate FPR for the last class
            if class_id == classes[-1]:
                FP = cm.iloc[:-1, -1].sum()
                TN = cm.iloc[:-1, :-1].sum().sum()
                FPR = FP / (FP + TN)
                print(f"Class {class_id} FPR: {FPR:.4f}")
        
        print()
