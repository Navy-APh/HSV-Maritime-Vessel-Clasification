
import pandas as pd
import os

df = pd.read_csv('MMSI_list_all.csv')

# Creating folders for shipType
unique_classes = df['shipType'].unique()
path_train_class = 'E:/APhD/_Projects/3) Handwritting signature/Data/Ships_Class/Train/'
path_test_class = 'E:/APhD/_Projects/3) Handwritting signature/Data/Ships_Class/Test/'

for class_name in unique_classes:
    train_path = os.path.join(path_train_class, class_name)
    test_path = os.path.join(path_test_class, class_name)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

# Creating folders for shipTypeSpecific
unique_subclasses = df['shipTypeSpecific'].unique()
path_train_subclass = 'E:/APhD/_Projects/3) Handwritting signature/Data/Ships_SubClass/Train/'
path_test_subclass = 'E:/APhD/_Projects/3) Handwritting signature/Data/Ships_SubClass/Test/'

for subclass_name in unique_subclasses:
    train_path = os.path.join(path_train_subclass, subclass_name)
    test_path = os.path.join(path_test_subclass, subclass_name)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)



import pandas as pd
import os
import random
import shutil
from PIL import Image
from skimage.color import rgb2gray

def generate_random_value():
    # Generate a list of 1's and 0's with probabilities 0.7 and 0.3
    values = [1] * 80 + [0] * 20
    # Return a randomly chosen element from the list
    return random.choice(values)

df = pd.read_csv('MMSI_list_all.csv')

folder_path = "E:/APhD/_Projects/3) Handwritting signature/Data/Mixed"

# Get all file names in the folder
file_names = os.listdir(folder_path)

# Print all file names
for file_name in file_names:
    split_name = file_name.split("_")
    MMSI = float(split_name[0])

    row = df.loc[df['MMSI'] == MMSI]
    if not row.empty:
        shipType = row['shipType'].iloc[0]
        shipTypeSpecific = row['shipTypeSpecific'].iloc[0]
    else:
        print(f"No data found for MMSI: {MMSI}")
        continue
        
    is_for_training = generate_random_value()
    
    if (is_for_training):
        src_file = f"{folder_path}/{file_name}"

        dst_file_shipType = f"E:/APhD/_Projects/3) Handwritting signature/Data/Ships_Class/Train/{shipType}/{file_name}"
        dst_file_shipTypeSpecific = f"E:/APhD/_Projects/3) Handwritting signature/Data/Ships_SubClass/Train/{shipTypeSpecific}/{file_name}"

    else:
        src_file = f"{folder_path}/{file_name}"

        dst_file_shipType = f"E:/APhD/_Projects/3) Handwritting signature/Data/Ships_Class/Test/{shipType}/{file_name}"
        dst_file_shipTypeSpecific = f"E:/APhD/_Projects/3) Handwritting signature/Data/Ships_SubClass/Test/{shipTypeSpecific}/{file_name}"

    # open image file
    img = Image.open(src_file)

    # crop image
    img_cropped = img.crop((50, 50, img.size[0] - 150, img.size[1] - 50))

        # check if the cropped image has non-zero pixels
    if rgb2gray(img_cropped.getdata()).sum() > 0.9122109373622625e-06:
        # save cropped image to destination directories
        if not os.path.exists(os.path.dirname(dst_file_shipType)):
            os.makedirs(os.path.dirname(dst_file_shipType))
            if not os.path.exists(os.path.dirname(dst_file_shipTypeSpecific)):
                os.makedirs(os.path.dirname(dst_file_shipTypeSpecific))

        img_cropped.save(dst_file_shipType)
        img_cropped.save(dst_file_shipTypeSpecific)
    else:
        print(f"Image {file_name} is completely black and was not saved.")
        