import os
import pandas as pd
import glob
from PIL import Image
import numpy as np
import cv2
import csv

sliding_window_size = 50

def extract_features(image):
    
    image = np.array(image) # Convert the image to a numpy array
    
    # Extract the red, green, and blue channels
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]
    
    # Threshold the red channel to extract the traces
    _, traces = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY)
    
    # Find the bounding box of the traces
    contours, _ = cv2.findContours(traces, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        x = y = w = h = 0
    
    # Crop the image to the bounding box of the traces
    traces_image = image[y:y+h, x:x+w]
    try:
        gray = cv2.cvtColor(traces_image, cv2.COLOR_BGR2GRAY)
        width, height = gray.shape
        width = width/(sliding_window_size*1.0)
        height = height/(sliding_window_size*1.0)
        mask = np.where(gray == 0, False, True) # create the binary mask using the grayscale image
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0])) # resize the mask to match the size of the original image
    except:
        gray = None
        mask = None
        width = height = 0

    # calculate the aspect ratio
    try:
        aspect_ratio = width / height
    except:
        aspect_ratio = 0

    # calculate the number of concavities and convexities using edge detection
    if gray is not None:
       edges = cv2.Canny(gray, 1, 255)
       concavities = np.count_nonzero(edges == 255)
       convexity = np.count_nonzero(edges == 0)
    else:
        concavities = convexity = 0

    # calculate the density of pixels
    if mask is not None:
        density = np.count_nonzero(mask) / (width * height)
    else:
        density = 0

    # calculate the slant and skew of the signature
    if gray is not None:
        try:
            moments = cv2.moments(gray)
            slant = moments['mu11'] / moments['mu02']
            skew = moments['mu20'] / moments['mu02']
        except:
            slant = skew = 0
    else:
        slant = skew = 0

    # Calculate the standard deviation of the red, green, and blue channels using the mask
    if mask is not None:
        red_std = np.std(red_channel[mask])
        green_std = np.std(green_channel[mask])
        blue_std = np.std(blue_channel[mask])
        
        red_mean = np.mean(red_channel[mask])
        green_mean = np.mean(green_channel[mask])
        blue_mean = np.mean(blue_channel[mask])
        
    else:
        red_std = green_std = blue_std = 0
        red_mean = green_mean = blue_mean = 0

    # return the features as a dictionary
    return [
        height,
        width,
        aspect_ratio,
        
        concavities,
        convexity,
        density,
        slant,
        skew,
        
        red_std,
        green_std,
        blue_std,
        
        red_mean,
        green_mean,
        blue_mean
    ]
#--------------------------------------------------------------------------------------------------------------

datasets = ["Ships_Main", "Ships_Class", "Ships_SubClass"]
types_of_data = ["Test","Train"]

for dataset in datasets:
    
    for type_of_data in types_of_data:
        
        path_dataset = f"E:/APhD/_Projects/3) Handwritting signature/Data/{dataset}/{type_of_data}"
        print (path_dataset)
        print()
        
        global_list = []
        
        for root, dirs, files in os.walk(path_dataset):
            for class_of_vessels in dirs:
                
                path_to_class_pictures = f'{path_dataset}/{class_of_vessels}'
                print(path_to_class_pictures)
                print()
                
                # create a list of all PNG file names in the folder path
                png_files = os.listdir(path_to_class_pictures)

                for png in png_files:
                    path_to_png = f"{path_to_class_pictures}/{png}"
                    #print(path_to_png)
                    image = Image.open(path_to_png)
                    
                    sub_images = []
                    width, height = image.size
                    
                    for i in range(0, width, sliding_window_size):
                        for j in range(0, height, sliding_window_size):
                            sub_images.append(image.crop((i, j, i + 75, j + 75)))
                            
                    all_features_list = []
                    
                    for i, sub_image in enumerate(sub_images):
                        features = extract_features(sub_image)
                        all_features_list.append(features)
                    
                    all_features_array = np.array(all_features_list).flatten()
                    
                    all_features_array = np.append(all_features_array, png)
                    all_features_array = np.append(all_features_array, class_of_vessels)
                        
                    global_list.append(all_features_array)
        
        with open(f'CSV_Data/{dataset}_{type_of_data}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for sublist in global_list:
                writer.writerow(sublist)
        
            

    
    
