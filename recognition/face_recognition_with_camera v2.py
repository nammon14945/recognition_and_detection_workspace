import cv2
import pickle
import numpy as np
import os
import time




root_directory = '/home/nammon/Desktop/my_workspace/workspace/CS64_Project/dataset _img_testandtrain/for_train'
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_data=[]
i=0


def list_directories(path):
    items = os.listdir(path)
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
    return directories


def find_photo_files(root_dir):
    photo_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                full_path = os.path.join(root, file)
                photo_files.append(full_path)
    return photo_files


found_files = find_photo_files(root_directory)


for file in found_files:
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = img[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i = i+1
    print("Program has saved your file: ",file)


faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)


directories = list_directories(root_directory)
for directory in directories:
    if 'names.pkl' not in os.listdir('model/'):
        names = [directory]*100
        with open('model/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('model/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names+[directory]*100
        with open('model/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    if 'faces_data.pkl' not in os.listdir('model/'):
        with open('model/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('model/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('model/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)