"""
import deepface
"""
import os

import kagglehub
from deepface import DeepFace
img_path = 'deepface'
DeepFace.verify()

# path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
# print("Path to dataset files: ", path)
# for file in os.listdir(path):
#     #print(file)
#     for s_file in os.listdir(os.path.join(path, file)):
#         print(s_file)
#         for sub_file in os.listdir(os.path.join(path, file, s_file)):
#             print(sub_file)