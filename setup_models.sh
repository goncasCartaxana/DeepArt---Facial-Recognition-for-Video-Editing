#!/bin/bash

mkdir -p models

curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
curl -L -o models/dlib_face_recognition_resnet_model_v1.dat.bz2 http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

echo "Download complete. Extracting..."

bzip2 -dk models/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk models/dlib_face_recognition_resnet_model_v1.dat.bz2

echo "Models ready in models/ folder."