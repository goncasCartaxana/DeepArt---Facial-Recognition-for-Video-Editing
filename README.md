# DeepArt---Facial-Recognition-for-Video-Editing
A Python, tkinter-based GUI, video editing tool designed to detect intervals in video files where faces appear. Ideal for video editors who want to efficiently identify and extract face-containing segments for further editing or review.




# Models Used
dlib Frontal Face Detector:
- Does: Detects faces in images and creates bounding boxes. 
- Extra: Uses the dlib's built-in HOG + Linear SVM face detection model. Best for frontal or near-frontal faces. (HOG + SVM)

Facial Landmark Model:
- Name: shape_predictor_68_face_landmarks.dat
- Does: Detects facial landmarks (e.g. yes, nose, mouth, jawline)
- Link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Face Embedding Model:
- Name: dlib_face_recognition_resnet_model_v1.dat
- Does: Produces 128-dimensional numerical face embeddings. It's used for recognition/comparison. 
- Extra: Deep learning face recognition model based on ResNet.
- Link: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

Models can also be found here: https://github.com/davisking/dlib-models 



# Licenses
This project uses the dlib library and pre-trained models created by Davis E. King. dlib is licensed under the Boost Software License 1.0.
For more details, see https://dlib.net/license.html

The Boost Software License allows free use, modification, and distribution of dlib for both commercial and non-commercial purposes, provided the license text is included.
