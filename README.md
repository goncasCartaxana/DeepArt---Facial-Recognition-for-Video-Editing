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

# How to Use Project
1) Open a terminal and go to a folder of your choice - where you want the files all to go into.
2) Download repository: In the project's github (here!) Click "Code" -> Copy "HTTPS" url and do "git clone <url>"
3) Download Models: Run setup_models.sh or manually create models/ folder and place there the extracted models.
4) Create python virtual environemnt (venv): Have python 3.13.9 installed:
    - Create venv: Do "py -3.13 -m venv .venv" 
    - Ativacte venv: ".venv\Scripts\activate"
    - Install dependencies: "pip install -r requirements.txt".
4) Run the script (ensure venv is activated): "python main..py"

** To deactivate venv, simply type "deactivate" in the terminal.

# Licenses
This project uses the dlib library and pre-trained models created by Davis E. King. dlib is licensed under the Boost Software License 1.0.
For more details, see https://dlib.net/license.html

The Boost Software License allows free use, modification, and distribution of dlib for both commercial and non-commercial purposes, provided the license text is included.
