"""
models.py
----------
Part of the FaceCut project (facecut package).

Responsible for locating and loading the dlib models used throughout the
app: the face detector, the facial-landmark predictor, and the face-
embedding model. This is the single source of truth for model paths and
loaded model objects — every other module that needs face detection or
embedding capabilities imports the already-loaded objects from here instead
of re-loading them (loading these models is not cheap, so we want it done
exactly once, at import time).

Major objects:
- detector (dlib.get_frontal_face_detector):
    Locates faces in an image and returns bounding boxes. Uses dlib's
    built-in HOG + Linear SVM detector — fast, works best on frontal or
    near-frontal faces. Used by both face_utils.py (to find the single
    face in a reference image) and video_processing.py (to find faces in
    each scanned video frame).

- predictor (dlib.shape_predictor):
    Given an image and a face bounding box, predicts 68 facial landmarks
    (eyes, nose, mouth, jawline). Used internally by face_utils.py to
    align a detected face before it's turned into an embedding — this
    step is what makes the embedding comparison robust to head pose.

- face_rec_model (dlib.face_recognition_model_v1):
    Turns an aligned face into a 128-dimensional embedding vector. This
    vector is the actual representation compared between the reference
    photo and each video frame to decide "is this the same person".
"""

import dlib
from pathlib import Path

# Project root is one level up from this file's parent (facecut/ -> project root)
script_dir = Path(__file__).parent.parent.resolve()
models_dir = script_dir / "models"

shape_predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = models_dir / "dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(shape_predictor_path))
face_rec_model = dlib.face_recognition_model_v1(str(face_rec_model_path))

