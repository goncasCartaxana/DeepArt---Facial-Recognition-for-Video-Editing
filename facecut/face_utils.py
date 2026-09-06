"""
face_utils.py
--------------
Part of the FaceCut project (facecut package).

The "face math" layer: turning images into embeddings and comparing them.
Contains no video, threading, or GUI code — only dlib/cv2/numpy operations
on single images. That isolation is deliberate: this is the module you'd
write unit tests against (e.g. with a couple of sample face photos and
known-distance assertions), without needing a video file or a display.

Major functions:
- get_face_embedding(image, face_location):
    Computes a 128-dimensional embedding vector for one detected face
    (given its bounding box). This embedding is the core representation
    the rest of the app relies on to answer "is this the same person" -
    both for the one-time reference photo and for every face found in the
    video. Handles greyscale input by converting to 3-channel color,
    since the embedding model expects a color image.

- compare_faces(known_embedding, candidate_embedding, tolerance=0.6):
    Compares two embeddings via Euclidean distance and returns whether
    they're likely the same person. `tolerance` is the main knob for
    match strictness: lower = stricter match, higher = more permissive.
    This is the single place that threshold logic lives, so it's easy to
    later expose it as a user-configurable setting in the GUI.

- load_reference_image(image_path):
    Loads the user-selected reference photo, ensures it contains exactly
    one face, and returns that face's embedding. Raises ValueError if the
    image can't be read, or if zero or multiple faces are found - the
    tool needs one unambiguous reference face to search for.
"""

import cv2
import numpy as np

from .models import detector, predictor, face_rec_model


def get_face_embedding(image, face_location):
    shape = predictor(image, face_location)

    # If image is greyscale (height, width), convert to 3 channels
    # (height, width, channels), since the recognition model expects a
    # 3-channel color image.
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)


def compare_faces(known_embedding, candidate_embedding, tolerance=0.6):
    distance = np.linalg.norm(known_embedding - candidate_embedding)
    return distance <= tolerance


def load_reference_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        # cv2.imread fails silently (returns None) on a bad path or
        # unreadable file; surface that clearly instead of letting the
        # detector crash on a None image with a cryptic error.
        raise ValueError(f"Could not read image file: {image_path}")

    faces = detector(image)
    if len(faces) != 1:
        raise ValueError("Reference image must contain exactly one face.")

    return get_face_embedding(image, faces[0])