"""
video_processing.py
---------------------
Part of the FaceCut project (facecut package).

The core detection / business-logic layer: scans a video frame-by-frame
looking for a reference face, and turns raw per-frame detections into clean
time intervals. Deliberately UI-agnostic - no Tkinter, no messagebox, no
file dialogs - so it can be reused by a future CLI/batch mode and unit
tested independently of the GUI. gui.py is the only module allowed to know
how results get shown to a user; this module only computes them.

Major functions:
- open_video(video_path):
    Opens a video file and returns the capture object plus its fps and
    frame_count. These two properties drive the main scanning loop below
    and the conversion from frame index to a timestamp in seconds.

- process_frame(frame, resize_factor):
    Resizes a frame and produces a grayscale copy. Resizing trades a bit
    of detection accuracy for a large speed gain, since face detection
    cost scales with image size; grayscale is what the face detector
    itself operates on.

- detect_reference_face_in_frame(frame, gray, reference_embedding):
    Runs the face detector on a frame and checks whether any face found
    matches the reference embedding. Returns as soon as a match is found
    ("does the reference face appear anywhere in this frame", not "list
    every face"), which is all the interval logic below needs.

- update_intervals(...):
    Stateless-per-call bookkeeping step that tracks when a "face present"
    interval starts and ends, tolerating brief detection gaps
    (frame_skip_tolerance) so a few missed frames - due to a blink, a
    quick turn of the head, motion blur - don't fragment one continuous
    appearance into many separate intervals.

- detect_face_in_video(...):
    Orchestrates the full scan of a video: loops over frames, applies
    frame_skip to only process every Nth frame for speed, calls the
    helpers above, and returns both a binary per-frame detection array
    and the final list of (start_time, end_time) intervals.

- run_face_detection(reference_image_path, video_path, progress_callback):
    The single top-level entry point for a full detection run. Loads the
    reference embedding, runs detect_face_in_video, and returns
    (time_intervals, binary_detection_array) - or raises an exception on
    failure (bad reference image, unreadable video, etc). This function
    used to also show a messagebox with the results; it no longer does -
    that responsibility now lives entirely in gui.py. Keeping this
    function "pure" (return values or exceptions, no UI side effects) is
    what makes it reusable from a future CLI/batch mode, and testable
    without a display.
"""

import cv2
import numpy as np

from .face_utils import get_face_embedding, compare_faces, load_reference_image
from .models import detector


def open_video(video_path):
    """
    Open a video file and retrieve its properties.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        video (cv2.VideoCapture): Opened video capture object.
        fps (float): Frames per second of the video.
        frame_count (int): Total number of frames in the video.
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return video, fps, frame_count


def process_frame(frame, resize_factor):
    """
    Resize and convert a video frame to grayscale.

    Parameters:
        frame (ndarray): Original color frame.
        resize_factor (float): Factor to resize frame dimensions.

    Returns:
        frame (ndarray): Resized original frame.
        gray (ndarray): Resized grayscale version of the frame.
    """
    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray


def detect_reference_face_in_frame(frame, gray, reference_embedding):
    """
    Detect faces in a grayscale frame and check against a reference embedding.

    Parameters:
        frame (ndarray): Color frame (used for embedding computation).
        gray (ndarray): Grayscale image/frame (used for face detection).
        reference_embedding (ndarray): Known face embedding for comparison.

    Returns:
        bool: True if reference face is detected, False otherwise.
    """
    faces = detector(gray)
    for face in faces:
        face_embedding = get_face_embedding(frame, face)
        if compare_faces(reference_embedding, face_embedding):
            return True
    return False


def update_intervals(face_detected, frame_index, fps, current_interval_start,
                      frames_without_detection, frame_skip_tolerance, time_intervals,
                      last_detected_frame):
    """
    Track continuous intervals where a face is detected in video frames.

    Parameters:
        face_detected (bool): Whether face detected in current frame.
        frame_index (int): Index of current frame.
        fps (float): Video frames per second.
        current_interval_start (int or None): Start frame of current detection interval.
        frames_without_detection (int): Count of consecutive frames with no detection.
        frame_skip_tolerance (int): Max allowed skipped frames before closing interval.
        time_intervals (list): List of tuples recording (start_time, end_time) of intervals.
        last_detected_frame (int or None): Frame index where the face was last actually
            seen during the current interval. Used (rather than the current frame_index)
            to close an interval, so its reported end time reflects when the face was
            last really there, not how many tolerance frames were spent waiting to see
            if it would come back.

    Returns:
        tuple: Updated current_interval_start, frames_without_detection, last_detected_frame.
    """
    if face_detected:
        if current_interval_start is None:
            current_interval_start = frame_index
        frames_without_detection = 0
        last_detected_frame = frame_index
    else:
        if current_interval_start is not None:
            frames_without_detection += 1
            if frames_without_detection > frame_skip_tolerance:
                start_time = current_interval_start / fps
                end_time = last_detected_frame / fps
                time_intervals.append((start_time, end_time))
                current_interval_start = None
                frames_without_detection = 0
                last_detected_frame = None

    return current_interval_start, frames_without_detection, last_detected_frame


def detect_face_in_video(video_path, reference_embedding, frame_skip_tolerance=3,
                          frame_skip=3, resize_factor=0.5, progress_callback=None,
                          cancel_event=None):
    """
    Detect reference face appearances in a video and return detection info.

    Parameters:
        video_path (str): Path to the video file.
        reference_embedding (ndarray): Embedded vector of the reference face.
        frame_skip_tolerance (int): Number of allowed missed detections before interval ends.
        frame_skip (int): Number of frames to skip between processing.
        resize_factor (float): Factor to resize frames to improve speed.
        progress_callback (callable, optional): Function for reporting progress.
        cancel_event (threading.Event, optional): Checked once per frame; if set,
            the scan stops early. Any interval that was open at that point is
            closed using the last frame the face was actually seen in, exactly
            like a normal end-of-video close - so a cancelled scan still
            returns a valid, exportable partial result rather than a
            truncated/incorrect one.

    Returns:
        binary_detection_array (ndarray): Binary array marking frames with face detected.
        time_intervals (list): List of (start_time, end_time) tuples for detected face intervals.
        fps (float): Frames per second of the source video. Returned alongside the
            results because EDL export needs it to compute frame-accurate timecodes.
        cancelled (bool): True if the scan stopped early because cancel_event was
            set, rather than reaching the end of the video normally.
    """
    video, fps, frame_count = open_video(video_path)
    binary_detection_array = np.zeros(frame_count, dtype=int)
    time_intervals = []
    current_interval_start = None
    frames_without_detection = 0
    last_detected_frame = None
    cancelled = False

    for frame_index in range(frame_count):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        ret, frame = video.read()
        if not ret:
            break

        if frame_index % (frame_skip + 1) == 0:
            frame, gray = process_frame(frame, resize_factor)
            face_detected = detect_reference_face_in_frame(frame, gray, reference_embedding)
            binary_detection_array[frame_index] = 1 if face_detected else 0

            current_interval_start, frames_without_detection, last_detected_frame = update_intervals(
                face_detected, frame_index, fps, current_interval_start,
                frames_without_detection, frame_skip_tolerance, time_intervals,
                last_detected_frame)

        if progress_callback:
            progress_callback(frame_index + 1, frame_count)

    if current_interval_start is not None:
        start_time = current_interval_start / fps
        end_time = last_detected_frame / fps
        time_intervals.append((start_time, end_time))

    video.release()
    return binary_detection_array, time_intervals, fps, cancelled


def run_face_detection(reference_image_path, video_path, progress_callback=None,
                        frame_skip=3, frame_skip_tolerance=3, cancel_event=None):
    """
    Run a full detection pass: load the reference face, scan the video, and
    return the results. Raises on failure (e.g. bad reference image, missing
    or unreadable video) instead of handling the error itself - the caller
    (gui.py, or a future CLI) decides how to report it.

    Parameters:
        reference_image_path (str): Path to the reference face image.
        video_path (str): Path to the video file to scan.
        progress_callback (callable, optional): Function for reporting progress.
        frame_skip (int): Number of frames to skip between processed frames.
            Higher = faster scan, lower detection resolution ("efficiency" knob).
        frame_skip_tolerance (int): Consecutive missed detections allowed
            before an interval is closed. Higher = more forgiving of brief
            misses ("tolerance" knob).
        cancel_event (threading.Event, optional): Forwarded to detect_face_in_video;
            allows a caller running this on a background thread to request an
            early stop.

    Returns:
        time_intervals (list): List of (start_time, end_time) tuples.
        binary_detection_array (ndarray): Binary array marking frames with face detected.
        fps (float): Frames per second of the source video (needed for EDL export).
        cancelled (bool): True if the scan was stopped early via cancel_event.
    """
    reference_embedding = load_reference_image(reference_image_path)
    binary_detection_array, time_intervals, fps, cancelled = detect_face_in_video(
        video_path, reference_embedding,
        frame_skip=frame_skip,
        frame_skip_tolerance=frame_skip_tolerance,
        progress_callback=progress_callback,
        cancel_event=cancel_event)
    return time_intervals, binary_detection_array, fps, cancelled