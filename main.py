
import cv2
import dlib
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path


# =============== Folders =============== #
script_dir = Path(__file__).parent.resolve() # Python file's directory
models_dir = script_dir / "models" # Build "models" folder Path
shape_predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = models_dir / "dlib_face_recognition_resnet_model_v1.dat"

# =============== Load Models  =============== # 
detector = dlib.get_frontal_face_detector() # Detect faces, create bounding boxes for each face
predictor = dlib.shape_predictor(str(shape_predictor_path)) # Predict the location of specific facial landmarks
face_rec_model = dlib.face_recognition_model_v1(str(face_rec_model_path)) # Generates face embedding for each face. Embeddings enable comparison.



# =============== Image and Embedding Utilities =============== #

def get_face_embedding(image, face_location):
    shape = predictor(image, face_location)
    if len(image.shape) == 2: # If image is greyscale (height, width), convert to 3d to add color channels (height, width, channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape) # face recognition model expects a 3-channel color image
    return np.array(face_descriptor)    


def compare_faces(known_embedding, candidate_embedding, tolerance=0.6):
    distance = np.linalg.norm(known_embedding - candidate_embedding)
    return distance <= tolerance


def load_reference_image(image_path):
    image = cv2.imread(image_path)
    faces = detector(image)
    if len(faces) != 1:
        raise ValueError("Reference image must contain exactly one face.")
    return get_face_embedding(image, faces[0])



# =============== Core Functionality =============== #

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
        gray (ndarray): Grayscale image/frame.
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


def update_intervals(face_detected, frame_index, fps, current_interval_start, frames_without_detection, frame_skip_tolerance, time_intervals):
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

    Returns:
        tuple: Updated current_interval_start and frames_without_detection.
    """

    if face_detected:
        if current_interval_start is None:
            current_interval_start = frame_index
        frames_without_detection = 0
    else:
        if current_interval_start is not None:
            frames_without_detection += 1
            if frames_without_detection > frame_skip_tolerance:
                start_time = current_interval_start / fps
                end_time = frame_index / fps
                time_intervals.append((start_time, end_time))
                current_interval_start = None
                frames_without_detection = 0
    return current_interval_start, frames_without_detection


def detect_face_in_video(video_path, reference_embedding, frame_skip_tolerance=3, frame_skip=3, resize_factor=0.5, progress_callback=None):
    """
    Detect reference face appearances in a video and return detection info.

    Parameters:
        video_path (str): Path to the video file.
        reference_embedding (ndarray): Embedded vector of the reference face.
        frame_skip_tolerance (int): Number of allowed missed detections before interval ends.
        frame_skip (int): Number of frames to skip between processing.
        resize_factor (float): Factor to resize frames to improve speed.
        progress_callback (callable, optional): Function for reporting progress.

    Returns:
        binary_detection_array (ndarray): Binary array marking frames with face detected.
        time_intervals (list): List of (start_time, end_time) tuples for detected face intervals.
    """
    video, fps, frame_count = open_video(video_path)
    binary_detection_array = np.zeros(frame_count, dtype=int)
    time_intervals = []

    current_interval_start = None
    frames_without_detection = 0

    for frame_index in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        if frame_index % (frame_skip + 1) == 0:
            frame, gray = process_frame(frame, resize_factor)
            face_detected = detect_reference_face_in_frame(frame, gray, reference_embedding)
            binary_detection_array[frame_index] = 1 if face_detected else 0
            current_interval_start, frames_without_detection = update_intervals(
                face_detected, frame_index, fps, current_interval_start, frames_without_detection, frame_skip_tolerance, time_intervals)

        if progress_callback:
            progress_callback(frame_index + 1, frame_count)

    if current_interval_start is not None:
        start_time = current_interval_start / fps
        end_time = frame_index / fps
        time_intervals.append((start_time, end_time))

    video.release()
    return binary_detection_array, time_intervals


def run_face_detection(reference_image_path, video_path, progress_callback):
    try:
        reference_embedding = load_reference_image(reference_image_path)
        binary_detection_array, time_intervals = detect_face_in_video(video_path, reference_embedding, progress_callback=progress_callback)

        output = f"Time Intervals (in seconds):"
        for start, end in time_intervals:
            output += f"\nStart: {start:.2f}s, End: {end:.2f}s"
        
        messagebox.showinfo("Detection Results", output)
    except Exception as e:
        messagebox.showerror("Error", str(e))


# =============== Progress Reporting =============== #
def update_progress(current, total):
    pass

# =============== App / User Interaction =============== #

def start_tkinter_GUI():
        
    def select_reference_image():
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        reference_image_var.set(path)

    def select_video():
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        video_path_var.set(path)

    def start_detection():
        reference_image_path = reference_image_var.get()
        video_path = video_path_var.get()
        
        if not reference_image_path or not video_path:
            messagebox.showwarning("Input Error", "Please select both reference image and video file.")
            return

        # Prepare the progress bar for a new detection run
        progress_bar['value'] = 0  # Reset value
        thread = threading.Thread(target=run_face_detection, args=(reference_image_path, video_path, update_progress))
        thread.start()

    def update_progress(current, total):
        if total > 0:  # Ensure total is not zero
            progress_bar['maximum'] = total
            progress_bar['value'] = current

    # Set up Tkinter GUI
    root = tk.Tk()
    root.title("Face Detection App")

    # Variables
    reference_image_var = tk.StringVar()
    video_path_var = tk.StringVar()

    # UI Elements
    ## Image
    tk.Label(root, text="Select Reference Image:").pack(pady=5)
    tk.Entry(root, textvariable=reference_image_var, width=50).pack(pady=5)
    tk.Button(root, text="Browse", command=select_reference_image).pack(pady=5)
    
    ## Image
    tk.Label(root, text="Select Video File:").pack(pady=5)
    tk.Entry(root, textvariable=video_path_var, width=50).pack(pady=5)
    tk.Button(root, text="Browse", command=select_video).pack(pady=5)
    # Run Detection
    tk.Button(root, text="Start Detection", command=start_detection).pack(pady=20)

    # Progress bar
    progress_bar = ttk.Progressbar(root, length=300, mode='determinate')
    progress_bar.pack(pady=10)
    
    # Loop
    root.mainloop()


def main():
    start_tkinter_GUI()


if __name__ == "__main__":
    main()
