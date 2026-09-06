
"""
gui.py
-------
Part of the FaceCut project (facecut package).

All Tkinter code lives here, and only here. This is the only module that
imports tkinter/filedialog/messagebox/ttk, and the only module that knows
how detection results (or errors) should be presented to the user. It calls
into video_processing.run_face_detection for the actual work; it never
implements detection logic itself.

Major functions:
- start_tkinter_GUI():
    Builds and runs the whole Tkinter window: file pickers for the
    reference image and video, a "Start Detection" button, and a progress
    bar. This is the module's single public entry point, called by
    main.py at the project root. It defines the following nested
    handlers:

    - select_reference_image() / select_video():
        Open native file dialogs and store the chosen paths in the
        Tkinter StringVars bound to the entry fields.

    - update_progress(current, total):
        Callback passed down into the detection pipeline; updates the
        progress bar. Called from the background thread, so it only
        touches simple widget properties (safe in practice with ttk's
        Progressbar, though heavier UI updates from a worker thread
        would need root.after() to be fully safe).

    - _run_detection_and_report(reference_image_path, video_path, frame_skip,
      frame_skip_tolerance):
        The background-thread target. Wraps
        video_processing.run_face_detection in a try/except, formats the
        resulting time intervals into a message box, and shows an error
        dialog on failure. This function is where the "pure core, UI
        shell" boundary from the refactor is enforced: run_face_detection
        never touches messagebox itself, only this function does.

    - start_detection():
        Validates that both a reference image and a video have been
        selected, reads the frame_skip and frame_skip_tolerance values
        from their Spinbox widgets, resets the progress bar, and starts
        _run_detection_and_report on a background thread so the GUI
        doesn't freeze during a long scan.

Two Spinbox controls expose the scan's speed/thoroughness tradeoff directly
to the user, rather than relying on video_processing.py's hardcoded
defaults:
    - frame_skip_var ("Frame Skip / Efficiency"): how many frames to skip
      between processed frames. Higher = faster scan, but coarser detection.
    - frame_skip_tolerance_var ("Skip Tolerance"): how many consecutive
      missed detections are allowed before an open interval is closed.
      Higher = more forgiving of brief misses (blinks, quick turns), but
      can merge separate appearances together if set too high.
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .video_processing import run_face_detection


def start_tkinter_GUI():
    def select_reference_image():
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        reference_image_var.set(path)

    def select_video():
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        video_path_var.set(path)

    def update_progress(current, total):
        if total > 0:  # Ensure total is not zero
            progress_bar['maximum'] = total
            progress_bar['value'] = current

    def _run_detection_and_report(reference_image_path, video_path,
                                   frame_skip, frame_skip_tolerance):
        try:
            time_intervals, _ = run_face_detection(
                reference_image_path, video_path, update_progress,
                frame_skip=frame_skip, frame_skip_tolerance=frame_skip_tolerance)

            output = "Time Intervals (in seconds):"
            for start, end in time_intervals:
                output += f"\nStart: {start:.2f}s, End: {end:.2f}s"
            messagebox.showinfo("Detection Results", output)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_detection():
        reference_image_path = reference_image_var.get()
        video_path = video_path_var.get()

        if not reference_image_path or not video_path:
            messagebox.showwarning("Input Error", "Please select both reference image and video file.")
            return

        try:
            frame_skip = int(frame_skip_var.get())
            frame_skip_tolerance = int(frame_skip_tolerance_var.get())
            if frame_skip < 0 or frame_skip_tolerance < 0:
                raise ValueError
        except (tk.TclError, ValueError):
            messagebox.showwarning(
                "Input Error",
                "Frame Skip and Skip Tolerance must be whole numbers of 0 or higher.")
            return

        # Prepare the progress bar for a new detection run
        progress_bar['value'] = 0  # Reset value

        thread = threading.Thread(
            target=_run_detection_and_report,
            args=(reference_image_path, video_path, frame_skip, frame_skip_tolerance))
        thread.start()

    # Set up Tkinter GUI
    root = tk.Tk()
    root.title("Face Detection App")

    # Variables
    reference_image_var = tk.StringVar()
    video_path_var = tk.StringVar()
    # Defaults match video_processing.detect_face_in_video's own defaults,
    # so leaving these untouched preserves the previous behavior exactly.
    frame_skip_var = tk.StringVar(value="3")
    frame_skip_tolerance_var = tk.StringVar(value="3")

    # UI Elements

    ## Reference image
    tk.Label(root, text="Select Reference Image:").pack(pady=5)
    tk.Entry(root, textvariable=reference_image_var, width=50).pack(pady=5)
    tk.Button(root, text="Browse", command=select_reference_image).pack(pady=5)

    ## Video
    tk.Label(root, text="Select Video File:").pack(pady=5)
    tk.Entry(root, textvariable=video_path_var, width=50).pack(pady=5)
    tk.Button(root, text="Browse", command=select_video).pack(pady=5)

    ## Detection parameters
    params_frame = tk.Frame(root)
    params_frame.pack(pady=10)

    tk.Label(params_frame, text="Frame Skip (efficiency):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
    tk.Spinbox(params_frame, from_=0, to=30, textvariable=frame_skip_var, width=5).grid(row=0, column=1, padx=5, pady=5)

    tk.Label(params_frame, text="Skip Tolerance (leniency):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
    tk.Spinbox(params_frame, from_=0, to=30, textvariable=frame_skip_tolerance_var, width=5).grid(row=1, column=1, padx=5, pady=5)

    # Run Detection
    tk.Button(root, text="Start Detection", command=start_detection).pack(pady=20)

    # Progress bar
    progress_bar = ttk.Progressbar(root, length=300, mode='determinate')
    progress_bar.pack(pady=10)

    # Loop
    root.mainloop()