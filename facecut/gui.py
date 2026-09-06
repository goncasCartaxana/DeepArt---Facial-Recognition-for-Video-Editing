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
      frame_skip_tolerance, cancel_event):
        The background-thread target. Wraps
        video_processing.run_face_detection in a try/except, formats the
        resulting time intervals into a message box, and shows an error
        dialog on failure. Passes cancel_event through so the scan can be
        stopped early; if it was cancelled, the message box reports a
        partial result instead of a completed one. On success (complete
        or cancelled), also stores the results (and the video's fps) and
        re-enables the export buttons and disables Cancel via root.after,
        since widget updates must happen on the main thread. This function
        is where the "pure core, UI shell" boundary from the refactor is
        enforced: run_face_detection never touches messagebox itself, only
        this function does.

    - start_detection():
        Validates that both a reference image and a video have been
        selected, reads the frame_skip and frame_skip_tolerance values
        from their Spinbox widgets, resets the progress bar and disables
        the export buttons (since they'd refer to a stale prior run),
        creates a fresh threading.Event for this run and enables the
        Cancel button, and starts _run_detection_and_report on a
        background thread so the GUI doesn't freeze during a long scan.

    - cancel_detection():
        Sets the current run's cancel_event and disables the Cancel
        button immediately, so a double-click can't queue a second
        cancellation. The scan itself notices the event on its next
        per-frame check (see video_processing.detect_face_in_video) and
        winds down from there - this handler doesn't stop anything by
        itself, it only signals the intent to stop.

    - export_csv() / export_edl() / export_otio():
        Prompt for a save location via asksaveasfilename, then call
        export.export_intervals_to_csv / export.export_intervals_to_edl /
        export.export_intervals_to_otio on the most recently stored
        detection results. Disabled (via button state) until a scan has
        completed successfully, so they can't be triggered with no
        results to export.

Two Spinbox controls expose the scan's speed/thoroughness tradeoff directly
to the user, rather than relying on video_processing.py's hardcoded
defaults:
    - frame_skip_var ("Frame Skip / Efficiency"): how many frames to skip
      between processed frames. Higher = faster scan, but coarser detection.
    - frame_skip_tolerance_var ("Skip Tolerance"): how many consecutive
      missed detections are allowed before an open interval is closed.
      Higher = more forgiving of brief misses (blinks, quick turns), but
      can merge separate appearances together if set too high.

Export buttons let the user save the most recent scan's results as a CSV
(plain seconds, universal), a CMX3600 EDL (frame-accurate, importable
directly into Premiere/Resolve/Final Cut/Avid as a cut list), or an
OpenTimelineIO (.otio) file (a single clip referencing the original video
with a marker at each detected interval - natively importable in Kdenlive
25.04+, and convertible to other formats via `otioconvert`) - see export.py
for the actual formatting logic, which this module never duplicates.
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .video_processing import run_face_detection
from .export import export_intervals_to_csv, export_intervals_to_edl, export_intervals_to_otio


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

    # Holds the results of the most recent successful scan, so the export
    # buttons can act on them. A plain dict (rather than separate variables)
    # since it's mutated from the worker thread and read from the main thread.
    last_results = {"time_intervals": None, "fps": None, "video_path": None}
    # Holds the threading.Event for whichever scan is currently running (or
    # None if none is running), so cancel_detection() can reach it.
    current_run = {"cancel_event": None}

    def _run_detection_and_report(reference_image_path, video_path,
                                   frame_skip, frame_skip_tolerance, cancel_event):
        try:
            time_intervals, _, fps, cancelled = run_face_detection(
                reference_image_path, video_path, update_progress,
                frame_skip=frame_skip, frame_skip_tolerance=frame_skip_tolerance,
                cancel_event=cancel_event)

            last_results["time_intervals"] = time_intervals
            last_results["fps"] = fps
            last_results["video_path"] = video_path
            current_run["cancel_event"] = None
            # Widget state must be touched on the main thread; root.after
            # schedules this safely instead of touching widgets directly
            # from this background thread.
            root.after(0, lambda: (
                export_csv_button.config(state=tk.NORMAL),
                export_edl_button.config(state=tk.NORMAL),
                export_otio_button.config(state=tk.NORMAL),
                cancel_button.config(state=tk.DISABLED),
            ))

            if cancelled:
                output = "Scan cancelled. Partial results (in seconds):"
            else:
                output = "Time Intervals (in seconds):"
            for start, end in time_intervals:
                output += f"\nStart: {start:.2f}s, End: {end:.2f}s"
            messagebox.showinfo(
                "Detection Cancelled" if cancelled else "Detection Results", output)
        except Exception as e:
            current_run["cancel_event"] = None
            root.after(0, lambda: cancel_button.config(state=tk.DISABLED))
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

        # Prepare the progress bar for a new detection run, and disable the
        # export buttons since they'd otherwise still point at the previous
        # run's results until this new one finishes.
        progress_bar['value'] = 0  # Reset value
        export_csv_button.config(state=tk.DISABLED)
        export_edl_button.config(state=tk.DISABLED)
        export_otio_button.config(state=tk.DISABLED)

        cancel_event = threading.Event()
        current_run["cancel_event"] = cancel_event
        cancel_button.config(state=tk.NORMAL)

        thread = threading.Thread(
            target=_run_detection_and_report,
            args=(reference_image_path, video_path, frame_skip, frame_skip_tolerance,
                  cancel_event))
        thread.start()

    def cancel_detection():
        cancel_event = current_run["cancel_event"]
        if cancel_event is not None:
            cancel_event.set()
        # Disable immediately so a second click can't queue another
        # cancellation; the running scan notices the event on its own and
        # _run_detection_and_report re-disables this once it actually stops.
        cancel_button.config(state=tk.DISABLED)

    def export_csv():
        if last_results["time_intervals"] is None:
            messagebox.showwarning("No Results", "Run a detection scan first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        try:
            export_intervals_to_csv(last_results["time_intervals"], path)
            messagebox.showinfo("Export Complete", f"Saved CSV to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_edl():
        if last_results["time_intervals"] is None:
            messagebox.showwarning("No Results", "Run a detection scan first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".edl", filetypes=[("EDL Files", "*.edl")])
        if not path:
            return
        try:
            export_intervals_to_edl(
                last_results["time_intervals"], path, last_results["fps"])
            messagebox.showinfo("Export Complete", f"Saved EDL to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_otio():
        if last_results["time_intervals"] is None:
            messagebox.showwarning("No Results", "Run a detection scan first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".otio", filetypes=[("OpenTimelineIO Files", "*.otio")])
        if not path:
            return
        try:
            export_intervals_to_otio(
                last_results["time_intervals"], path, last_results["fps"],
                last_results["video_path"])
            messagebox.showinfo("Export Complete", f"Saved OTIO to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

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

    # Run / Cancel Detection
    run_frame = tk.Frame(root)
    run_frame.pack(pady=20)

    tk.Button(run_frame, text="Start Detection", command=start_detection).grid(row=0, column=0, padx=5)

    cancel_button = tk.Button(
        run_frame, text="Cancel", command=cancel_detection, state=tk.DISABLED)
    cancel_button.grid(row=0, column=1, padx=5)

    # Progress bar
    progress_bar = ttk.Progressbar(root, length=300, mode='determinate')
    progress_bar.pack(pady=10)

    # Export buttons — disabled until a scan completes successfully
    export_frame = tk.Frame(root)
    export_frame.pack(pady=10)

    export_csv_button = tk.Button(
        export_frame, text="Export CSV", command=export_csv, state=tk.DISABLED)
    export_csv_button.grid(row=0, column=0, padx=5)

    export_edl_button = tk.Button(
        export_frame, text="Export EDL", command=export_edl, state=tk.DISABLED)
    export_edl_button.grid(row=0, column=1, padx=5)

    export_otio_button = tk.Button(
        export_frame, text="Export OTIO", command=export_otio, state=tk.DISABLED)
    export_otio_button.grid(row=0, column=2, padx=5)

    # Loop
    root.mainloop()