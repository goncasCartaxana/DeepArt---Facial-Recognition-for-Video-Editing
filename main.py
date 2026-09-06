"""
main.py
--------
Entry point for FaceCut.

Its only job is to launch the GUI. 

All actual logic lives inside the facecut/ package, split into multiple python code files.

Like so:
Model loading: models.py
Face-embedding math: face_utils.py
Video scanning: video_processing.py
Tkinter interface: gui.py.

Run with: python main.py
"""

from facecut.gui import start_tkinter_GUI


def main():
    start_tkinter_GUI()


if __name__ == "__main__":
    main()