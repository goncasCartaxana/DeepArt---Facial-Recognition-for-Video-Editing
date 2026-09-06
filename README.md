# FaceCut

FaceCut is a Tkinter-powered Python GUI utility designed to automate video logging by detecting facial appearance time intervals.

Built for video editors, the tool scans long-form footage to extract time intervals containing a given human face, streamlining the rough-cut editing and review workflow. Detected intervals can be exported directly as CSV, a CMX3600 EDL, or an OpenTimelineIO (`.otio`) file for use in your editor of choice.

## Features

- **Reference-face matching**: select one photo of a person, scan a video, and get back every time interval where that face appears.
- **Adjustable scan parameters**: two GUI controls let you tune the speed/thoroughness tradeoff without touching code:
  - **Frame Skip (efficiency)** — how many frames to skip between checks. Higher = faster scan, coarser detection.
  - **Skip Tolerance (leniency)** — how many consecutive missed detections are tolerated before an interval is considered ended. Higher = more forgiving of brief misses (blinks, quick head turns), at the risk of merging separate appearances if set too high.
- **Cancel mid-scan**: a running scan can be stopped early; you still get a valid, exportable partial result rather than nothing.
- **Export formats**:
  - **CSV** — plain start/end/duration in seconds. Universal, no fps required.
  - **EDL (CMX3600)** — frame-accurate cut list, importable into Premiere, DaVinci Resolve, Final Cut Pro, and Avid.
  - **OTIO (OpenTimelineIO)** — a single clip referencing your original video with a marker at each detected interval. Natively importable in Kdenlive 25.04+, and convertible to other formats via `otioconvert`.

## Project Structure

```
FaceCut/
├── main.py                   # entry point — run this
├── facecut/                  # application package
│   ├── models.py              # loads the dlib models (single source of truth)
│   ├── face_utils.py          # face embedding + comparison (pure logic, no UI)
│   ├── video_processing.py    # frame-by-frame scanning + interval bookkeeping (UI-agnostic)
│   ├── export.py              # CSV / EDL / OTIO export (UI-agnostic)
│   └── gui.py                 # all Tkinter code lives here, and only here
├── models/                    # downloaded .dat model files (see below)
├── requirements.txt
├── setup_models.sh
└── README.md
```

Detection logic is intentionally kept separate from the GUI: `video_processing.py` and `export.py` know nothing about Tkinter, which is what makes them reusable (e.g. for a future CLI/batch mode) and independently testable.

## Models Used

**dlib Frontal Face Detector**
- Does: Detects faces in images and creates bounding boxes.
- Extra: Uses dlib's built-in HOG + Linear SVM face detection model. Best for frontal or near-frontal faces.

**Facial Landmark Model**
- Name: `shape_predictor_68_face_landmarks.dat`
- Does: Detects facial landmarks (eyes, nose, mouth, jawline).
- Link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

**Face Embedding Model**
- Name: `dlib_face_recognition_resnet_model_v1.dat`
- Does: Produces 128-dimensional numerical face embeddings, used for recognition/comparison.
- Extra: Deep learning face recognition model based on ResNet.
- Link: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

Models can also be found here: https://github.com/davisking/dlib-models

## Requirements

- **Python 3.13.x** (tested on 3.13.9 and 3.13.15)
- A C++ build toolchain + CMake + Boost — `dlib` doesn't always have a prebuilt wheel for every Python version/platform, and may need to compile from source. This is expected and can take several minutes on first install; it's not a hang.
- Tk (Tkinter) — a system package, not a pip package, so it must be installed separately and must match your Python version.

### OS-specific setup

**Windows**
- Install Python 3.13 (64-bit) from [python.org](https://www.python.org/)
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ workload) — needed if `dlib` compiles from source
- Install [CMake](https://cmake.org/download/) (or `pip install cmake`)

**Linux (Debian/Ubuntu)**
```bash
sudo apt-get install -y build-essential cmake libboost-all-dev python3-tk python3.13-dev
```

**Linux (Fedora/Nobara/RHEL)**
```bash
sudo dnf install -y gcc-c++ cmake boost-devel python3.13-tkinter python3.13-devel
```
If your distro's repos don't carry an exact `python3.13` package, use [pyenv](https://github.com/pyenv/pyenv) instead — but install `tk-devel` (`sudo dnf install tk-devel`) **before** running `pyenv install 3.13.x`, or the resulting Python will silently lack Tkinter support.

**macOS**
```bash
brew install cmake boost python-tk@3.13
```

## How to Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/goncasCartaxana/FaceCut.git
   cd FaceCut
   ```

2. **Download the models**
   ```bash
   bash setup_models.sh
   ```
   Or manually:
   ```bash
   mkdir -p models && cd models
   curl -L -o shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   curl -L -o dlib_face_recognition_resnet_model_v1.dat.bz2 http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
   bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -dk dlib_face_recognition_resnet_model_v1.dat.bz2
   cd ..
   ```

3. **Create and activate a virtual environment**

   Windows:
   ```powershell
   py -3.13 -m venv .venv
   .venv\Scripts\activate
   ```

   Linux / macOS:
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app**
   ```bash
   python main.py
   ```

6. In the GUI: select a reference image (a photo with exactly one face), select a video, adjust Frame Skip / Skip Tolerance if desired, and click **Start Detection**. Once it completes, use **Export CSV / EDL / OTIO** to save the results, or **Cancel** to stop a scan early.

To leave the virtual environment when you're done, run `deactivate`.

## Licenses

This project uses the dlib library and pre-trained models created by Davis E. King. dlib is licensed under the Boost Software License 1.0.
For more details, see https://dlib.net/license.html

The Boost Software License allows free use, modification, and distribution of dlib for both commercial and non-commercial purposes, provided the license text is included.
