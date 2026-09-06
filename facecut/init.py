"""
facecut
--------
The FaceCut application package.

Groups the project's four concerns into separate modules so they can be
developed, tested, and reused independently:

- models.py            -> loads the dlib models (shared, single source of truth)
- face_utils.py         -> pure face-embedding / comparison math
- video_processing.py   -> frame-by-frame scanning + interval bookkeeping (UI-agnostic)
- gui.py                -> Tkinter interface, the only place UI code lives

main.py (at the project root) is the only thing outside this package, and
its only job is to call facecut.gui.start_tkinter_GUI().
"""
