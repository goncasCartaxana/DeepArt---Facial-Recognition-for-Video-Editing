"""
export.py
----------
Part of the FaceCut project (facecut package).

Turns FaceCut's detection results (a list of (start_time, end_time) tuples,
in seconds) into files a video editor can actually import, instead of a
messagebox the user has to read and re-type by hand. This is the "point"
of the tool for its intended audience: getting detected face intervals
into Premiere, DaVinci Resolve, Final Cut, or Avid with a few clicks.

Deliberately UI-agnostic, like video_processing.py - no tkinter, no file
dialogs. gui.py is responsible for asking the user where to save; this
module only knows how to format and write once a path is given.

Major functions:
- format_timecode(total_seconds, fps):
    Converts a plain seconds value into the HH:MM:SS:FF timecode format
    editors expect for in/out points, using the video's own fps to
    compute the frame component. This is the building block both export
    formats below rely on for anything frame-accurate.

- export_intervals_to_csv(time_intervals, output_path):
    Writes a simple CSV with start/end/duration in seconds. The most
    universal option - importable into a spreadsheet, or into any editor
    or script that accepts plain timecodes/seconds, without needing to
    know the video's fps.

- export_intervals_to_edl(time_intervals, output_path, fps, title):
    Writes a CMX3600-format EDL (Edit Decision List), the long-standing
    industry-standard format for exchanging cut lists between different
    NLEs. Each detected interval becomes one video (V) cut event, with
    matching source/record timecodes since FaceCut only marks existing
    footage rather than repositioning it. This is the format most
    directly useful for a "rough-cut" workflow: importing it creates a
    sequence of clips wherever the reference face was found.

- export_intervals_to_otio(time_intervals, output_path, fps, video_path, title):
    Writes an OpenTimelineIO (.otio) file: a single clip referencing the
    original video, with one colored Marker placed at each detected
    interval. Unlike the EDL export - which fabricates a sequential cut
    list out of separate events - this keeps the original footage as one
    continuous clip and just flags the relevant moments on it, which is
    a more direct fit for "here's where the face shows up in this video"
    than pretending to reassemble a new sequence. OTIO is also the format
    with the widest and most actively maintained cross-NLE support today
    (native import in Kdenlive 25.04+, and convertible via `otioconvert`
    to Final Cut XML, AAF, and CMX3600 EDL for tools without native OTIO
    support).
"""

import csv
from pathlib import Path

import opentimelineio as otio


def format_timecode(total_seconds, fps):
    """
    Convert a time in seconds into an HH:MM:SS:FF timecode string.

    Parameters:
        total_seconds (float): Time in seconds to convert.
        fps (float): Frames per second of the source video, used to
            compute the frame component (FF) and to round seconds down
            to whole frames.

    Returns:
        str: Timecode formatted as "HH:MM:SS:FF".
    """
    fps_int = int(round(fps))
    total_frames = int(round(total_seconds * fps))

    frames = total_frames % fps_int
    total_whole_seconds = total_frames // fps_int
    seconds = total_whole_seconds % 60
    minutes = (total_whole_seconds // 60) % 60
    hours = total_whole_seconds // 3600

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def export_intervals_to_csv(time_intervals, output_path):
    """
    Write detected intervals to a CSV file (start/end/duration in seconds).

    Parameters:
        time_intervals (list): List of (start_time, end_time) tuples, in seconds.
        output_path (str): Path to write the .csv file to.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_seconds", "end_seconds", "duration_seconds"])
        for start, end in time_intervals:
            writer.writerow([f"{start:.3f}", f"{end:.3f}", f"{end - start:.3f}"])


def export_intervals_to_edl(time_intervals, output_path, fps, title="FaceCut Export"):
    """
    Write detected intervals to a CMX3600-format EDL file.

    Each interval becomes one video cut event. Source and record timecodes
    are identical for every event, since FaceCut is marking moments in the
    original footage rather than reassembling it into a new timeline order.

    Parameters:
        time_intervals (list): List of (start_time, end_time) tuples, in seconds.
        output_path (str): Path to write the .edl file to.
        fps (float): Frames per second of the source video (required for
            frame-accurate timecodes).
        title (str): Title recorded in the EDL header.
    """
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]

    for event_number, (start, end) in enumerate(time_intervals, start=1):
        start_tc = format_timecode(start, fps)
        end_tc = format_timecode(end, fps)
        reel = "001"  # Placeholder reel/tape name; FaceCut doesn't track reel identity.
        lines.append(
            f"{event_number:03d}  {reel}      V     C        "
            f"{start_tc} {end_tc} {start_tc} {end_tc}"
        )
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def export_intervals_to_otio(time_intervals, output_path, fps, video_path,
                              title="FaceCut Export"):
    """
    Write detected intervals to an OpenTimelineIO (.otio) file.

    Builds a single Timeline containing one video Track, holding one Clip
    that references the original video file end-to-end, with a Marker
    placed at each detected interval. This preserves the footage as one
    continuous clip - matching how the video actually exists - rather
    than fabricating a cut-together sequence the way EDL export does.

    Parameters:
        time_intervals (list): List of (start_time, end_time) tuples, in seconds.
        output_path (str): Path to write the .otio file to.
        fps (float): Frames per second of the source video.
        video_path (str): Path to the original video file, used to build
            the clip's media reference so the editor can load the actual
            footage, not just an empty timeline.
        title (str): Name recorded on the resulting Timeline.
    """
    resolved_video_path = Path(video_path).resolve()
    last_end_seconds = max((end for _, end in time_intervals), default=0.0)

    clip_range = otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, fps),
        duration=otio.opentime.RationalTime(last_end_seconds * fps, fps),
    )

    media_reference = otio.schema.ExternalReference(
        target_url=resolved_video_path.as_uri(),
        available_range=clip_range,
    )

    clip = otio.schema.Clip(
        name=resolved_video_path.stem,
        media_reference=media_reference,
        source_range=clip_range,
    )

    for event_number, (start, end) in enumerate(time_intervals, start=1):
        marker_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(start * fps, fps),
            duration=otio.opentime.RationalTime((end - start) * fps, fps),
        )
        clip.markers.append(otio.schema.Marker(
            name=f"Face detected #{event_number}",
            marked_range=marker_range,
            color=otio.schema.MarkerColor.GREEN,
        ))

    track = otio.schema.Track(name="Video", kind=otio.schema.TrackKind.Video)
    track.append(clip)

    timeline = otio.schema.Timeline(name=title)
    timeline.tracks.append(track)

    otio.adapters.write_to_file(timeline, str(output_path))