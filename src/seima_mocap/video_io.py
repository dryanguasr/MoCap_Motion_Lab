"""Exact source-frame video I/O for reviewed clips (no FPS fallback)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
from collections import OrderedDict

import cv2
import numpy as np

_METADATA_CACHE = {}
_FRAME_CACHE = OrderedDict()
_TOOL_PATHS = {}


def _video_key(video):
    path = Path(video).resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def bind_verified_source_metadata(video, metadata):
    """Reuse the persisted probe only after the caller verifies its source hash."""
    _METADATA_CACHE[_video_key(video)] = metadata


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename short: appending to long artifact names can
    # exceed Windows path limits even when the destination itself is valid.
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_video_tool(name: str, explicit: str | None = None) -> str:
    candidate = explicit or _TOOL_PATHS.get(name) or os.environ.get(f"SEIMA_{name.upper()}") or shutil.which(name)
    if candidate:
        resolved = shutil.which(candidate) or (str(Path(candidate).resolve()) if Path(candidate).is_file() else None)
        if resolved:
            _TOOL_PATHS[name] = resolved
            return resolved
        raise FileNotFoundError(f"Invalid {name} executable: {candidate}")
    local = Path(__file__).resolve().parents[2] / ".cache/video-tools" / (name + (".exe" if os.name == "nt" else ""))
    if local.is_file():
        return str(local)
    root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"
    for package in sorted(root.glob("*FFmpeg*")):
        matches = sorted(package.rglob(f"{name}.exe"))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"{name} required. Set SEIMA_{name.upper()} or pass --{name}.")


def probe_video(video: Path, ffprobe: str | None = None) -> dict:
    key = _video_key(video)
    if key in _METADATA_CACHE:
        return _METADATA_CACHE[key]
    command = [resolve_video_tool("ffprobe", ffprobe), "-v", "error", "-select_streams", "v:0",
               "-show_frames", "-show_streams", "-show_entries",
               "frame=best_effort_timestamp_time,pkt_duration_time:stream=width,height,time_base",
               "-of", "json", str(video)]
    payload = json.loads(subprocess.check_output(command, text=True, encoding="utf-8"))
    frames = payload["frames"]
    pts = np.asarray([float(row["best_effort_timestamp_time"]) for row in frames])
    if len(pts) < 3 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("Video needs at least three decoded frames with strictly increasing PTS")
    stream = payload["streams"][0]
    duration = float(frames[-1].get("pkt_duration_time", np.median(np.diff(pts))))
    if duration <= 0:
        duration = float(np.median(np.diff(pts)))
    result = {"width": stream["width"], "height": stream["height"], "pts_s": pts.tolist(),
            "frame_count": len(pts), "end_pts_s": float(pts[-1] + duration),
            "time_base": stream["time_base"], "timestamp_source": "ffprobe_best_effort_timestamp_time"}
    _METADATA_CACHE[key] = result
    return result


def read_frame(video: Path, index: int) -> np.ndarray:
    key = (_video_key(video), index)
    if key not in _FRAME_CACHE:
        # OpenCV CAP_PROP_POS_FRAMES seeks via average FPS for this VFR source.
        # Its reported frame number can be correct while the image is wrong.
        stream = iter_frames(video, index, index+1)
        try:
            _, frame = next(stream)
        finally:
            stream.close()
        _FRAME_CACHE[key] = frame
        if len(_FRAME_CACHE) > 12:
            _FRAME_CACHE.popitem(last=False)
    _FRAME_CACHE.move_to_end(key)
    return _FRAME_CACHE[key].copy()


def iter_frames(video: Path, start: int, end: int):
    metadata = probe_video(video)
    if not 0 <= start < end <= metadata["frame_count"]:
        raise ValueError("Source frame interval outside video")
    if start == 0:
        capture = cv2.VideoCapture(str(video))
        try:
            for index in range(end):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode source frame {index}")
                yield index, frame
        finally:
            capture.release()
        return
    # Accurate timestamp seeking decodes from a preceding keyframe. Never use
    # frame/average-FPS to locate a source image. The tiny tolerance handles
    # FFprobe's six-decimal representation of rational source PTS.
    timestamp = max(0., metadata["pts_s"][start]-1e-6)
    command = [resolve_video_tool("ffmpeg"), "-v", "error", "-seek_timestamp", "1", "-ss", f"{timestamp:.9f}",
               "-i", str(video), "-map", "0:v:0", "-frames:v", str(end-start), "-fps_mode", "passthrough",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = metadata["width"]*metadata["height"]*3
    try:
        for index in range(start, end):
            data = process.stdout.read(size)
            if len(data) != size:
                raise ValueError(f"Cannot decode source frame {index}")
            yield index, np.frombuffer(data, np.uint8).reshape(metadata["height"], metadata["width"], 3).copy()
        if process.wait(timeout=30) != 0:
            raise RuntimeError(process.stderr.read().decode("utf-8", errors="replace"))
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=30)
        process.stderr.close()


def export_clip(video: Path, destination: Path, start: int, end: int, metadata: dict,
                *, ffmpeg: str | None = None, ffprobe: str | None = None) -> dict:
    """Reencode exact decoded-frame interval [start,end), preserving every PTS."""
    if not 0 <= start < end <= metadata["frame_count"] or end - start < 3:
        raise ValueError("Invalid clip interval")
    if video.resolve() == destination.resolve():
        raise ValueError("Cannot overwrite the source video")
    pts = np.asarray(metadata["pts_s"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.mp4")
    command = [resolve_video_tool("ffmpeg", ffmpeg), "-v", "error", "-y", "-i", str(video),
               "-map", "0:v:0", "-an", "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
               "-c:v", "libx264", "-preset", "fast", "-qp", "0", "-fps_mode", "passthrough",
               "-enc_time_base", "1:1000000", "-video_track_timescale", "1000000", str(temporary)]
    try:
        subprocess.run(command, check=True)
        result = probe_video(temporary, ffprobe)
        expected = pts[start:end] - pts[start]
        if (result["width"], result["height"]) != (metadata["width"], metadata["height"]):
            raise ValueError("Export changed image dimensions")
        if len(result["pts_s"]) != len(expected) or not np.allclose(result["pts_s"], expected, atol=2e-5, rtol=0):
            raise ValueError("Export changed source-frame timestamps")
        # Verify actual OpenCV decoding, including the final frame used downstream.
        decoded = sum(1 for _ in iter_frames(temporary, 0, len(expected)))
        if decoded != len(expected):
            raise ValueError("Export frame count mismatch")
        temporary.replace(destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def write_timed_video(frames, times, destination: Path, *, ffmpeg=None, ffprobe=None):
    """Encode an annotated stream at original variable PTS using temporary images."""
    times = np.asarray(times, dtype=float)
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("Annotated video requires three monotonic frames")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="seima-overlay-", dir=destination.parent) as folder:
        directory = Path(folder)
        listing = ["ffconcat version 1.0"]
        count = 0
        for count, frame in enumerate(frames, 1):
            name = f"frame_{count:06d}.jpg"
            if not cv2.imwrite(str(directory/name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError("Could not write overlay frame")
            dt = times[count]-times[count-1] if count < len(times) else float(np.median(np.diff(times)))
            listing += [f"file {name}", "option framerate 1000000", f"duration {dt:.9f}"]
        if count != len(times):
            raise ValueError("Overlay frames/timestamps differ")
        listing += [f"file frame_{count:06d}.jpg", "option framerate 1000000"]
        concat = directory / "frames.ffconcat"
        concat.write_text("\n".join(listing)+"\n", encoding="utf-8")
        output = directory / "overlay.mp4"
        subprocess.run([resolve_video_tool("ffmpeg", ffmpeg), "-v", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(concat), "-frames:v", str(count), "-c:v", "libx264", "-preset", "fast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-enc_time_base",
                        "1:1000000", "-video_track_timescale", "1000000", str(output)], check=True)
        actual = probe_video(output, ffprobe)
        if len(actual["pts_s"]) != len(times) or not np.allclose(actual["pts_s"], times-times[0], atol=2e-5, rtol=0):
            raise ValueError("Annotated output changed frame timestamps")
        output.replace(destination)
