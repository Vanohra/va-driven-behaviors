"""
Live Capture Module

Records a fixed-duration window of webcam video and microphone audio
to temporary files, which are then consumed by the existing offline
VA pipeline (extract_video_features / extract_audio_features).

Design note:
  The existing pipeline expects file paths (video + audio). Rather than
  rewriting the feature extractors, we record to temp files and hand
  those paths to the existing functions unchanged. This is the minimal
  adaptation needed to go from offline to online.

Usage:
    capture = LiveCapture(camera_index=0, fps=15.0)
    capture.open()
    result = capture.capture_window(duration_s=10.0, window_id=0)
    # result['video_path'] and result['audio_path'] are ready to pass to pipeline
    capture.close()

    # Or as a context manager:
    with LiveCapture() as capture:
        result = capture.capture_window(10.0, 0)
"""

import cv2
import numpy as np
import time
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict

# sounddevice is optional — video-only mode works without it.
# Install with: pip install sounddevice scipy
try:
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class LiveCapture:
    """
    Captures a fixed-duration window of webcam + microphone to temp files.

    Each call to capture_window() is blocking: it records for the full
    duration_s and then returns.  Audio capture runs in a background
    thread so video timing is not disrupted.

    Temp files are named by window_id so you can keep multiple windows
    on disk simultaneously if needed (though the default is to clean them
    up after analysis).
    """

    def __init__(
        self,
        camera_index: int = 0,
        fps: float = 15.0,
        resolution: Tuple[int, int] = (320, 240),
        sample_rate: int = 16000,
        capture_audio: bool = True,
        temp_dir: Optional[str] = None,
    ):
        """
        Args:
            camera_index:  OpenCV camera index (0 = default webcam).
            fps:           Target video frame rate.  Actual FPS may be lower
                           on slow machines; the analysis handles variable-length
                           clips gracefully.
            resolution:    (width, height) to capture.  Smaller = faster inference.
                           320x240 is enough for face-based VA estimation.
            sample_rate:   Audio sample rate in Hz (16 kHz matches Aff-Wild2 setup).
            capture_audio: Whether to attempt microphone capture.  Requires
                           sounddevice to be installed.  Falls back gracefully.
            temp_dir:      Where to write temp files.  Defaults to the system
                           temp directory (e.g., C:/Users/.../AppData/Local/Temp).
        """
        self.camera_index = camera_index
        self.fps = fps
        self.resolution = resolution
        self.sample_rate = sample_rate
        self.capture_audio = capture_audio and HAS_SOUNDDEVICE
        self.temp_dir = temp_dir or tempfile.gettempdir()

        self.cap: Optional[cv2.VideoCapture] = None

        if capture_audio and not HAS_SOUNDDEVICE:
            print(
                "Warning: sounddevice is not installed — running video-only mode.\n"
                "         Install with: pip install sounddevice scipy"
            )

    # -------------------------------------------------------------------------
    # Context-manager support
    # -------------------------------------------------------------------------

    def open(self) -> None:
        """Open and configure the webcam.  Must be called before capture_window()."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.camera_index}. "
                "Check that a webcam is connected and not in use by another app."
            )
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Read back actual settings (camera may not honor exact requests)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(
            f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} fps  "
            f"(requested {self.resolution[0]}x{self.resolution[1]} @ {self.fps})"
        )
        # Use actual resolution for the writer
        self.resolution = (actual_w, actual_h)

    def close(self) -> None:
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "LiveCapture":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Core capture method
    # -------------------------------------------------------------------------

    def capture_window(self, duration_s: float = 10.0, window_id: int = 0) -> Dict:
        """
        Capture duration_s seconds of video (and optionally audio) to temp files.

        This method blocks until the full duration has elapsed.

        Args:
            duration_s:  How many seconds to record.
            window_id:   Index used to name temp files (prevents collisions when
                         multiple windows are kept on disk simultaneously).

        Returns:
            Dict with keys:
                video_path       – path to the recorded .avi file
                audio_path       – path to .wav file, or same as video_path if
                                   no separate audio was captured
                n_frames         – number of frames written
                actual_duration_s – wall-clock seconds elapsed during recording
                fps_achieved     – frames / actual_duration_s
                has_real_audio   – True if a separate WAV was captured
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera not open. Call open() or use context manager.")

        # --- File paths ---
        video_path = os.path.join(self.temp_dir, f"va_window_{window_id}.avi")
        audio_path = os.path.join(self.temp_dir, f"va_window_{window_id}.wav")

        # --- Video writer (XVID codec → .avi, universally readable by cv2) ---
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.resolution)
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not open VideoWriter for {video_path}. "
                "Check that the XVID codec (ffmpeg/OpenCV) is available."
            )

        # --- Audio capture in background thread ---
        audio_chunks: list = []
        audio_stop = threading.Event()
        audio_thread_error: list = []  # Thread puts exceptions here

        if self.capture_audio:
            def _audio_callback(indata, frames, time_info, status):
                if not audio_stop.is_set():
                    audio_chunks.append(indata.copy())

            def _audio_thread():
                try:
                    with sd.InputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype="float32",
                        callback=_audio_callback,
                    ):
                        audio_stop.wait()  # Blocks until we signal stop
                except Exception as e:
                    audio_thread_error.append(e)

            t_audio = threading.Thread(target=_audio_thread, daemon=True)
            t_audio.start()

        # --- Video capture loop ---
        frame_count = 0
        target_frames = int(duration_s * self.fps)
        t_start = time.time()

        print(f"  [capture] Recording {duration_s:.0f}s  "
              f"(target ~{target_frames} frames at {self.fps} fps) ...", flush=True)

        while True:
            elapsed = time.time() - t_start
            if elapsed >= duration_s:
                break

            ret, frame = self.cap.read()
            if not ret:
                print("  Warning: Frame read failed — camera may have disconnected.")
                break

            # Resize if the camera didn't honour our requested resolution
            if (frame.shape[1], frame.shape[0]) != self.resolution:
                frame = cv2.resize(frame, self.resolution)

            writer.write(frame)
            frame_count += 1

        actual_duration = time.time() - t_start
        fps_achieved = frame_count / actual_duration if actual_duration > 0 else 0.0

        # --- Stop audio ---
        audio_stop.set()
        if self.capture_audio:
            t_audio.join(timeout=2.0)
            if audio_thread_error:
                print(f"  Warning: Audio thread error: {audio_thread_error[0]}")

        writer.release()

        print(
            f"  [capture] Done: {frame_count} frames, {actual_duration:.1f}s, "
            f"{fps_achieved:.1f} fps achieved",
            flush=True,
        )

        # --- Save audio if we got anything ---
        has_real_audio = False
        if audio_chunks and not audio_thread_error:
            try:
                audio_data = np.concatenate(audio_chunks, axis=0).squeeze()
                # sounddevice gives float32 in [-1, 1]; wavfile wants int16
                audio_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
                wavfile.write(audio_path, self.sample_rate, audio_int16)
                has_real_audio = True
                print(
                    f"  [capture] Audio saved: {len(audio_int16)/self.sample_rate:.1f}s "
                    f"@ {self.sample_rate} Hz",
                    flush=True,
                )
            except Exception as e:
                print(f"  Warning: Could not save audio: {e}")
                audio_path = video_path  # Fallback: let pipeline extract from video

        if not has_real_audio:
            audio_path = video_path  # Pipeline will extract (silent) audio from the .avi

        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "n_frames": frame_count,
            "actual_duration_s": actual_duration,
            "fps_achieved": fps_achieved,
            "has_real_audio": has_real_audio,
        }

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    @staticmethod
    def cleanup_files(capture_result: Dict) -> None:
        """Delete temp files produced by capture_window()."""
        for key in ("video_path", "audio_path"):
            path = capture_result.get(key)
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
