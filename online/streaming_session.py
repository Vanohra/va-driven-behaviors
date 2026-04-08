"""
Streaming Online Session — Ring Buffer + Pipelined Architecture

Architecture:
    Loop A (capture thread): Runs for the full session, writing camera frames
                             and microphone audio into a fixed-size ring buffer.
                             The camera never stops or pauses.

    Loop B (main thread):    Fires every window_dur seconds, takes a thread-safe
                             snapshot of the ring buffer, and submits the snapshot
                             to a worker thread for analysis.  Loop A continues
                             uninterrupted while the worker runs.

    Worker (thread pool):    Saves the snapshot to temp files, runs the full VA
                             pipeline, prints results and analysis timing.
                             Multiple workers can be in-flight simultaneously;
                             overflow is detected and reported.

Practical safeguards:
    - Thread-safe ring buffer with a single lock (brief critical sections).
    - Snapshot is a frozen copy — worker never touches the live buffer.
    - Audio is copied immediately in the sounddevice callback before the buffer
      is overwritten by the next chunk.
    - Worker overflow detection: warns if a previous worker is still running
      when the next snapshot is due.
    - Analysis timing printed for every window so you can see drift early.
    - Audio and video snapshotted atomically under the same lock.
    - Temp files cleaned up after each worker (configurable).
    - Print lock prevents interleaved output from concurrent workers.
    - Result lock guards the shared ReactionHistory state.
"""

import os
import time
import threading
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

from online.online_session import ReactionHistory, _extract_intent


# ─────────────────────────────────────────────────────────────────────────────
# RingBuffer
# ─────────────────────────────────────────────────────────────────────────────

class RingBuffer:
    """
    Thread-safe ring buffer for continuous video frames and audio samples.

    Frames are stored as numpy arrays in a fixed-size deque.  When the deque is
    full the oldest frame falls off automatically — no explicit trimming needed.

    Audio chunks are stored in a separate deque and trimmed from the front when
    the total sample count exceeds max_audio_samples.

    Both structures are snapshotted atomically under the same lock so that the
    returned frames and audio always correspond to the same moment in time.
    """

    def __init__(self, max_frames: int, max_audio_samples: int, sample_rate: int):
        self._lock              = threading.Lock()
        self._frames: deque     = deque(maxlen=max_frames)
        self._audio: deque      = deque()
        self._audio_n_samples   = 0
        self._max_audio_samples = max_audio_samples
        self.sample_rate        = sample_rate

    # ── Writers (called from capture / audio threads) ─────────────────────────

    def push_frame(self, frame: np.ndarray) -> None:
        """Append one frame.  cap.read() returns a new array each call — no copy needed."""
        with self._lock:
            self._frames.append(frame)

    def push_audio(self, chunk: np.ndarray) -> None:
        """
        Append one audio chunk.

        The sounddevice callback passes a view into an internal buffer that will
        be overwritten on the next callback, so we copy before acquiring the lock
        to keep the critical section as short as possible.
        """
        chunk = chunk.copy()
        n = len(chunk)
        with self._lock:
            self._audio.append(chunk)
            self._audio_n_samples += n
            # Drop oldest chunks until we are within the sample budget
            while self._audio_n_samples > self._max_audio_samples and self._audio:
                dropped = self._audio.popleft()
                self._audio_n_samples -= len(dropped)

    # ── Snapshot (called from the analysis timer, main thread) ────────────────

    def snapshot(self) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Atomically copy the current buffer state.

        Copying inside the lock ensures frames and audio always correspond to the
        same point in time.  The copy also means the worker thread never touches
        the live buffer.

        Returns:
            frames  — list of numpy arrays, oldest → newest
            audio   — concatenated float32 array, or None if no audio yet
        """
        with self._lock:
            frames = [f.copy() for f in self._frames]
            audio  = np.concatenate(list(self._audio), axis=0) if self._audio else None
        return frames, audio

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)


# ─────────────────────────────────────────────────────────────────────────────
# StreamingSession
# ─────────────────────────────────────────────────────────────────────────────

class StreamingSession:
    """
    Continuous live VA session using a ring buffer + pipelined architecture.

    The camera runs in a dedicated thread for the full session duration.
    Analysis is triggered on a timer and runs in a thread-pool worker so it
    never blocks the camera.

    Usage:
        session = StreamingSession(analyzer, ...)
        results = session.run()
    """

    def __init__(
        self,
        window_analyzer,
        camera_index:         int   = 0,
        fps:                  float = 15.0,
        resolution:           Tuple = (320, 240),
        session_duration_s:   float = 30.0,
        window_duration_s:    float = 3.0,
        capture_audio:        bool  = True,
        sample_rate:          int   = 16000,
        min_confidence:       float = 0.55,
        cooldown_s:           float = 1.5,
        debug:                bool  = False,
        cleanup_temp:         bool  = True,
        temp_dir: Optional[str]     = None,
        on_behavior_update          = None,
        num_samples: Optional[int]  = None,
    ):
        """
        on_behavior_update: optional callable invoked immediately when each
            worker reaches a behavior decision, before any detailed printing.
            Signature: on_behavior_update(intent: str,
                                          reaction_action,   # ReactionAction or None
                                          analysis: dict)    # full analysis dict or None
            Use this hook to drive robot hardware, websocket updates, etc.
        """
        self.analyzer            = window_analyzer
        self.camera_index        = camera_index
        self.fps                 = fps
        self.resolution          = resolution
        self.session_dur         = session_duration_s
        self.window_dur          = window_duration_s
        self.capture_audio       = capture_audio and HAS_SOUNDDEVICE
        self.sample_rate         = sample_rate
        self.debug               = debug
        self.cleanup_temp        = cleanup_temp
        self.temp_dir             = temp_dir or tempfile.gettempdir()
        self.on_behavior_update   = on_behavior_update
        self.num_samples          = num_samples
        self.cooldown_s           = cooldown_s

        self.n_windows = max(1, int(session_duration_s / window_duration_s))

        # Ring buffer: hold 1.5× the window duration worth of frames and audio.
        # The extra 0.5× gives the snapshot a small amount of overlap context
        # and absorbs any timer jitter without gaps.
        max_frames        = int(fps * window_duration_s * 1.5)
        max_audio_samples = int(sample_rate * window_duration_s * 1.5)
        self._buffer = RingBuffer(
            max_frames=max_frames,
            max_audio_samples=max_audio_samples,
            sample_rate=sample_rate,
        )

        # Threading primitives
        self._stop_event  = threading.Event()
        self._print_lock  = threading.Lock()   # Prevents interleaved terminal output
        self._result_lock = threading.Lock()   # Guards history + session_results

        # Worker overflow detection
        self._worker_lock    = threading.Lock()
        self._active_workers = 0

        # Session state
        self.history = ReactionHistory(
            min_confidence=min_confidence,
            cooldown_s=cooldown_s,
        )
        self.session_results: List[Dict] = []

        if capture_audio and not HAS_SOUNDDEVICE:
            print(
                "Warning: sounddevice not installed — running video-only mode.\n"
                "         Install with: pip install sounddevice scipy"
            )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> List[Dict]:
        """
        Run the streaming session.  Blocks until session_duration_s elapses.

        Loop A (capture thread) starts immediately and runs for the full session.
        Loop B (this thread) drives the analysis timer and waits for workers via
        the ThreadPoolExecutor context.
        """
        self._print_header()
        session_start = time.time()

        cap = self._open_camera()

        # Loop A — capture thread
        capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(cap,),
            daemon=True,
            name="CaptureLoop",
        )
        capture_thread.start()

        # Optional continuous audio thread
        if self.capture_audio:
            audio_thread = threading.Thread(
                target=self._audio_loop,
                daemon=True,
                name="AudioLoop",
            )
            audio_thread.start()

        print(
            f"  Camera running.  First analysis snapshot in {self.window_dur:.0f}s.",
            flush=True,
        )

        # Loop B — analysis timer
        # max_workers=2: one active worker + one queued if overflow occurs.
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="AnalysisWorker"
        ) as executor:
            self._executor = executor

            next_snapshot_t = session_start + self.window_dur
            window_idx      = 0

            while True:
                now     = time.time()
                elapsed = now - session_start

                # Snapshot check runs BEFORE the break so that the final
                # window (t == session_dur) is never skipped.
                if now >= next_snapshot_t and window_idx < self.n_windows:
                    self._submit_snapshot(window_idx, now - session_start)
                    window_idx      += 1
                    next_snapshot_t += self.window_dur

                if elapsed >= self.session_dur:
                    break

                time.sleep(0.02)  # 20 ms poll — light on the CPU

        # Executor __exit__ already waited for all pending workers to finish.
        self._stop_event.set()
        capture_thread.join(timeout=3.0)
        cap.release()

        # Sort by window index since workers may finish out of order
        self.session_results.sort(key=lambda r: r["window_idx"])
        self._print_summary(time.time() - session_start)
        return self.session_results

    # ── Loop A: continuous capture ────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.camera_index}. "
                "Check that a webcam is connected and not in use by another app."
            )
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        requested    = self.resolution
        actual_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps   = cap.get(cv2.CAP_PROP_FPS)
        self.resolution = (actual_w, actual_h)   # Update so writer matches camera output

        print(
            f"Camera opened: {actual_w}×{actual_h} @ {actual_fps:.1f} fps  "
            f"(requested {requested[0]}×{requested[1]} @ {self.fps})"
        )
        return cap

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        """Loop A — never pauses.  Writes every frame into the ring buffer."""
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("  [capture] Frame read failed — camera may have disconnected.")
                break
            # Resize if the camera did not honour our requested resolution
            if (frame.shape[1], frame.shape[0]) != self.resolution:
                frame = cv2.resize(frame, self.resolution)
            self._buffer.push_frame(frame)

    def _audio_loop(self) -> None:
        """Continuous sounddevice stream — writes chunks into the ring buffer."""
        def _callback(indata, frames, time_info, status):
            # push_audio copies indata before acquiring the lock
            self._buffer.push_audio(indata)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=_callback,
            ):
                self._stop_event.wait()
        except Exception as e:
            print(f"  [audio] Stream error: {e}")

    # ── Loop B: snapshot + worker submission ──────────────────────────────────

    def _submit_snapshot(self, window_idx: int, t_offset: float) -> None:
        """
        Atomically snapshot the ring buffer and submit an analysis worker.

        Overflow detection: if a previous worker is still running when this
        snapshot fires, a warning is printed.  The new worker is still submitted
        (the executor queues it) so no data is lost.
        """
        frames, audio = self._buffer.snapshot()

        with self._worker_lock:
            overflow = self._active_workers > 0
            self._active_workers += 1

        with self._print_lock:
            print()
            print("─" * 66)
            print(
                f"  WINDOW {window_idx+1}/{self.n_windows}  "
                f"|  t≈{t_offset:.0f}s  "
                f"|  {datetime.now().strftime('%H:%M:%S')}  "
                f"|  {len(frames)} frames in buffer"
            )
            if overflow:
                print(
                    f"  [WARNING] Worker overflow: previous analysis still running "
                    f"when this snapshot fired.  Consider increasing --window."
                )
            print("─" * 66, flush=True)

        try:
            self._executor.submit(
                self._analysis_worker, window_idx, t_offset, frames, audio
            )
        except Exception as e:
            print(f"  [ERROR] Could not submit worker for window {window_idx+1}: {e}")
            with self._worker_lock:
                self._active_workers -= 1

    # ── Worker (runs in thread pool) ──────────────────────────────────────────

    def _analysis_worker(
        self,
        window_idx: int,
        t_offset:   float,
        frames:     List[np.ndarray],
        audio:      Optional[np.ndarray],
    ) -> None:
        """
        Saves the snapshot to temp files, runs analysis, prints results + timing.
        Runs entirely in a worker thread — never blocks the camera.
        """
        t_worker_start = time.time()

        # ── 1. Save snapshot to temp files ────────────────────────────────────
        video_path, audio_path, has_audio = self._save_snapshot(
            frames, audio, window_idx
        )

        if video_path is None:
            with self._print_lock:
                print(f"  [Window {window_idx+1}] Snapshot had no frames — skipping.")
            with self._worker_lock:
                self._active_workers -= 1
            return

        # ── 2. Run analysis ───────────────────────────────────────────────────
        with self._print_lock:
            print(
                f"  [Window {window_idx+1}] Running VA analysis pipeline...",
                flush=True,
            )

        analysis = None
        try:
            analysis = self.analyzer.analyze_window(
                video_path=video_path,
                audio_path=audio_path,
                num_samples=self.num_samples,
            )
        except Exception as e:
            with self._print_lock:
                print(f"  [Window {window_idx+1}] Analysis error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

        t_analysis_elapsed = time.time() - t_worker_start

        # ── 3. Decide ─────────────────────────────────────────────────────────
        now = time.time()
        if analysis is not None:
            proposed = _extract_intent(analysis)
            conf     = analysis.get("state_confidence", 0.0)
            label    = analysis.get("va_state_label", "unknown")
        else:
            proposed = self.history.fallback_intent
            conf     = 0.0
            label    = "unknown"

        with self._result_lock:
            effective, reason, did_change = self.history.evaluate(
                proposed, conf, label, now
            )

        # ── 4. Dispatch behavior immediately, then print detailed result ─────
        with self._print_lock:
            self._dispatch_behavior(
                window_idx, t_offset, effective, did_change, analysis
            )
            self._print_window_result(
                window_idx, analysis, effective, reason, did_change,
                len(frames), has_audio, t_analysis_elapsed,
            )

        # ── 5. Store result ───────────────────────────────────────────────────
        with self._result_lock:
            self.session_results.append({
                "window_idx":         window_idx,
                "window_start_s":     t_offset,
                "analysis":           analysis,
                "effective_intent":   effective,
                "reason":             reason,
                "did_change":         did_change,
                "n_frames":           len(frames),
                "has_audio":          has_audio,
                "analysis_elapsed_s": t_analysis_elapsed,
            })

        # ── 6. Cleanup temp files ─────────────────────────────────────────────
        if self.cleanup_temp:
            for path in set([video_path, audio_path]):
                if path and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        with self._worker_lock:
            self._active_workers -= 1

    # ── Snapshot saving ───────────────────────────────────────────────────────

    def _save_snapshot(
        self,
        frames:    List[np.ndarray],
        audio:     Optional[np.ndarray],
        window_id: int,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Write snapshot frames and audio to temp files.

        Returns:
            (video_path, audio_path, has_real_audio)
            video_path is None if there were no frames to write.
            audio_path falls back to video_path when no separate WAV was saved,
            matching the convention used by the existing offline pipeline.
        """
        if not frames:
            return None, None, False

        video_path = os.path.join(self.temp_dir, f"va_stream_{window_id}.avi")
        audio_path = os.path.join(self.temp_dir, f"va_stream_{window_id}.wav")

        # Write video frames
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.resolution)
        for frame in frames:
            writer.write(frame)
        writer.release()

        # Write audio
        has_audio = False
        if audio is not None and len(audio) > 0 and HAS_SOUNDDEVICE:
            try:
                audio_flat  = audio.squeeze()
                audio_int16 = (audio_flat * 32767).clip(-32768, 32767).astype(np.int16)
                wavfile.write(audio_path, self.sample_rate, audio_int16)
                has_audio = True
            except Exception as e:
                print(f"  [snapshot] Could not save audio: {e}")

        return video_path, audio_path if has_audio else video_path, has_audio

    # ── Immediate behavior dispatch ───────────────────────────────────────────

    def _dispatch_behavior(
        self,
        window_idx:       int,
        t_offset:         float,
        effective_intent: str,
        did_change:       bool,
        analysis:         Optional[Dict],
    ) -> None:
        """
        Emit the actionable behavior output immediately when a worker finishes.

        This is the first thing printed after a decision is reached — before
        the detailed VA breakdown — so that any robot controller or downstream
        consumer receives the decision with minimum latency.

        If on_behavior_update was supplied at construction time, it is called
        here with (intent, reaction_action, analysis) for hardware integration.
        """
        ra    = analysis.get("reaction_action") if analysis else None
        label = analysis.get("va_state_label",  "unknown") if analysis else "unknown"
        conf  = analysis.get("state_confidence", 0.0)      if analysis else 0.0
        v     = analysis.get("valence",           0.0)     if analysis else 0.0
        a     = analysis.get("arousal",           0.0)     if analysis else 0.0

        change_tag = "CHANGED" if did_change else "maintained"
        bar = "─" * 52

        print(f"\n  ┌─ BEHAVIOR DISPATCH  "
              f"[window {window_idx+1}  t≈{t_offset:.0f}s] {bar}┐")
        print(f"  │  Intent :  {effective_intent:<12}  ({change_tag})")
        if ra:
            print(f"  │  Pose   :  {ra.pose_mode:<14}  "
                  f"speed={ra.speed_mult:.2f}×  dist={ra.distance_mult:.2f}×  "
                  f"dur={ra.duration_s:.1f}s")
        print(f"  │  State  :  {label}  (conf: {conf:.3f})")
        print(f"  │  VA     :  valence={v:+.3f}   arousal={a:+.3f}")
        print(f"  └{bar}──────────────────────┘")

        # ── Optional robot / hardware callback ────────────────────────────────
        if self.on_behavior_update is not None:
            try:
                self.on_behavior_update(effective_intent, ra, analysis)
            except Exception as e:
                print(f"  [dispatch] Callback error: {e}")

    # ── Terminal output ───────────────────────────────────────────────────────

    def _print_header(self) -> None:
        w = 64
        print()
        print("╔" + "═" * w + "╗")
        print("║" + "  VA STREAMING SESSION  ".center(w) + "║")
        print("║" + "  (Ring Buffer + Pipelined)  ".center(w) + "║")
        print("╚" + "═" * w + "╝")
        print(f"  Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration:     {self.session_dur:.0f}s  "
              f"|  {self.n_windows} windows × {self.window_dur:.0f}s each")
        print(f"  Mode:         Camera never pauses — analysis runs in parallel")
        print(f"  Overflow:     warned if analysis takes longer than {self.window_dur:.0f}s")
        print(f"  Confidence:   {self.history.min_confidence:.2f}  "
              f"|  Cooldown: {self.history.cooldown_s:.0f}s")
        print()

    def _print_window_result(
        self,
        idx:               int,
        analysis:          Optional[Dict],
        effective_intent:  str,
        reason:            str,
        did_change:        bool,
        n_frames:          int,
        has_audio:         bool,
        analysis_elapsed:  float,
    ) -> None:
        au = "mic" if has_audio else "silent"
        print(f"\n  Snapshot: {n_frames} frames  |  Audio: {au}")

        # Analysis timing — flag if it exceeded the window duration
        overtime = analysis_elapsed > self.window_dur
        tag = (
            f"  *** exceeds window ({self.window_dur:.0f}s) — worker overflow likely next!"
            if overtime else ""
        )
        print(f"  Analysis time: {analysis_elapsed:.2f}s{tag}")

        if analysis is None:
            print("\n  [ANALYSIS FAILED — using fallback]")
        else:
            v     = analysis.get("valence",           0.0)
            a     = analysis.get("arousal",            0.0)
            v_dir = analysis.get("valence_direction",  "?")
            a_dir = analysis.get("arousal_direction",  "?")
            v_vol = analysis.get("valence_volatility", 0.0)
            a_vol = analysis.get("arousal_volatility", 0.0)
            label = analysis.get("va_state_label",     "?")
            conf  = analysis.get("state_confidence",   0.0)
            vol   = analysis.get("volatility",         max(v_vol, a_vol))
            ra    = analysis.get("reaction_action")
            mode  = ra.pose_mode if ra else "N/A"

            print(f"\n  VA Baseline:")
            print(f"    Valence:  {v:+.3f}  | Trend: {v_dir:<12} | Volatility: {v_vol:.3f}")
            print(f"    Arousal:  {a:+.3f}  | Trend: {a_dir:<12} | Volatility: {a_vol:.3f}")
            print(f"\n  State:")
            print(f"    Label:      {label}")
            print(f"    Confidence: {conf:.3f}   Volatility: {vol:.3f}")
            print(f"    Pose mode:  {mode}")

            if ra and self.debug:
                print(f"\n  Reaction detail:")
                print(f"    Speed:    {ra.speed_mult:.2f}x  "
                      f"Distance: {ra.distance_mult:.2f}x  "
                      f"Duration: {ra.duration_s:.1f}s")
                print(f"    Explain:  {ra.explain}")

        prev   = self.history.history[-2]["intent"] if len(self.history.history) >= 2 else "—"
        labels = {
            "first":          "first window",
            "low_confidence": f"confidence < {self.history.min_confidence:.2f}",
            "cooldown":       "cooldown active",
            "same":           "no change",
            "changed":        "accepted",
        }
        print()
        if did_change:
            print(f"  ★  BEHAVIOR UPDATE:  {prev}  →  {effective_intent}  "
                  f"({labels.get(reason, reason)})")
        else:
            print(f"  ·  Maintaining:  {effective_intent}  ({labels.get(reason, reason)})")

    def _print_summary(self, elapsed_s: float) -> None:
        w = 64
        print()
        print("╔" + "═" * w + "╗")
        print("║" + "  SESSION COMPLETE  ".center(w) + "║")
        print("╚" + "═" * w + "╝")
        print(f"  Total time: {elapsed_s:.1f}s")
        print()
        print("  Behavior timeline:")
        print(
            f"  {'Win':<5} {'t':>5}  {'VA Label':<26} "
            f"{'Conf':>5}  {'Analysis':>9}  {'Intent':<15}  Note"
        )
        print("  " + "─" * 80)
        for r in self.session_results:
            a       = r.get("analysis")
            label   = a.get("va_state_label",    "error") if a else "error"
            conf    = a.get("state_confidence",  0.0)     if a else 0.0
            intent  = r.get("effective_intent",  "N/A")
            note    = "← changed" if r.get("did_change") else ""
            t       = r["window_start_s"]
            idx     = r["window_idx"]
            timing  = r.get("analysis_elapsed_s", 0.0)
            overtime_flag = " !" if timing > self.window_dur else "  "
            print(
                f"  {idx+1:<5} {t:>5.0f}s  {label:<26} "
                f"{conf:>5.2f}  {timing:>7.2f}s{overtime_flag}  {intent:<15}  {note}"
            )
        print()
        print("  Note: '!' in the Analysis column means that window's analysis")
        print(f"        exceeded the {self.window_dur:.0f}s window duration (worker overflow risk).")
        print()
