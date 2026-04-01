"""
Online Pipeline Package

Provides live webcam/microphone capture and session orchestration
for the interactive VA pipeline.

Components:
  StreamingSession — ring buffer + pipelined session (camera never pauses)
  WindowAnalyzer   — runs the full VA pipeline on one video window
  LiveCapture      — legacy: records fixed-duration windows to temp files
  OnlineSession    — legacy: sequential capture → analyze session
"""

from .streaming_session import StreamingSession
from .window_analyzer   import WindowAnalyzer
from .live_capture      import LiveCapture
from .online_session    import OnlineSession

__all__ = ["StreamingSession", "WindowAnalyzer", "LiveCapture", "OnlineSession"]
