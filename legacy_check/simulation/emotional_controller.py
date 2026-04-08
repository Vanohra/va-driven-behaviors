"""
Emotional Controller for Boston Dynamics Spot Robot

This module integrates valence-arousal emotion recognition with Spot robot control.
It processes video streams (or simulated camera feeds) and generates appropriate
robot behaviors based on detected emotional states and trends.

The system uses:
- Calibration-based threshold calculations
- Trend analysis from time series data
- VA state classification
- Robot reaction recommendations
- Spot command mapping and execution

Now integrated with locomotive-capable Spot simulation using BezierStepper.
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path

# Force unbuffered output to ensure prints appear immediately (especially with GUI)
# Set environment variable to force Python unbuffered mode
os.environ['PYTHONUNBUFFERED'] = '1'

# Calculate absolute path to project root dynamically
_current_file = os.path.abspath(__file__)
# Go up: simulation/ -> legacy_check/ -> root
EMOTION_REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(_current_file)))

if EMOTION_REPO_PATH not in sys.path:
    sys.path.append(EMOTION_REPO_PATH)

try:
    from test_emotions import (
        load_model,
        extract_video_features,
        extract_audio_features,
        align_features,
        predict_emotions
    )
except ImportError as e:
    print(f"Error importing emotion model: {e}")
    print(f"Make sure requirements are installed and path is correct: {EMOTION_REPO_PATH}")
    sys.exit(1)

# Add spotmicro to path for new simulation
# Calculate absolute path to spotmicro directory
_current_file = os.path.abspath(__file__)
# From: spot_mini_mini/spot_bullet/src/emotion/integration/emotional_controller.py
# Add spotmicro to path for new simulation
# Calculate absolute path to spotmicro directory
# From: legacy_check/simulation/emotional_controller.py
_current_file = os.path.abspath(__file__)
# Go up: simulation/ -> legacy_check/ -> root
_legacy_dir = os.path.dirname(_current_file)
_root_dir = os.path.dirname(os.path.dirname(_legacy_dir))

# The simulation expects a specific structure if running in the full spot_mini_mini repo
_spotmicro_path = os.path.join(_root_dir, 'spotmicro')

if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if _spotmicro_path not in sys.path:
    sys.path.insert(0, _spotmicro_path)
from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv

# Import emotion pipeline components
from ..core.emotion_analyzer import load_calibration, analyze_emotion_stream
from ..core.spot_reaction_mapper import SpotReactionMapper
from ..core.affect_filter import AffectFilter
from ..core.intent_selector import IntentSelector
from ..execution.behavior_renderer import BehaviorRenderer
from ..execution.reaction_executor import ReactionExecutor
from .new_sim_adapter import NewSimSpotAdapter
from .rl_agent_adapter import RLAgentAdapter


class EmotionalSpotController:
    """
    Main controller class that integrates emotion recognition with Spot robot control.
    
    Now uses the new locomotive-capable Spot simulation with BezierStepper integration.
    """
    
    def __init__(self, 
                 calibration_path: str = None,
                 model_path: str = None,
                 device: str = 'cpu',
                 trend_threshold: float = None,
                 volatility_threshold: float = None,
                 debug: bool = False):
        """
        Initialize the emotional Spot controller.
        
        Args:
            calibration_path: Path to calibration.json file (optional)
            model_path: Path to JointCAM model checkpoint
            device: 'cpu' or 'cuda'
            trend_threshold: Optional threshold for trend direction (uses scale-aware default if None)
            volatility_threshold: Optional threshold for high volatility (uses scale-aware default if None)
            debug: If True, print debug information
        """
        self.device = device
        self.debug = debug
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        
        # Load calibration (try data folder first, then fallback)
        self.calibration = None
        if calibration_path:
            self.calibration = load_calibration(calibration_path)
        else:
            # Try to find calibration.json in data folder
            data_dir = Path(__file__).parent.parent / "data"
            default_calibration = data_dir / "calibration.json"
            if default_calibration.exists():
                self.calibration = load_calibration(str(default_calibration))
            else:
                # Fallback to current directory
                default_calibration = Path("calibration.json")
                if default_calibration.exists():
                    self.calibration = load_calibration(str(default_calibration))
        
        # Load model (try data folder first, then fallback)
        if model_path is None:
            # Try data folder first
            data_dir = Path(__file__).parent.parent / "data"
            model_path_data = data_dir / "jointcam_model.pt"
            if model_path_data.exists():
                model_path = str(model_path_data)
            else:
                # Fallback to emotion-poc directory
                model_path = os.path.join(EMOTION_REPO_PATH, "jointcam_model.pt")
        
        print(f"Loading emotion model from: {model_path}")
        self.model = load_model(model_path, device)
        print("Model loaded successfully.")
        
        # Initialize reaction mapper
        self.reaction_mapper = SpotReactionMapper()
        
        # Initialize intent selector (for 5-intent emotion policy)
        self.intent_selector = IntentSelector(
            volatility_high_threshold=0.25,
            volatility_med_threshold=0.15,
            confidence_threshold=0.6
        )
        
        # Initialize affect filter - maximum smoothing for ultra-smooth behavior
        self.affect_filter = AffectFilter(
            ema_alpha=0.10,  # Maximum smoothing (ultra-smooth)
            max_delta_per_update=0.05,  # Very restrictive rate limiting (very gradual)
            hysteresis_margin=0.1
        )
        
        # Initialize environment, renderer, and executor (will be set when run is called)
        self.env = None
        self.renderer = None
        self.executor = None
        # Timing for simulation-only stabilization phases (seconds)
        self._stand_settle_duration_s = 2.0   # Time to let physics settle after commanding stand
        self._neutral_freeze_duration_s = 2.0 # Extra neutral hold before first visible reaction
    
    def process_video_stream(self, video_path: str, audio_path: str = None) -> dict:
        """
        Process a video file and return emotion analysis results.
        
        Args:
            video_path: Path to video file
            audio_path: Optional path to separate audio file (defaults to video_path)
        
        Returns:
            Dictionary with comprehensive emotion analysis results
        """
        if audio_path is None:
            audio_path = video_path
        
        print(f"Processing video: {video_path}", flush=True)
        
        try:
            # Extract features
            video_features, fps = extract_video_features(video_path, self.device)
            audio_features = extract_audio_features(audio_path)
            
            # Align features
            video_aligned, audio_aligned = align_features(video_features, audio_features)
            
            # Run inference - returns per-frame arrays
            valence_series, arousal_series = predict_emotions(
                self.model, video_aligned, audio_aligned, self.device
            )
            
            # Ensure 1D arrays
            if valence_series.ndim > 1:
                valence_series = valence_series.flatten()
            if arousal_series.ndim > 1:
                arousal_series = arousal_series.flatten()
            
            # Analyze emotion stream
            analysis = analyze_emotion_stream(
                valence_series,
                arousal_series,
                calibration=self.calibration,
                trend_threshold=self.trend_threshold,
                volatility_threshold=self.volatility_threshold,
                export_timeseries=True,
                debug=self.debug
            )
            
            # Add metadata
            analysis['fps'] = fps
            analysis['num_frames'] = len(valence_series)
            analysis['video_path'] = video_path
            
            # Print analysis results immediately (use flush=True to ensure output appears even with GUI)
            print("\n" + "=" * 60, flush=True)
            print("VIDEO PROCESSING COMPLETE - EMOTION ANALYSIS RESULTS", flush=True)
            print("=" * 60, flush=True)
            print(f"Video: {video_path}", flush=True)
            print(f"Frames: {len(valence_series)}, FPS: {fps:.2f}", flush=True)
            print(f"\nBaseline VA State:", flush=True)
            print(f"  Valence: {analysis['valence']:.4f} (std: {analysis['valence_std']:.4f})", flush=True)
            print(f"  Arousal: {analysis['arousal']:.4f} (std: {analysis['arousal_std']:.4f})", flush=True)
            print(f"\nVA State Label: {analysis['va_state_label']}", flush=True)
            if 'state_confidence' in analysis:
                print(f"State Confidence: {analysis['state_confidence']:.2f}", flush=True)
            print(f"\nTrends:", flush=True)
            print(f"  Valence: {analysis['valence_direction']} (delta: {analysis['valence_delta']:.4f}, slope: {analysis['valence_slope']:.6f})", flush=True)
            print(f"  Arousal: {analysis['arousal_direction']} (delta: {analysis['arousal_delta']:.4f}, slope: {analysis['arousal_slope']:.6f})", flush=True)
            print(f"\nVolatility:", flush=True)
            print(f"  Valence: {analysis['valence_volatility']:.4f}", flush=True)
            print(f"  Arousal: {analysis['arousal_volatility']:.4f}", flush=True)
            if 'va_baseline' in analysis:
                v_baseline = analysis['va_baseline']['valence']
                a_baseline = analysis['va_baseline']['arousal']
                print(f"\nRobust Baseline Statistics:", flush=True)
                print(f"  Valence - MAD: {v_baseline['mad']:.4f}, IQR: {v_baseline['iqr']:.4f}, Stability: {v_baseline['stability_score']:.2f}", flush=True)
                print(f"  Arousal - MAD: {a_baseline['mad']:.4f}, IQR: {a_baseline['iqr']:.4f}, Stability: {a_baseline['stability_score']:.2f}", flush=True)
            print(f"\nReaction Recommendation:", flush=True)
            print(f"  {analysis['reaction_recommendation']}", flush=True)
            print(f"  Notes: {analysis['notes']}", flush=True)
            print("=" * 60 + "\n", flush=True)
            
            return analysis
            
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def execute_reaction(self, analysis: dict, env):
        """
        Execute robot reaction based on emotion analysis.
        
        Uses filtered affect signals, ReactionExecutor for safe execution,
        and BehaviorRenderer for rendering.
        
        Args:
            analysis: Dictionary from analyze_emotion_stream
            env: SpotEnv-compatible instance (NewSimSpotAdapter)
        """
        if analysis is None:
            return
        
        # Extract raw analysis data
        raw_valence = analysis['valence']
        raw_arousal = analysis['arousal']
        state_label = analysis['va_state_label']
        valence_volatility = analysis.get('valence_volatility', 0.0)
        arousal_volatility = analysis.get('arousal_volatility', 0.0)
        volatility = max(valence_volatility, arousal_volatility)
        
        # Filter affect signals (smooth and rate-limit)
        filtered_valence, filtered_arousal = self.affect_filter.update(raw_valence, raw_arousal)
        
        # Build trends dict
        trends = {
            'valence_direction': analysis.get('valence_direction', 'stable'),
            'arousal_direction': analysis.get('arousal_direction', 'stable'),
            'valence_delta': analysis.get('valence_delta', 0.0),
            'arousal_delta': analysis.get('arousal_delta', 0.0)
        }
        
        # Estimate confidence (simplified: based on volatility)
        # Lower volatility = higher confidence
        confidence = max(0.3, min(1.0, 1.0 - volatility / 0.5))
        
        # Select intent using 5-intent emotion policy
        intent, pose_mode, intent_explain = self.intent_selector.select_intent(
            valence=filtered_valence,
            arousal=filtered_arousal,
            volatility=volatility,
            confidence=confidence,
            va_label=state_label
        )
        
        # If using RL agent, update emotion-based locomotion parameters with intent and pose mode
        if isinstance(self.env, RLAgentAdapter):
            # Get speed and distance multipliers from reaction mapper
            # (We'll use the reaction mapper's logic to get these values)
            action, _ = self.reaction_mapper.map_to_action(
                va_label=state_label,
                trends=trends,
                volatility=volatility,
                confidence=confidence,
                valence=filtered_valence,
                arousal=filtered_arousal
            )
            
            # Update RL agent with intent-based emotion
            self.env.set_emotion(
                filtered_valence, 
                filtered_arousal,
                volatility=volatility,
                trends=trends,
                intent=intent.value,  # Pass intent string
                pose_mode=pose_mode,  # Pass pose mode
                confidence=confidence,
                speed_mult=action.speed_mult,
                distance_mult=action.distance_mult
            )
        
        # Check if state transition should occur (hysteresis)
        should_transition = self.affect_filter.should_transition_state(state_label)
        if not should_transition and self.executor and self.executor.current_intent:
            # Maintain current state if transition not warranted
            if self.debug:
                print(f"[FILTER] Maintaining current state (hysteresis), proposed: {state_label}", flush=True)
            return
        
        # Map to structured action (using filtered values for more stable behavior)
        # Note: We use the intent from intent_selector, but reaction_mapper provides speed/distance multipliers
        action, explain = self.reaction_mapper.map_to_action(
            va_label=state_label,
            trends=trends,
            volatility=volatility,
            confidence=confidence,
            valence=filtered_valence,  # Use filtered values
            arousal=filtered_arousal
        )
        
        # Override action intent with intent from intent selector (5-intent policy)
        # This ensures we use the correct intent from our policy logic
        action.intent = intent.value
        action.pose_mode = pose_mode
        action.pose_name = self.intent_selector.get_pose_name(intent, pose_mode)
        
        # Print summary (use flush=True to ensure output appears even with GUI)
        print("\n" + "=" * 60, flush=True)
        print("EXECUTING REACTION - DETAILED ANALYSIS", flush=True)
        print("=" * 60, flush=True)
        print(f"Raw Valence: {raw_valence:.4f} -> Filtered: {filtered_valence:.4f}", flush=True)
        print(f"Raw Arousal: {raw_arousal:.4f} -> Filtered: {filtered_arousal:.4f}", flush=True)
        print(f"VA State: {state_label}", flush=True)
        print(f"Volatility: {volatility:.4f}, Confidence: {confidence:.2f}", flush=True)
        print(f"Valence Trend: {trends['valence_direction']} (delta: {trends['valence_delta']:.4f})", flush=True)
        print(f"Arousal Trend: {trends['arousal_direction']} (delta: {trends['arousal_delta']:.4f})", flush=True)
        print(f"\nIntent: {intent.value} (from 5-intent policy)", flush=True)
        print(f"Pose Mode: {pose_mode}", flush=True)
        print(f"Intent Explanation: {intent_explain}", flush=True)
        print(f"Action: {action.intent}", flush=True)
        print(f"Speed Mult: {action.speed_mult:.2f}, Distance Mult: {action.distance_mult:.2f}", flush=True)
        print(f"Explain: {explain}", flush=True)
        print("=" * 60 + "\n", flush=True)
        
        # Initialize renderer and executor if needed
        if self.renderer is None:
            self.renderer = BehaviorRenderer(env, verbose=True)
        
        if self.executor is None:
            self.executor = ReactionExecutor(
                env=env,
                renderer=self.renderer,
                min_dwell_time=1.5,  # Minimum 1.5s before allowing new reaction
                blend_duration=1.2,  # Increased to 1.2s for smoother transitions (becomes 3.0s internally)
                max_joint_velocity=2.0,  # rad/s (reduced to 1.4 rad/s internally)
                max_base_velocity=0.3,  # m/s (reduced to 0.21 m/s internally)
                verbose=True
            )
            # Link executor to env for safety callbacks
            env.executor = self.executor
        
        # Attempt to start reaction (executor handles state machine, dwell time, blending)
        started = self.executor.start_reaction(action, valence=filtered_valence, arousal=filtered_arousal)
        
        if not started:
            if self.debug:
                print(f"[EXECUTOR] Reaction {action.intent} not started (dwell time or state conflict)", flush=True)
        
        # Update executor (handles trajectory following, state transitions)
        # This should be called in the main control loop, but we do it here for batch mode
        dt = 1.0 / 240.0  # Physics timestep
        self.executor.update(dt)
    
    def _stand_and_settle(self):
        """
        Simulation-only helper: command a neutral stand and let physics fully settle
        before any affect-aware behaviors are rendered.
        """
        if self.env is None:
            return

        print("\n[SIM] Stand-and-settle phase: commanding neutral stand and waiting to stabilize...", flush=True)
        # Force a clean neutral baseline and pose
        try:
            # Prefer explicit baseline if available
            if hasattr(self.env, "set_baseline_pose"):
                self.env.set_baseline_pose("NEUTRAL")
        except Exception:
            # Fallback: ignore if baseline helper not present
            pass

        # Always ensure we issue a stand command so joint motors are engaged
        if hasattr(self.env, "stand"):
            self.env.stand()

        # Let physics run for a short period so gravity / contacts fully settle
        # Use high-rate stepping so the robot lands and stabilizes before any reactions.
        settle_steps = int(self._stand_settle_duration_s * 240.0)
        for _ in range(settle_steps):
            self.env.step()

        print("[SIM] Stand-and-settle complete.\n", flush=True)

    def _neutral_freeze_before_first_reaction(self):
        """
        Simulation-only helper: hold a neutral, frozen stand *after* emotion
        analysis completes but *before* the first visible reaction executes.

        This ensures the first noticeable motion is a controlled reaction
        and not the residual physics settling.
        """
        if self.env is None:
            return

        print(f"[SIM] Neutral freeze: holding stand for {self._neutral_freeze_duration_s:.1f}s before first reaction...", flush=True)

        # Re-assert neutral posture without starting any overlays or base motion.
        try:
            if hasattr(self.env, "set_baseline_pose"):
                self.env.set_baseline_pose("NEUTRAL")
        except Exception:
            pass

        if hasattr(self.env, "stand"):
            self.env.stand()

        # Use the environment's hold helper if present; otherwise, just step physics.
        freeze_duration = self._neutral_freeze_duration_s
        if hasattr(self.env, "hold"):
            self.env.hold(freeze_duration)
        else:
            steps = int(freeze_duration * 240.0)
            for _ in range(steps):
                self.env.step()

        print("[SIM] Neutral freeze complete. Starting first reaction.\n", flush=True)

    def run_simulation(self, video_path: str, 
                      audio_path: str = None,
                      gui: bool = True,
                      realtime: bool = False,
                      use_rl_agent: bool = False,
                      agent_num: int = 2229):
        """
        Run the full simulation: process video and control Spot robot.
        
        Now uses the new locomotive-capable Spot simulation with BezierStepper.
        
        IMPORTANT: Video processing happens BEFORE GUI initialization for speed.
        This ensures video processing is fast regardless of GUI setting.
        
        Args:
            video_path: Path to video file
            audio_path: Optional path to separate audio file
            gui: Whether to show PyBullet GUI
            realtime: If True, process frame-by-frame in real-time. If False, process entire video first.
            use_rl_agent: If True, use pretrained RL agent for intelligent locomotion. If False, use direct control.
            agent_num: Agent number to load (default: 2229). Only used if use_rl_agent=True.
        """
        # IMPORTANT: Process video FIRST (before GUI initialization) for speed
        # This ensures video processing is fast regardless of GUI setting
        print("\n" + "=" * 60, flush=True)
        print("PROCESSING VIDEO (before GUI initialization)", flush=True)
        print("=" * 60, flush=True)
        
        # Process entire video first to get full analysis
        analysis = self.process_video_stream(video_path, audio_path)
        if analysis is None:
            print("Failed to process video. Exiting.", flush=True)
            return
        
        # NOW initialize GUI (after video processing is complete)
        if use_rl_agent:
            print("Initializing new Spot simulation environment with PRETRAINED RL AGENT...", flush=True)
            print(f"  Agent: {agent_num}", flush=True)
            print("  The robot will maintain balance while expressing emotions!", flush=True)
        else:
            print("Initializing new Spot simulation environment (direct control mode)...", flush=True)
        
        # Create spotBezierEnv (GUI starts here, but video is already processed)
        new_env = spotBezierEnv(
            render=gui,
            AutoStepper=True,  # Enable BezierStepper for locomotion
            contacts=True
        )
        
        # Reset environment
        new_env.reset()
        
        # Wrap with appropriate adapter
        if use_rl_agent:
            self.env = RLAgentAdapter(new_env, agent_num=agent_num, use_contacts=True)
            print(f"[RL Agent] Using pretrained agent {agent_num} for intelligent locomotion", flush=True)
        else:
            self.env = NewSimSpotAdapter(new_env)
            print("[Direct Control] Using direct joint control for poses", flush=True)
        
        self.renderer = BehaviorRenderer(self.env, verbose=True)
        
        # Explicit stand-and-settle phase before any affect-aware control
        self._stand_and_settle()
        
        if realtime:
            # Real-time processing (frame-by-frame) with already-processed analysis
            self._run_realtime(video_path, audio_path, analysis)
        else:
            # Batch processing - execute reaction with already-processed analysis
            self._run_batch(analysis)
    
    def _run_batch(self, analysis: dict):
        """
        Execute reaction with already-processed analysis.
        Note: Video processing happens in run_simulation() before GUI initialization.
        
        Args:
            analysis: Already-processed emotion analysis dictionary
        """
        # Before executing the first affect-driven reaction in simulation,
        # insert a neutral freeze window so the robot is visibly stable.
        print("\n[Simulation] Initializing robot and preparing for reaction...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()  # Also flush stderr
        self._neutral_freeze_before_first_reaction()

        # Execute reaction based on overall analysis
        self.execute_reaction(analysis, self.env)
        
        # Main control loop: update executor and step physics
        physics_hz = 240.0
        dt = 1.0 / physics_hz
        target_fps = 25.0
        
        # Run for reaction duration
        if self.executor and self.executor.current_action:
            duration = self.executor.current_action.duration_s
            total_steps = int(duration * physics_hz)
            
            for step_idx in range(total_steps):
                # Update executor (handles trajectory, state transitions)
                self.executor.update(dt)
                
                # Step physics
                self.env.step()
        
        # Optionally, we can also process frame-by-frame for visualization
        if 'valence_timeseries' in analysis and 'arousal_timeseries' in analysis:
            print("\n[Simulation] Playing back frame-by-frame emotional trajectory...", flush=True)
            valence_seq = np.array(analysis['valence_timeseries'])
            arousal_seq = np.array(analysis['arousal_timeseries'])
            
            fps = analysis.get('fps', 25.0)
            physics_steps_per_frame = int((1.0 / fps) / (1.0 / physics_hz))
            
            for i in range(len(valence_seq)):
                val = valence_seq[i]
                aro = arousal_seq[i]
                
                # Apply emotional pose or locomotion based on adapter type
                if isinstance(self.env, RLAgentAdapter):
                    # RL agent: update emotion-based locomotion
                    self.env.set_emotion(val, aro)
                else:
                    # Direct control: apply emotional pose
                    self.env.emotional_pose(valence=val, arousal=aro)
                
                # Update executor
                if self.executor:
                    self.executor.update(dt)
                
                # Step physics at high rate so subtle emotional poses are visible
                for _ in range(physics_steps_per_frame):
                    self.env.step()
        
        print("\n[Simulation] Complete. Press Ctrl+C to exit.", flush=True)
        while True:
            if self.executor:
                self.executor.update(dt)
            self.env.step()
    
    def _run_realtime(self, video_path: str, audio_path: str = None, analysis: dict = None):
        """
        Process video frame-by-frame in real-time (sliding window approach).
        Note: Video processing happens in run_simulation() before GUI initialization.
        
        Args:
            video_path: Path to video file (for reference, processing already done)
            audio_path: Optional path to separate audio file
            analysis: Already-processed emotion analysis dictionary
        """
        # For real-time processing, we would use a sliding window
        # This is a simplified version that processes the video in chunks
        print("Real-time processing mode (simplified chunk-based)...", flush=True)
        
        if analysis is None:
            print("Error: Analysis not provided to _run_realtime", flush=True)
            return
        
        # As in batch mode, ensure a stand-and-settle followed by a short
        # neutral freeze so the first visible behavior is a controlled reaction.
        self._neutral_freeze_before_first_reaction()

        # Execute overall reaction
        self.execute_reaction(analysis, self.env)
        
        # Main control loop with executor updates
        physics_hz = 240.0
        dt = 1.0 / physics_hz
        
        # Then play back with frame-by-frame poses
        if 'valence_timeseries' in analysis and 'arousal_timeseries' in analysis:
            valence_seq = np.array(analysis['valence_timeseries'])
            arousal_seq = np.array(analysis['arousal_timeseries'])
            
            fps = analysis.get('fps', 25.0)
            physics_steps_per_frame = int((1.0 / fps) / (1.0 / physics_hz))
            
            window_size = min(30, len(valence_seq))  # 30-frame window for trend analysis
            
            for i in range(len(valence_seq)):
                val = valence_seq[i]
                aro = arousal_seq[i]
                
                # Apply emotional pose
                self.env.emotional_pose(valence=val, arousal=aro)
                
                # Periodically re-analyze trends using sliding window
                if i >= window_size and i % window_size == 0:
                    window_valence = valence_seq[max(0, i - window_size):i]
                    window_arousal = arousal_seq[max(0, i - window_size):i]
                    
                    window_analysis = analyze_emotion_stream(
                        window_valence,
                        window_arousal,
                        calibration=self.calibration,
                        trend_threshold=self.trend_threshold,
                        volatility_threshold=self.volatility_threshold,
                        debug=False
                    )
                    
                    # Update reaction if significant change detected
                    if window_analysis['va_state_label'] != analysis['va_state_label']:
                        print(f"\n[Frame {i}] State changed to: {window_analysis['va_state_label']}", flush=True)
                        self.execute_reaction(window_analysis, self.env)
                
                # Update executor every frame
                if self.executor:
                    self.executor.update(dt)
                
                # Step physics
                for _ in range(physics_steps_per_frame):
                    self.env.step()
        
        print("\n[Simulation] Real-time processing complete. Press Ctrl+C to exit.", flush=True)
        while True:
            if self.executor:
                self.executor.update(dt)
            self.env.step()


def main():
    """Main entry point for the emotional Spot controller."""
    parser = argparse.ArgumentParser(
        description="Emotional Spot Controller - Integrate emotion recognition with Spot robot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--audio_path",
        help="Path to audio file (optional, defaults to video_path)",
        default=None
    )
    parser.add_argument(
        "--calibration",
        help="Path to calibration.json file",
        default=None
    )
    parser.add_argument(
        "--model_path",
        help="Path to JointCAM model checkpoint",
        default=None
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for inference (default: cpu)"
    )
    parser.add_argument(
        "--trend_threshold",
        type=float,
        default=None,
        help="Threshold for trend direction (uses scale-aware default if not provided)"
    )
    parser.add_argument(
        "--volatility_threshold",
        type=float,
        default=None,
        help="Threshold for high volatility (uses scale-aware default if not provided)"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable PyBullet GUI"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Process video in real-time (frame-by-frame) instead of batch"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information"
    )
    
    args = parser.parse_args()
    
    # Create controller
    controller = EmotionalSpotController(
        calibration_path=args.calibration,
        model_path=args.model_path,
        device=args.device,
        trend_threshold=args.trend_threshold,
        volatility_threshold=args.volatility_threshold,
        debug=args.debug
    )
    
    # Run simulation
    controller.run_simulation(
        video_path=args.video_path,
        audio_path=args.audio_path,
        gui=not args.no_gui,
        realtime=args.realtime
    )


if __name__ == "__main__":
    main()
