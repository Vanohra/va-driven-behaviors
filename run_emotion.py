#!/usr/bin/env python3
"""
Quick start script for emotion pipeline integration.

Usage:
    python run_emotion.py <video_path> [--no-gui] [--realtime] [--debug]
"""

import sys
import os
from pathlib import Path

# Add spot_mini_mini to path
project_root = Path(__file__).parent
spot_mini_path = project_root / "spot_mini_mini"
sys.path.insert(0, str(spot_mini_path))

# Import controller
from spot_bullet.src.emotion.integration import EmotionalSpotController

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_emotion.py <video_path> [--no-gui] [--realtime] [--debug] [--rl-agent] [--agent-num N]")
        print("\nOptions:")
        print("  --no-gui     Run without PyBullet GUI")
        print("  --realtime   Process video frame-by-frame in real-time")
        print("  --debug      Print debug information")
        print("  --rl-agent   Use pretrained RL agent for intelligent locomotion (maintains balance)")
        print("  --agent-num N  Agent number to load (default: 2229, only used with --rl-agent)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    gui = "--no-gui" not in sys.argv
    realtime = "--realtime" in sys.argv
    debug = "--debug" in sys.argv
    use_rl_agent = "--rl-agent" in sys.argv
    
    # Parse agent number
    agent_num = 2229  # Default
    if "--agent-num" in sys.argv:
        try:
            idx = sys.argv.index("--agent-num")
            if idx + 1 < len(sys.argv):
                agent_num = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            print("Warning: Invalid agent number, using default 2229")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Set up paths
    data_dir = spot_mini_path / "spot_bullet" / "src" / "emotion" / "data"
    calibration_path = data_dir / "calibration.json"
    model_path = data_dir / "jointcam_model.pt"
    
    # Check if files exist
    if not calibration_path.exists():
        print(f"Warning: Calibration file not found: {calibration_path}")
        print("Will try to use default or fallback calibration.")
        calibration_path = None
    
    if not model_path.exists():
        print(f"Warning: Model file not found: {model_path}")
        print("Will try to use default model path.")
        model_path = None
    
    print("=" * 60)
    print("Emotion Pipeline - Spot Robot Controller")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"GUI: {gui}")
    print(f"Realtime: {realtime}")
    print(f"Debug: {debug}")
    print(f"RL Agent: {use_rl_agent}")
    if use_rl_agent:
        print(f"Agent Number: {agent_num}")
    print("=" * 60)
    print()
    
    # Create controller
    try:
        controller = EmotionalSpotController(
            calibration_path=str(calibration_path) if calibration_path else None,
            model_path=str(model_path) if model_path else None,
            device="cpu",  # Change to "cuda" if you have GPU
            debug=debug
        )
        
        # Run simulation
        controller.run_simulation(
            video_path=video_path,
            gui=gui,
            realtime=realtime,
            use_rl_agent=use_rl_agent,
            agent_num=agent_num
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
