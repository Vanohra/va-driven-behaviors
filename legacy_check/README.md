# legacy_check/

These files were part of the original `Spot_Mini_Simulation` project but are
**not required** for the current VA pipeline (offline or online).

They are kept here so they can be referenced or recovered if needed.
They are **not imported** by any code in this project.

---

## simulation/

These are the PyBullet-based simulation components from the original pipeline.
They require:
- `pybullet`
- `gym`
- `spotmicro` (the full simulation framework)
- A working PyBullet installation with GUI support

| File | Original role |
|---|---|
| `emotional_controller.py` | Main offline controller — loaded model, ran simulation, executed reactions |
| `new_sim_adapter.py` | Wrapped `spotBezierEnv`; mapped emotion intents to joint positions and BezierStepper locomotion |
| `rl_agent_adapter.py` | Used a pretrained ARS RL agent for stable locomotion while applying VA modulation |
| `behavior_renderer.py` | Rendered `ReactionAction` objects as visible robot behaviors (pose + overlays + locomotion) |
| `reaction_executor.py` | State-machine executor that managed smooth blending between reactions |

### Why they were removed from the main pipeline

The current goal is a **terminal-output VA pipeline** that does not depend on
a physics simulator.  The core analysis logic (preprocessing → baseline →
trends → volatility → classification → reaction mapping) is fully preserved in
`pipeline/` and `online/` without any simulation dependency.

If you want to re-enable simulation:
1. Install PyBullet, gym, and the spotmicro dependencies.
2. Copy these files back to a suitable location.
3. Use `emotional_controller.py` as the entry point with `run_emotion.py`
   from the original `Spot_Mini_Simulation` project.

---

## tools/

| File | Description |
|---|---|
| `run_batch_videos.py` | Batch-processes a directory of videos through the offline pipeline and prints CCC metrics. Useful for evaluating model performance on multiple clips. |

### Using run_batch_videos.py

This script still works but requires the emotion-poc path to be configured.
Before running it, add the project root to your Python path and update the
hardcoded paths inside the script:

```bash
cd "[PROJECT_ROOT]"
python legacy_check/tools/run_batch_videos.py
```

---

## Not included here (in original Spot_Mini_Simulation)

The following were also excluded and are not copied here at all because they
are entirely unrelated to the VA pipeline:

- **spotmicro/** — full PyBullet Spot simulation (3D assets, kinematics, gym env)
- **ars_lib/, sac_lib/, td3_lib/, tg_lib/** — RL training frameworks
- **spot_real/** — real robot hardware interfaces (Teensy, RPi, servos)
- **mini_ros/** — deprecated ROS package
- **mini_bullet/** — legacy Minitaur framework
- **Bittle_X_Handover/** — experimental Bittle robot integration (not used in main pipeline)
- **paper/** — data collection / plotting scripts for RL training paper
