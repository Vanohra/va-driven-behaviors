# samples/

Place test video files here for use with `run_offline.py`.

## Sample videos from Aff-Wild2

The original project includes 10 sample videos from the Aff-Wild2 dataset.
Copy any of these here for testing:

```
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video28.mp4
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video38.mp4
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video44.mp4
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video46.mp4
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video64.mp4
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video67.mp4
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video71.mp4
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video83.mp4
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video87.mp4
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\video90.mp4
```

## Usage

```bash
python run_offline.py samples/video67.mp4
python run_offline.py samples/video67.mp4 --debug
```

## Notes on sample video selection

- `video67.mp4` is a good starting point — typical expression range, clean audio.
- Use `--debug` flag to inspect per-stage outputs if results look unexpected.
- Sample videos are Aff-Wild2 clips; the calibration.json was computed from a
  subset of these, so they represent in-distribution inputs.
