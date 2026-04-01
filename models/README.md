# models/

Place JointCAM model checkpoint files here.

| File | Description |
|---|---|
| `jointcam_finetuned_v4.pt` | **Recommended.** Fine-tuned on 100+ Aff-Wild2 videos with hybrid CCC+MSE loss. Use this by default. |
| `jointcam_model.pt` | Original model. Used as a fallback if the fine-tuned version is not present. |

Both `run_offline.py` and `run_online.py` look for models in this folder automatically,
preferring `jointcam_finetuned_v4.pt`.  You can override with `--model path/to/file.pt`.

## Source

Copy from the original project:
```
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\jointcam_finetuned_v4.pt
Spot_Mini_Simulation\spot_mini_mini\spot_bullet\src\emotion\data\jointcam_model.pt
```

These files are large (~500 MB each) and are not checked into version control.
