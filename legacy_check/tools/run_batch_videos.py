# run_batch_videos.py - pipeline only, no PyBullet
import os, sys, json
from pathlib import Path
# Resolve project root dynamically
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE / 'data'
CAL = DATA / 'calibration.json'

# Add project root to path for imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(str(HERE))
with open(CAL) as f:
    cal = json.load(f)
for k in ('valence','arousal'):
    d = cal[k]
    if 'std' not in d: d['std'] = 0.15
    if 'mad' not in d: d['mad'] = d.get('std', 0.15)
    for p in ('p10','p25','p50','p75','p90'):
        if p not in d and 'p30' in d:
            if p=='p50': d[p] = (d['p30']+d['p70'])/2
            elif p in ('p10','p25'): d[p] = d['p30'] - 0.02
            else: d[p] = d['p70'] + 0.02
from test_emotions import load_model, extract_video_features, extract_audio_features, align_features, predict_emotions
from core.emotion_analyzer import analyze_emotion_stream
from core.intent_selector import IntentSelector
import numpy as np
def ccc(a,b):
    a,b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a)!=len(b) or len(a)<2: return float('nan')
    mt,mp = np.mean(a),np.mean(b)
    cov = np.mean((a-mt)*(b-mp))
    d = np.var(a)+np.var(b)+(mt-mp)**2
    return 2*cov/d if d>0 else float('nan')
def sm(x,w=5):
    x = np.asarray(x,dtype=float).ravel()
    if len(x)<w: return x.copy()
    return np.convolve(x, np.ones(w)/w, mode='same')
mp = DATA / 'jointcam_finetuned_v4.pt'
if not mp.exists(): mp = DATA / 'jointcam_model.pt'
print('Model:', mp, flush=True)
model = load_model(str(mp), 'cpu')
sel = IntentSelector()
vids = sorted(DATA.glob('video*.mp4'))
cv, ca = [], []
for vp in vids:
    print('='*72, flush=True)
    print('VIDEO:', vp.name, flush=True)
    print('='*72, flush=True)
    vf,_ = extract_video_features(str(vp), 'cpu')
    af = extract_audio_features(str(vp))
    va,aa = align_features(vf, af)
    v,a = predict_emotions(model, va, aa, 'cpu')
    v,a = v.flatten(), a.flatten()
    cv.append(ccc(v, sm(v))); ca.append(ccc(a, sm(a)))
    an = analyze_emotion_stream(v, a, calibration=cal, export_timeseries=False, debug=False)
    vol = max(an.get('valence_volatility',0), an.get('arousal_volatility',0))
    conf = max(0.3, min(1.0, 1.0 - vol/0.5))
    intent, pm, _ = sel.select_intent(an['valence'], an['arousal'], vol, conf, an['va_state_label'])
    print('Calibration baselines (calibration.json) p30/p70:', flush=True)
    print('  valence p30=%.4f p70=%.4f' % (cal['valence']['p30'], cal['valence']['p70']), flush=True)
    print('  arousal p30=%.4f p70=%.4f' % (cal['arousal']['p30'], cal['arousal']['p70']), flush=True)
    print('Session baselines: valence=%.6f arousal=%.6f' % (an['valence'], an['arousal']), flush=True)
    print('Trends: valence %s delta=%.6f | arousal %s delta=%.6f' % (an['valence_direction'], an['valence_delta'], an['arousal_direction'], an['arousal_delta']), flush=True)
    print('State:', an['va_state_label'], '| Intent:', intent.value, '| pose_mode:', pm, flush=True)
    print('Reaction:', an['reaction_recommendation'][:200], flush=True)
    print('CCC (raw vs smoothed): valence=%.4f arousal=%.4f' % (cv[-1], ca[-1]), flush=True)
print('='*72, flush=True)
print('FINAL CCC (mean over all videos)', flush=True)
print('  Mean CCC valence:  %.4f' % np.nanmean(cv), flush=True)
print('  Mean CCC arousal: %.4f' % np.nanmean(ca), flush=True)
print('  Overall mean CCC: %.4f' % ((np.nanmean(cv)+np.nanmean(ca))/2), flush=True)
