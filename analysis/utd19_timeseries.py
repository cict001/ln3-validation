import sys, os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\86152\Downloads\utd19_u.csv"
OUT_DIR  = os.path.join(os.path.expanduser("~"), "Downloads", "chengdu_results")
os.makedirs(OUT_DIR, exist_ok=True)

LN3     = np.log(3)
ELL_KM  = 0.30
VEH_LEN = 0.005   # 5m average vehicle length in km

print("="*65)
print("UTD19 timeseries phase transition analysis")
print("="*65)
print("Pass 1: scan all detectors, find crossing detectors...")

det_stats = {}

for chunk in pd.read_csv(CSV_PATH, chunksize=2_000_000,
                          dtype={'detid':str,'city':str,'flow':float,
                                 'occ':float,'speed':float,'error':float,
                                 'interval':float}):
    valid = chunk[
        chunk['error'].isna() &
        (chunk['occ'] >= 0) & (chunk['occ'] <= 1) &
        (chunk['flow'] >= 0) &
        chunk['speed'].notna() & (chunk['speed'] > 0)
    ].copy()
    if valid.empty:
        continue

    valid['lam_ell'] = (valid['occ'] / VEH_LEN) * ELL_KM
    valid['hour']    = (valid['interval'] % 86400) / 3600
    valid['is_night']= (valid['hour'] < 6) | (valid['hour'] >= 22)
    valid['is_day']  = (valid['hour'] >= 7) & (valid['hour'] <= 19)

    for detid, grp in valid.groupby('detid'):
        if detid not in det_stats:
            det_stats[detid] = {
                'city': grp['city'].iloc[0],
                'night_le': [], 'day_le': [],
                'night_sp': [], 'day_sp': [],
            }
        night = grp[grp['is_night']]
        day   = grp[grp['is_day']]
        det_stats[detid]['night_le'].extend(night['lam_ell'].tolist())
        det_stats[detid]['day_le'].extend(day['lam_ell'].tolist())
        det_stats[detid]['night_sp'].extend(night['speed'].tolist())
        det_stats[detid]['day_sp'].extend(day['speed'].tolist())

    print("  %.0fk valid rows  %d detectors" %
          (len(valid)/1e3, len(det_stats)), end='\r')

print("\nPass 1 done: %d detectors" % len(det_stats))

crossers = []
for detid, d in det_stats.items():
    if len(d['night_le']) < 10 or len(d['day_le']) < 10:
        continue
    nm = np.median(d['night_le'])
    dm = np.median(d['day_le'])
    if nm < LN3 and dm > LN3:
        crossers.append({
            'detid':       detid,
            'city':        d['city'],
            'night_le':    nm,
            'day_le':      dm,
            'delta_le':    dm - nm,
            'night_speed': np.median(d['night_sp']) if d['night_sp'] else 0,
            'day_speed':   np.median(d['day_sp'])   if d['day_sp']   else 0,
        })

df_cross = pd.DataFrame(crossers).sort_values('delta_le', ascending=False)
print("Crossing detectors (night<ln3<day): %d" % len(df_cross))
if len(df_cross) > 0:
    print(df_cross[['detid','city','night_le','day_le']].head(15).to_string(index=False))
else:
    print("No crossing detectors found.")
    print("Try increasing VEH_LEN (currently %.3f km)" % VEH_LEN)
    sys.exit(0)

target_dets = set(df_cross['detid'].head(30).tolist())
print("\nPass 2: detailed timeseries for top %d crossers..." % len(target_dets))

det_ts = {d: [] for d in target_dets}

for chunk in pd.read_csv(CSV_PATH, chunksize=2_000_000,
                          dtype={'detid':str,'city':str,'flow':float,
                                 'occ':float,'speed':float,'error':float,
                                 'interval':float}):
    valid = chunk[
        chunk['error'].isna() &
        chunk['detid'].isin(target_dets) &
        (chunk['occ'] >= 0) & (chunk['occ'] <= 1) &
        (chunk['flow'] >= 0) &
        chunk['speed'].notna() & (chunk['speed'] > 0)
    ].copy()
    if valid.empty:
        continue
    valid['lam_ell'] = (valid['occ'] / VEH_LEN) * ELL_KM
    valid['hour']    = (valid['interval'] % 86400) / 3600
    for detid, grp in valid.groupby('detid'):
        det_ts[detid].extend(
            zip(grp['hour'].tolist(),
                grp['lam_ell'].tolist(),
                grp['speed'].tolist()))
    print("  pass 2...", end='\r')

results = []
for detid in target_dets:
    if not det_ts[detid]:
        continue
    arr   = np.array(det_ts[detid])
    hours = arr[:, 0]
    le    = arr[:, 1]
    sp    = arr[:, 2]

    bin_le, bin_var = [], []
    for h in range(24):
        m = (hours >= h) & (hours < h+1)
        if m.sum() < 5:
            continue
        bin_le.append(float(np.median(le[m])))
        bin_var.append(float(np.var(sp[m])))

    if len(bin_le) < 8:
        continue
    bin_le  = np.array(bin_le)
    bin_var = np.array(bin_var)

    below = bin_var[bin_le <  LN3]
    above = bin_var[bin_le >= LN3]
    if len(below) < 2 or len(above) < 2:
        continue
    ratio = float(np.mean(below) / np.mean(above)) if np.mean(above) > 0 else 0
    _, p  = mannwhitneyu(below, above, alternative='greater')

    row = df_cross[df_cross['detid'] == detid].iloc[0]
    results.append({
        'detid':     detid,
        'city':      row['city'],
        'var_below': float(np.mean(below)),
        'var_above': float(np.mean(above)),
        'ratio':     ratio,
        'p':         p,
        'night_le':  row['night_le'],
        'day_le':    row['day_le'],
    })

df_res = pd.DataFrame(results).sort_values('ratio', ascending=False) if results else pd.DataFrame()

lines = []
lines.append("="*65)
lines.append("RESULTS: UTD19 timeseries phase transition")
lines.append("Prediction: var drops when lambda*ell crosses ln(3) upward")
lines.append("="*65)
lines.append("")
lines.append("Total detectors: %d" % len(det_stats))
lines.append("Crossing detectors: %d" % len(df_cross))
lines.append("Analysed: %d" % len(results))
lines.append("")

if not df_res.empty:
    lines.append("%-12s %-14s %10s %10s %8s %8s" %
                 ("detid","city","var_below","var_above","ratio","p"))
    lines.append("-"*65)
    for _, r in df_res.iterrows():
        lines.append("%-12s %-14s %10.1f %10.1f %8.3f %8.4f" % (
            str(r['detid'])[:12], str(r['city'])[:14],
            r['var_below'], r['var_above'], r['ratio'], r['p']))
    lines.append("")
    sig = df_res[(df_res['ratio'] > 1.2) & (df_res['p'] < 0.05)]
    lines.append("Significant results (ratio>1.2 and p<0.05): %d" % len(sig))
    if len(sig) > 0:
        lines.append("PHASE TRANSITION EVIDENCE FOUND!")
        lines.append("Cities: %s" % ', '.join(sig['city'].unique()[:5].tolist()))
    else:
        ratio_gt1 = (df_res['ratio'] > 1.0).sum()
        lines.append("Detectors with ratio>1 (correct direction): %d / %d" %
                     (ratio_gt1, len(df_res)))
else:
    lines.append("No results.")

txt = '\n'.join(lines)
print('\n' + txt)

with open(os.path.join(OUT_DIR, "utd19_timeseries.txt"), 'w', encoding='utf-8') as f:
    f.write(txt)
if not df_res.empty:
    df_res.to_csv(os.path.join(OUT_DIR, "utd19_timeseries.csv"), index=False)
df_cross.to_csv(os.path.join(OUT_DIR, "utd19_crossers.csv"), index=False)
print("\nSaved to:", OUT_DIR)
