#!/usr/bin/env python3
"""
UTD19 批量分析脚本（针对 utd19_u.csv 列结构定制）
列: day, interval, detid, flow, occ, error, city, speed

原理:
  occ (占有率) ∝ 密度，故 occ_c/occ_j = k_c/k_j，与 Harvard FITS 分析完全等价
  无需 speed 数据

用法:
  python analyze_UTD19_final.py "C:\\Users\\86152\\Downloads\\utd19_u.csv"
"""

import sys, os
import pandas as pd
import numpy as np
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ─── ln(3) 理论 ──────────────────────────────────────────────────────────────
LN3  = np.log(3)
PRED = (1/(1+LN3))**(1/LN3)   # 0.5093

def ratio_from_keff(k):
    if k <= 1.0: return 0.0
    lk = np.log(k)
    return float((1/(1+lk))**(1/lk))

def infer_r(ratio_obs):
    """精确非线性反推红灯占比"""
    if ratio_obs >= PRED: return 0.0
    if ratio_obs <= 0.001: return None
    for r_max in [0.89, 0.93, 0.97, 0.995]:
        try:
            lo = ratio_from_keff(3*(1-1e-4)) - ratio_obs
            hi = ratio_from_keff(3*(1-r_max)) - ratio_obs
            if lo * hi < 0:
                return float(brentq(
                    lambda r: ratio_from_keff(3*(1-r)) - ratio_obs,
                    1e-4, r_max, xtol=1e-7))
        except: pass
    return None

# ─── 单检测器分析（基于 occ，与 Harvard FITS 完全等价）─────────────────────
def analyze_detector(occ, flow, min_obs=100, n_bins=30, min_occ_max=0.30):
    """
    occ:  占有率数组 (0-1)，作为密度代理
    flow: 流量数组（用于识别流量峰值）
    """
    mask = (occ > 0) & (occ < 1.0) & (flow >= 0) & np.isfinite(occ) & np.isfinite(flow)
    o, fl = occ[mask], flow[mask]
    if len(o) < min_obs: return None

    occ_max = float(np.percentile(o, 99))
    if occ_max < min_occ_max: return None  # 未达到真正拥堵

    # 分箱找流量峰值 → occ_c
    be = np.linspace(0, occ_max, n_bins + 1)
    bo, bf = [], []
    for i in range(n_bins):
        m = (o >= be[i]) & (o < be[i+1])
        if m.sum() >= 3:
            bo.append((be[i]+be[i+1])/2)
            bf.append(float(fl[m].mean()))
    if len(bo) < 6: return None

    bo, bf = np.array(bo), np.array(bf)
    pi       = bf.argmax()
    occ_c    = float(bo[pi])
    flow_pk  = float(bf[pi])

    # occ_j: 低流量、高占有率区
    jam_m = (fl < 0.15 * flow_pk) & (o > occ_c)
    occ_j = float(np.percentile(o[jam_m], 50)) if jam_m.sum() >= 20 else occ_max

    if occ_j <= occ_c: return None

    ratio = occ_c / occ_j
    r_inf = infer_r(ratio)
    return dict(ratio=ratio, occ_c=occ_c, occ_j=occ_j, occ_max=occ_max,
                n_obs=int(len(o)), r_inferred=r_inf,
                r_pct=r_inf*100 if r_inf else 0.0,
                above_baseline=(ratio >= PRED))

# ─── 主程序 ──────────────────────────────────────────────────────────────────
def main():
    fpath = sys.argv[1] if len(sys.argv) > 1 else "utd19_u.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "UTD19_results"
    os.makedirs(out_dir, exist_ok=True)

    fsize_gb = os.path.getsize(fpath)/1e9
    print(f"文件: {os.path.basename(fpath)}  ({fsize_gb:.2f} GB)")
    print(f"分块读取中（每块 50万行，内存友好）...")
    print()

    # 按城市+检测器累积 occ 和 flow
    # 结构: buf[city][detid] = {'occ': list, 'flow': list}
    buf = {}
    total_rows = 0
    valid_rows = 0
    chunk_n = 0

    CHUNKSIZE = 500_000

    reader = pd.read_csv(
        fpath,
        chunksize=CHUNKSIZE,
        usecols=['detid','flow','occ','error','city'],
        dtype={'detid': str, 'city': str, 'flow': float,
               'occ': float, 'error': float},
    )

    for chunk in reader:
        chunk_n += 1
        total_rows += len(chunk)

        # 过滤: error 为 null 才是有效数据
        valid = chunk[chunk['error'].isna()].copy()
        # 也过滤 occ 异常
        valid = valid[(valid['occ'] >= 0) & (valid['occ'] <= 1.0)]
        valid_rows += len(valid)

        for city, cgrp in valid.groupby('city'):
            if city not in buf:
                buf[city] = {}
            for detid, dgrp in cgrp.groupby('detid'):
                if detid not in buf[city]:
                    buf[city][detid] = {'occ': [], 'flow': []}
                buf[city][detid]['occ'].extend(dgrp['occ'].tolist())
                buf[city][detid]['flow'].extend(dgrp['flow'].tolist())

        cities_so_far = len(buf)
        dets_so_far   = sum(len(v) for v in buf.values())
        print(f"  chunk {chunk_n:3d}: {total_rows/1e6:6.1f}M 行已读  "
              f"有效率={valid_rows/total_rows*100:.0f}%  "
              f"城市={cities_so_far}  检测器={dets_so_far}", end='\r')

    print(f"\n\n读取完毕: {total_rows/1e6:.1f}M 行  有效 {valid_rows/1e6:.1f}M 行  "
          f"城市={len(buf)}  检测器={sum(len(v) for v in buf.values())}")

    # ── 分析每个检测器 ────────────────────────────────────────────────────────
    print("\n分析检测器...")
    city_summaries = []
    all_det_rows   = []

    for city in sorted(buf.keys()):
        dets = buf[city]
        city_results = []
        for detid, data in dets.items():
            occ  = np.array(data['occ'],  dtype=float)
            flow = np.array(data['flow'], dtype=float)
            res  = analyze_detector(occ, flow)
            if res:
                res['detid'] = detid
                res['city']  = city
                city_results.append(res)
                all_det_rows.append(res)

        if not city_results:
            continue

        ratios  = np.array([r['ratio'] for r in city_results])
        med     = float(np.median(ratios))
        std     = float(np.std(ratios))
        r_med   = infer_r(med)
        n_below = int((ratios < PRED).sum())
        n_above = int((ratios >= PRED).sum())
        n       = len(ratios)

        # 道路类型自动判断
        if   n_below/n > 0.60: road_class = 'urban_signalised'
        elif n_above/n > 0.60: road_class = 'freeway'
        else:                   road_class = 'mixed'

        city_summaries.append(dict(
            city=city, n_total=len(dets), n_valid=n,
            median_ratio=med, std_ratio=std,
            r_pct=r_med*100 if r_med else 0.0,
            r_inferred=r_med,
            deviation_pct=(med-PRED)/PRED*100,
            n_below=n_below, n_above=n_above,
            road_class=road_class,
        ))

    # ── 写输出 ────────────────────────────────────────────────────────────────
    # 城市汇总 CSV
    sum_path = os.path.join(out_dir, 'UTD19_city_summary.csv')
    pd.DataFrame(city_summaries).sort_values('median_ratio').to_csv(sum_path, index=False)
    print(f"城市汇总 → {sum_path}")

    # 全检测器明细 CSV
    det_path = os.path.join(out_dir, 'UTD19_all_detectors.csv')
    pd.DataFrame(all_det_rows).sort_values(['city','ratio']).to_csv(det_path, index=False)
    print(f"检测器明细 → {det_path}")

    # ── 控制台结果表 ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"UTD19 ln(3) 验证结果   基准: {PRED:.4f}（高速公路理论值）")
    print(f"{'='*72}")
    print(f"{'城市':<24} {'N':>5} {'k_c/k_j':>9} {'r':>7} {'偏差%':>7}  类型")
    print(f"{'-'*72}")
    print(f"{'Highway (theory)':<24} {'—':>5} {PRED:>9.4f} {'0%':>7} {'Ref':>7}")

    for row in sorted(city_summaries, key=lambda x: x['median_ratio']):
        r_s = f"{row['r_pct']:.1f}%"
        marker = '▼' if row['road_class']=='urban_signalised' else ('▲' if row['road_class']=='freeway' else '◆')
        print(f"{row['city']:<24} {row['n_valid']:>5} "
              f"{row['median_ratio']:>9.4f} {r_s:>7} "
              f"{row['deviation_pct']:>+6.1f}%  {marker} {row['road_class']}")

    print(f"\n▼=有信号城市路段  ▲=无信号快速路  ◆=混合")

    # 关键统计
    freeway_cities = [r for r in city_summaries if r['road_class']=='freeway']
    urban_cities   = [r for r in city_summaries if r['road_class']=='urban_signalised']
    all_below = sum(1 for r in city_summaries if r['median_ratio'] < PRED)
    print(f"\n城市总数: {len(city_summaries)}")
    print(f"快速路类城市 (k_c/k_j≥0.5093): {len(freeway_cities)}")
    print(f"有信号城市  (k_c/k_j<0.5093): {len(urban_cities)}")
    print(f"中位数低于基准的城市: {all_below}/{len(city_summaries)}")
    if freeway_cities:
        fw_med = np.median([r['median_ratio'] for r in freeway_cities])
        print(f"快速路城市 k_c/k_j 中位数: {fw_med:.4f}")
    if urban_cities:
        ur_med = np.median([r['median_ratio'] for r in urban_cities])
        print(f"有信号城市 k_c/k_j 中位数: {ur_med:.4f}")

if __name__ == '__main__':
    main()
