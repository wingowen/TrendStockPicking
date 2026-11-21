#!/usr/bin/env python3
"""横截面因子扫描台：换一个范式找 alpha（月度调仓的横截面选股），并与突破信号交叉验证。

与 v3（事件驱动的 20 日突破）完全正交的设计：
  每月最后一个交易日做一次全市场截面排序，取前 10 只等额持有下一个月，
  月内不交易、不设止损。评估口径与 v3 一致：
    训练 2020-01→2022-01 拟合，2022-01 之后冻结验证；成本 0.4%/次换手。

因子（全部只用当日及以前数据）：
  mom20 / mom60 / mom120-1（12-1 动量，跳过最近 5 日）
  rev5（5 日反转）
  vol20（20 日波动率，取负 = 低波）
  amt_trend（近 5 日成交额 / 近 60 日成交额，量能放大）
  dist_high250（距 250 日最高收盘的距离，越接近新高越好）
  ma_gap（收盘 / MA60 - 1，趋势位置）

交叉验证的三条口径（本脚本全部输出）：
  1) 时间分折：2020-21 / 2022-23 / 2024-26 三段独立算 IC
  2) 独立标签：同一因子分别对 20 日前瞻与 60 日前瞻做 IC，看符号是否一致
  3) 与突破信号交叉：v3 的 20430 个突破信号落在因子五分位的哪一档

    ../.venv-bt/bin/python alt_scans.py     # 约 1-2 分钟，输出 alt_scans.json / alt_scans.html
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DB = ROOT / "market.db"
OUT_JSON = ROOT / "alt_scans.json"
OUT_HTML = ROOT / "alt_scans.html"

TRAIN_END = "2022-01-01"
MIN_AMT20 = 5e7       # 日均成交额门槛，剔除流动性太差的票
COST = 0.004
TOP_N = 10
MIN_HISTORY = 250


def load_matrix():
    """读全市场后复权收盘价、成交额，对齐成 (交易日 × 代码) 矩阵。"""
    con = sqlite3.connect(DB)
    cur = con.cursor()
    dates = [r[0] for r in cur.execute(
        "select distinct date from bars where code not like 'IDX%' order by date")]
    codes = [r[0] for r in cur.execute(
        "select distinct code from bars where code not like 'IDX%' order by code")]
    tmap = {d: i for i, d in enumerate(dates)}
    cmap = {c: j for j, c in enumerate(codes)}
    T, N = len(dates), len(codes)
    close = np.full((T, N), np.nan, dtype=np.float32)
    amt = np.full((T, N), np.nan, dtype=np.float32)

    cur.execute("select code, date, close, amount from bars where code not like 'IDX%'")
    while True:
        rows = cur.fetchmany(400_000)
        if not rows:
            break
        ti = np.fromiter((tmap[r[1]] for r in rows), dtype=np.int32, count=len(rows))
        ci = np.fromiter((cmap[r[0]] for r in rows), dtype=np.int32, count=len(rows))
        close[ti, ci] = np.fromiter((r[2] for r in rows), dtype=np.float32, count=len(rows))
        amt[ti, ci] = np.fromiter((r[3] for r in rows), dtype=np.float32, count=len(rows))

    bench = {}
    for name in ("IDX_沪深300", "IDX_上证指数"):
        rows = cur.execute("select date, close from bars where code=? order by date", (name,)).fetchall()
        s = np.full(T, np.nan)
        for d, v in rows:
            if d in tmap:
                s[tmap[d]] = v
        bench[name.split("_")[1]] = s
    con.close()
    return dates, codes, close, amt, bench


def month_ends(dates):
    """每个月最后一个交易日在 dates 里的下标。"""
    out = []
    for i in range(1, len(dates)):
        if dates[i][:7] != dates[i - 1][:7]:
            out.append(i - 1)
    return out


def factors_at(i, close, amt):
    """在第 i 个交易日截面算因子；只用 i 及以前的数据。"""
    c = close[: i + 1]
    a = amt[: i + 1]
    last = c[-1]
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["mom20"] = last / c[-21] - 1 if len(c) > 21 else None
        out["mom60"] = last / c[-61] - 1 if len(c) > 61 else None
        out["mom120_1"] = c[-6] / c[-126] - 1 if len(c) > 126 else None
        out["rev5"] = -(last / c[-6] - 1) if len(c) > 6 else None
        rets = c[-21:] / c[-22:-1] - 1 if len(c) > 22 else None
        out["vol20"] = -np.nanstd(rets, axis=0) if rets is not None else None
        out["amt_trend"] = np.nanmean(a[-5:], axis=0) / np.nanmean(a[-60:], axis=0) \
            if len(a) > 60 else None
        out["dist_high250"] = last / np.nanmax(c[-250:], axis=0) - 1 if len(c) >= 250 else None
        ma60 = np.nanmean(c[-60:], axis=0) if len(c) >= 60 else None
        out["ma_gap"] = last / ma60 - 1 if ma60 is not None else None
    return out


def eligible_mask(i, close, amt, fwd):
    """可交易域：有足够历史、当日有价、成交额达标、前瞻收益存在。"""
    c = close[: i + 1]
    a = amt[: i + 1]
    with np.errstate(invalid="ignore"):
        valid_hist = np.sum(~np.isnan(c), axis=0) >= MIN_HISTORY
        amt20 = np.nanmean(a[-20:], axis=0)
    ok = valid_hist & np.isfinite(close[i]) & (amt20 > MIN_AMT20) & np.isfinite(fwd)
    return ok


def rank_pct(x, ok):
    """在可交易域内把因子转成 [0,1] 分位；域外 NaN。"""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    v = x[ok]
    if v.size < 20:
        return out
    order = np.argsort(v)
    ranks = np.empty(v.size)
    ranks[order] = np.arange(v.size)
    out[ok] = ranks / (v.size - 1)
    return out


def spearman(x, y, ok):
    a, b = x[ok], y[ok]
    if a.size < 30:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else np.nan


def tstat(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if xs.size < 3:
        return np.nan
    return float(xs.mean() / (xs.std(ddof=1) / np.sqrt(xs.size)))


def ic_series(fvals, fwd, ok_list):
    return [spearman(fvals[k], fwd[k], ok_list[k]) for k in range(len(fwd))]


def quintile_spread(fvals, fwd, ok_list, nq=5):
    """每期五分位的前瞻超额收益，按分位汇总平均。"""
    buckets = [[] for _ in range(nq)]
    for k in range(len(fwd)):
        ok = ok_list[k]
        f, y = fvals[k], fwd[k]
        v = f[ok]
        if v.size < 100:
            continue
        qs = np.argsort(np.argsort(v)) * nq // v.size
        yy = y[ok]
        base = np.nanmean(yy)
        for q in range(nq):
            buckets[q].append(float(np.nanmean(yy[qs == q] - base)) if (qs == q).any() else np.nan)
    return [float(np.nanmean(b)) if len(b) else np.nan for b in buckets]


def portfolio(fvals, dates, close, ok_list, idx_list, top_n=TOP_N, hold=1, cost=COST, offset=0):
    """按分数取前 top_n，每 hold 个月重排一次，等额持有；新进仓位按 cost 计成本。"""
    eq = [1.0]
    prev = set()          # 上一轮已持有的票（跨 hold 期累计）
    hold_rows = []
    for k in range(offset, len(idx_list) - hold, hold):
        i = idx_list[k]
        j = idx_list[k + hold]
        f = np.where(ok_list[k], fvals[k], -np.inf)
        top = [t for t in np.argsort(-f)[:top_n] if np.isfinite(f[t])]
        legs = []
        for t in top:
            g = close[j, t] / close[i, t] - 1
            legs.append(g if t in prev else g - cost)
        chosen = set(top)
        turned = len(chosen - prev) / max(len(chosen), 1)
        r = float(np.nanmean(legs)) if legs else 0.0
        eq.append(eq[-1] * (1 + r))
        hold_rows.append({"date": dates[j], "ret": r, "n": len(chosen), "turnover": turned})
        prev |= chosen     # 持有期内仍保留旧仓位
    return (np.asarray(eq), [dates[idx_list[0]]] + [h["date"] for h in hold_rows], hold_rows)


def metrics(eq, dates, bench_series, bench_dates):
    """累计/年化/回撤/Sharpe；与指数对齐到同一批日期。"""
    r = eq[1:] / eq[:-1] - 1
    years = len(r) / 12.0
    total = eq[-1] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(12)) if r.std(ddof=1) else np.nan
    return {"total": float(total), "cagr": float(cagr), "mdd": dd, "sharpe": sharpe,
            "months": int(len(r)), "win": float((r > 0).mean())}


def fold_of(d):
    y = int(d[:4])
    if y <= 2021:
        return "2020-2021 训练"
    if y <= 2023:
        return "2022-2023 验证A"
    return "2024-2026 验证B"


def main():
    dates, codes, close, amt, bench = load_matrix()
    idx_list = [i for i in month_ends(dates) if i >= MIN_HISTORY]
    idx_list = [i for i in idx_list if i + 20 < len(dates)]
    print(f"[data] {len(dates)} 交易日 × {len(codes)} 只 · {len(idx_list)} 个月末截面")

    fwd20, fwd60, oks = [], [], []
    for i in idx_list:
        with np.errstate(invalid="ignore", divide="ignore"):
            f20 = close[min(i + 20, len(dates) - 1)] / close[i] - 1
            f60 = close[min(i + 60, len(dates) - 1)] / close[i] - 1
        fwd20.append(f20); fwd60.append(f60)
        oks.append(eligible_mask(i, close, amt, f20))

    facs = {name: [] for name in
            ("mom20", "mom60", "mom120_1", "rev5", "vol20", "amt_trend", "dist_high250", "ma_gap")}
    for i in idx_list:
        f = factors_at(i, close, amt)
        for k, arr in f.items():
            facs[k].append(np.full(len(codes), np.nan) if arr is None else np.asarray(arr, float))

    md = [dates[i] for i in idx_list]
    train_k = [k for k, d in enumerate(md) if d < TRAIN_END]
    oos_k = [k for k, d in enumerate(md) if d >= TRAIN_END]

    rows = {}
    for name, series in facs.items():
        ic_tr = [spearman(series[k], fwd20[k], oks[k]) for k in train_k]
        ic_os = [spearman(series[k], fwd20[k], oks[k]) for k in oos_k]
        ic_os60 = [spearman(series[k], fwd60[k], oks[k]) for k in oos_k]
        q_os = quintile_spread([series[k] for k in oos_k],
                               [fwd20[k] for k in oos_k], [oks[k] for k in oos_k])
        rows[name] = {
            "ic_train": float(np.nanmean(ic_tr)), "t_train": tstat(ic_tr),
            "ic_oos20": float(np.nanmean(ic_os)), "t_oos20": tstat(ic_os),
            "ic_oos60": float(np.nanmean(ic_os60)), "t_oos60": tstat(ic_os60),
            "same_sign": bool(np.sign(np.nanmean(ic_os)) == np.sign(np.nanmean(ic_os60))
                              and np.isfinite(np.nanmean(ic_os))),
            "q_oos": q_os,
            "folds": {f: float(np.nanmean([spearman(series[k], fwd20[k], oks[k])
                                           for k in range(len(md)) if fold_of(md[k]) == f]))
                      for f in ("2020-2021 训练", "2022-2023 验证A", "2024-2026 验证B")},
        }

    # 只用训练期 IC 定权重（符号 + 幅度），OOS 冻结
    w = {n: rows[n]["ic_train"] for n in facs if np.isfinite(rows[n]["ic_train"])}
    w = {n: v for n, v in w.items() if abs(v) > 0.005}
    combo = []
    for k in range(len(md)):
        ok = oks[k]
        score = np.zeros(len(codes))
        for n, wt in w.items():
            rp = rank_pct(facs[n][k], ok)
            score += np.nan_to_num(rp - 0.5) * wt
        score[~ok] = np.nan
        combo.append(score)
    ic_combo_tr = [spearman(combo[k], fwd20[k], oks[k]) for k in train_k]
    ic_combo_os = [spearman(combo[k], fwd20[k], oks[k]) for k in oos_k]
    ic_combo_os60 = [spearman(combo[k], fwd60[k], oks[k]) for k in oos_k]
    ic_combo_folds = {f: float(np.nanmean([spearman(combo[k], fwd20[k], oks[k])
                                          for k in range(len(md)) if fold_of(md[k]) == f]))
                      for f in ("2020-2021 训练", "2022-2023 验证A", "2024-2026 验证B")}
    q_combo = quintile_spread([combo[k] for k in oos_k],
                              [fwd20[k] for k in oos_k], [oks[k] for k in oos_k])

    eq_all, d_all, holds_all = portfolio(combo, dates, close, oks, idx_list)
    def slice_port(eq, d):
        start = next((t for t, x in enumerate(d) if x >= TRAIN_END), None)
        if start is None:
            return None
        sub = np.concatenate([[1.0], np.asarray(eq[start:]) / eq[start]])
        return metrics(sub, d[start:], bench, dates)

    bench_ret = {}
    for name, s in bench.items():
        b0 = next((t for t, x in enumerate(dates) if x >= "2020-01-01"), 0)
        b1 = len(dates) - 1
        bench_ret[name] = {"total": float(s[b1] / s[b0] - 1),
                           "cagr": float((s[b1] / s[b0]) ** (1 / ((b1 - b0) / 244)) - 1)}
    oos_start_idx = next(i for i in idx_list if dates[i] >= TRAIN_END)
    oos_eq = eq_all[len(d_all) - len(d_all)]  # placeholder

    bt = [k for k in holds_all if k["date"] >= TRAIN_END]
    total_oos = 1.0
    for k in bt:
        total_oos *= 1 + k["ret"]
    all_total = eq_all[-1] - 1
    n_o = len(bt)
    win_o = float(np.mean([k["ret"] > 0 for k in bt])) if bt else None
    turn_o = float(np.mean([k["turnover"] for k in holds_all]))

    # 与突破信号交叉：v3 信号落在组合分的哪一档（用信号日前最近月末截面分，无未来函数）
    cross = {}
    recs = []          # (signal_date, code, q, exc20)
    try:
        import csv
        with open(ROOT / "features.csv", encoding="utf-8-sig") as f:
            sig = list(csv.DictReader(f))
        code_pos = {c: i for i, c in enumerate(codes)}
        for r in sig:
            sd = r["signal_date"]
            k = max((t for t, d in enumerate(md) if d <= sd), default=None)
            if k is None:
                continue
            j2 = code_pos.get(r["code"])
            if j2 is None or not oks[k][j2] or not np.isfinite(combo[k][j2]):
                continue
            try:
                exc = float(r["fwd20"]) - float(r["bench20"])
            except (TypeError, ValueError):
                continue
            v = combo[k][oks[k]]
            q = int(np.searchsorted(np.quantile(v, [0.2, 0.4, 0.6, 0.8]), combo[k][j2])) + 1
            recs.append((sd, r["code"], q, exc))

        def cluster_t(groups):
            """按 signal_date 聚类的 t 值。"""
            mean = float(np.mean([x[3] for x in groups]))
            per = {}
            for sd, _c, _q, exc in groups:
                per.setdefault(sd, []).append(exc - mean)
            se = np.sqrt(sum(sum(v) ** 2 for v in per.values())) / len(groups)
            return mean, (mean / se if se > 0 else np.nan)

        cross = {}
        for q in range(1, 6):
            g = [r for r in recs if r[2] == q]
            if not g:
                continue
            m, t = cluster_t(g)
            cross[q] = {"n": len(g), "exc20": m, "t": t}
        hi = [r for r in recs if r[2] >= 4]
        lo = [r for r in recs if r[2] <= 2]
        m_hi, t_hi = cluster_t(hi) if hi else (np.nan, np.nan)
        m_lo, t_lo = cluster_t(lo) if lo else (np.nan, np.nan)
        cross["hi_vs_lo"] = {"exc20": m_hi - m_lo,
                             "t": (m_hi - m_lo) / np.sqrt(
                                 (np.mean([r[3] for r in hi]) / t_hi) ** 2 +
                                 (np.mean([r[3] for r in lo]) / t_lo) ** 2)
                             if t_hi and t_lo else np.nan}
        # 落盘给引擎层交叉验证用（combo 分位 + 连续分）
        with open(ROOT / "combo_map.csv", "w", encoding="utf-8") as f:
            f.write("signal_date,code,q,score\n")
            for sd, c, q, _e in recs:
                k = max((t for t, d in enumerate(md) if d <= sd))
                f.write(f"{sd},{c},{q},{combo[k][code_pos[c]]:.6f}\n")
    except FileNotFoundError:
        cross = {}

    variants = {}
    for top_n, hold, name in ((10, 1, "10仓·月调"), (10, 1, "10仓·月调·零成本"),
                              (30, 3, "30仓·季调"),
                              (50, 6, "50仓·半年调"), (30, 3, "30仓·季调·零成本"),
                              (30, 3, "30仓·季调·起点+1月"), (30, 3, "30仓·季调·起点+2月")):
        cst = 0.0 if "零成本" in name else COST
        off = 1 if "起点+1月" in name else (2 if "起点+2月" in name else 0)
        eqv, dv, hv = portfolio(combo, dates, close, oks, idx_list,
                                top_n=top_n, hold=hold, cost=cst, offset=off)
        bt = [h for h in hv if h["date"] >= TRAIN_END]
        tot = 1.0
        for h in bt:
            tot *= 1 + h["ret"]
        variants[name] = {"all_total": float(eqv[-1] - 1), "oos_total": float(tot - 1),
                          "months_oos": len(bt),
                          "avg_turnover": float(np.mean([h["turnover"] for h in hv])),
                          "win_oos": float(np.mean([h["ret"] > 0 for h in bt])) if bt else None,
                          "t_oos": float(np.mean([h["ret"] for h in bt]) /
                                         (np.std([h["ret"] for h in bt], ddof=1) /
                                          np.sqrt(len(bt)))) if len(bt) > 2 else None}

    out = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "universe": {"dates": len(dates), "codes": len(codes), "monthly": len(md),
                     "train": [md[0], md[train_k[-1]]], "oos": [md[oos_k[0]], md[-1]]},
        "spec": {"top_n": TOP_N, "cost": COST, "min_amt20": MIN_AMT20,
                 "min_history": MIN_HISTORY, "rebalance": "每月最后一个交易日"},
        "factors": rows,
        "combo": {"weights": w, "ic_train": float(np.nanmean(ic_combo_tr)),
                  "t_train": tstat(ic_combo_tr),
                  "ic_oos20": float(np.nanmean(ic_combo_os)), "t_oos20": tstat(ic_combo_os),
                  "ic_oos60": float(np.nanmean(ic_combo_os60)), "t_oos60": tstat(ic_combo_os60),
                  "q_oos": q_combo, "ic_folds": ic_combo_folds},
        "portfolio": {"all_total": float(all_total), "oos_total": float(total_oos - 1),
                      "months_oos": n_o, "win_oos": win_o, "avg_turnover": turn_o,
                      "bench": bench_ret},
        "variants": variants,
        "cross_breakout": cross,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                        encoding="utf-8")

    def pct(x, d=1):
        return "—" if x is None or not np.isfinite(x) else f"{x*100:+.{d}f}%"

    lines = [f"# 横截面因子扫描（{out['generated']}）", "",
             f"域：{len(dates)} 交易日 × {len(codes)} 只，{len(md)} 个月末截面，"
             f"训练 {out['universe']['train'][0]}→{out['universe']['train'][1]}，"
             f"验证 {out['universe']['oos'][0]}→{out['universe']['oos'][1]}", "",
             "| 因子 | 训练 IC (t) | 验证 IC 20日 (t) | 验证 IC 60日 (t) | 符号一致 | 验证五分位 Q1→Q5 |",
             "|---|---|---|---|---|---|"]
    for n, r in rows.items():
        lines.append(f"| {n} | {r['ic_train']:+.3f} ({r['t_train']:+.2f}) | "
                     f"{r['ic_oos20']:+.3f} ({r['t_oos20']:+.2f}) | "
                     f"{r['ic_oos60']:+.3f} ({r['t_oos60']:+.2f}) | "
                     f"{'是' if r['same_sign'] else '否'} | "
                     + " → ".join(pct(v, 2) for v in r["q_oos"]) + " |")
    c = out["combo"]
    lines += ["", f"**组合（训练期 IC 定权，共 {len(w)} 个因子）**：训练 IC {c['ic_train']:+.3f}"
                  f" (t={c['t_train']:+.2f})；验证 20日 {c['ic_oos20']:+.3f} (t={c['t_oos20']:+.2f})，"
                  f"60日 {c['ic_oos60']:+.3f} (t={c['t_oos60']:+.2f})；"
                  f"验证五分位 " + " → ".join(pct(v, 2) for v in c["q_oos"]), "",
              f"**组合净值**：全样本 {pct(out['portfolio']['all_total'])}，"
              f"样本外 {pct(out['portfolio']['oos_total'])}（{out['portfolio']['months_oos']} 个月，"
              f"月度胜率 {pct(out['portfolio']['win_oos'])}，月均换手 "
              f"{out['portfolio']['avg_turnover']*100:.0f}%）；"
              f"同期沪深300 {pct(out['portfolio']['bench']['沪深300']['total'])}", "",
              "| 组合变体 | 全样本 | 样本外 | 月均换手 | 样本外胜率 |",
              "|---|---|---|---|---|"]
    for n, v in variants.items():
        lines.append(f"| {n} | {pct(v['all_total'])} | {pct(v['oos_total'])} | "
                     f"{v['avg_turnover']*100:.0f}% | {pct(v['win_oos'])} |")
    lines += ["", "| 突破信号所在组合分位 | 信号数 | 其后 20 日超额 | 聚类 t |", "|---|---|---|---|"]
    for q, v in (cross or {}).items():
        if not isinstance(q, int):
            continue
        lines.append(f"| Q{q} | {v['n']} | {pct(v['exc20'], 2)} | {v.get('t', 0):+.2f} |")
    if isinstance((cross or {}).get("hi_vs_lo"), dict):
        hv = cross["hi_vs_lo"]
        lines.append(f"| **Q4+Q5 减 Q1+Q2** | — | {pct(hv['exc20'], 2)} | {hv.get('t', 0):+.2f} |")
    OUT_HTML.write_text("<pre>" + "\n".join(lines) + "</pre>", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n→ {OUT_JSON.name} / {OUT_HTML.name}")


if __name__ == "__main__":
    main()
