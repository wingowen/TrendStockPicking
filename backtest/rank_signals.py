#!/usr/bin/env python3
"""每日精选：把当日通过筛选的信号排序，取 Top N 当作「今天最该关注的标的」

排序分只使用信号当日可知的信息（大盘状态、板块趋势、量比、站稳天数、涨幅），
回踩状态是信号之后才发生的，不进入排序分（避免未来函数），只在页面展示。

用法：
    .venv-bt/bin/python rank_signals.py            # 全历史评估 + 生成 rank.html/json
输出：rank.html、rank.json、picks.csv
"""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_backtest as R
from analyze_factors import regime_series
from sector_trend import build_industry_indices, snapshot

ROOT = Path(__file__).resolve().parent
OUT_HTML = ROOT / "rank.html"
OUT_JSON = ROOT / "rank.json"
OUT_CSV = ROOT / "picks.csv"


def load_amount(idx):
    con = sqlite3.connect(R.DB)
    rows = list(con.execute("select date, amount from bars where code='IDX_上证指数' order by date"))
    con.close()
    amt = {d: a for d, a in rows}
    idx["上证指数"] = dict(idx["上证指数"])
    idx["上证指数"]["amount"] = np.array(
        [amt.get(d, 0.0) for d in idx["上证指数"]["dates"]], dtype=float)
    return idx


def gather():
    stocks, bars, idx = R.load_data()
    idx = load_amount(idx)
    con = sqlite3.connect(R.DB)
    industry = {r[0]: (r[1] or "") for r in con.execute("select code, industry from stocks")}
    con.close()

    regime = regime_series(idx)
    cal = idx["上证指数"]["dates"]
    ind_series = build_industry_indices(bars, industry, cal)

    rows = []
    for code, b in bars.items():
        if len(b["dates"]) < R.WARMUP_BARS:
            continue
        for s in R.find_signals(b, R.BASE):
            k = s["signal_idx"]
            if b["dates"][k] < R.EVAL_START:
                continue
            tr = R.simulate_trade(b, s, R.BASE)
            if not tr:
                continue
            t = tr
            t["_bars"] = b
            t["_entry_i"] = k + 1
            t["code"] = code
            t["name"] = stocks.get(code, {}).get("name", code)
            t["industry"] = industry.get(code, "")
            t["regime"] = (regime.get(s["signal_date"]) or {}).get("label")
            t["days_above"] = k - s["breakout_idx"] + 1
            t["gain"] = s["gain"]
            t["vol_ratio"] = s["vol_ratio"]
            rows.append(t)

    need = sorted({r["signal_date"] for r in rows})
    sector = {d: snapshot(ind_series, d) for d in need}
    for r in rows:
        sec = (sector.get(r["signal_date"]) or {}).get(r["industry"])
        r["sec_label"] = sec["label"] if sec else None
        r["sec_score"] = sec["score"] if sec else None
        r["sec_mom20"] = sec["mom20"] if sec else None
        r["sec_flow5"] = sec["flow5"] if sec else None
        r["sec_consec"] = sec["consec"] if sec else None
    return rows, cal, idx["上证指数"]["close"]


def buckets(rows, keyfn, minn=60):
    g = defaultdict(list)
    for r in rows:
        k = keyfn(r)
        if k is not None:
            g[k].append(r)
    out = []
    for k in sorted(g, key=lambda x: str(x)):
        lst = g[k]
        if len(lst) < minn:
            continue
        rets = np.array([x["ret"] for x in lst])
        out.append((k, len(lst), float(rets.mean()), float(np.median(rets)),
                    float((rets > 0).mean()), float(np.mean([x["hold"] for x in lst])))
        )
    return out


def score(r):
    """综合排序分，只用当日可知信息；权重按分层实测的单笔期望校准"""
    s = 0.0
    # 大盘：强 +40，弱 +5，震荡 -20（震荡在这套信号里最差）
    s += {"强": 40, "弱": 5}.get(r["regime"], -20)
    # 板块评分：75（刚转强）+25，100 +10，50 +5，25/0 0
    s += {75: 25, 100: 10, 50: 5}.get(r["sec_score"], 0)
    # 板块连续净流入：≥3 日 +12，1-2 日 +6
    c = r["sec_consec"] or 0
    s += 12 if c >= 3 else (6 if c >= 1 else 0)
    # 站稳天数：6-8 天 +15，5 天 +10，3-4 天 +5，9+ +5
    d = r["days_above"]
    s += 15 if 6 <= d <= 8 else (10 if d == 5 else (5 if d <= 4 else 5))
    # 突破后涨幅：<5% +25，5-8% +12，8-12% 0，>12% -25
    g = r["gain"]
    s += 25 if g < 0.05 else (12 if g < 0.08 else (0 if g < 0.12 else -25))
    # 量比：1.65-1.8 +10，1.8-1.9 +2，1.9-2.0 +6，1.5-1.65 0
    v = r["vol_ratio"]
    s += 10 if v < 1.8 else (6 if v >= 1.9 else 2)
    return round(s, 1)


def evaluate(rows, topn):
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["signal_date"]].append(r)
    picked = []
    for d, lst in by_day.items():
        lst = sorted(lst, key=lambda x: (-score(x), x["code"]))
        picked.extend(lst[:topn])
    rets = np.array([x["ret"] for x in picked])
    return picked, rets


def main():
    rows, cal, idxclose = gather()
    print(f"[rows] {len(rows)} 笔")

    print("== 站稳天数 ==")
    for k, n, a, m, w, h in buckets(rows, lambda r: r["days_above"] if r["days_above"] <= 8 else "9+"):
        print(f"  {k:<6} {n:5d}  {a*100:+6.2f}%  胜率 {w*100:4.1f}%")
    print("== 突破后涨幅 ==")
    for k, n, a, m, w, h in buckets(
            rows, lambda r: ("3-5%" if r["gain"] < 0.05 else "5-8%" if r["gain"] < 0.08
                             else "8-12%" if r["gain"] < 0.12 else ">12%")):
        print(f"  {k:<6} {n:5d}  {a*100:+6.2f}%  胜率 {w*100:4.1f}%")
    print("== 量比 ==")
    for k, n, a, m, w, h in buckets(
            rows, lambda r: ("1.5-1.65" if r["vol_ratio"] < 1.65 else "1.65-1.8"
                             if r["vol_ratio"] < 1.8 else "1.8-1.9" if r["vol_ratio"] < 1.9 else "1.9-2.0")):
        print(f"  {k:<8} {n:5d}  {a*100:+6.2f}%  胜率 {w*100:4.1f}%")

    rows_s = [dict(r, sc=score(r)) for r in rows]
    print("== 综合分分层 ==")
    for k, n, a, m, w, h in buckets(rows_s, lambda r: f"{int(r['sc']//10)*10}-{int(r['sc']//10)*10+9}"):
        print(f"  {k:<8} {n:5d}  {a*100:+6.2f}%  胜率 {w*100:4.1f}%")

    stats = {}
    for n in (1, 2, 3, 5):
        picked, rets = evaluate(rows_s, n)
        eq = R.build_equity(picked, cal, idxclose, dict(R.BASE, slots=n))
        pm = R.metrics_from_equity(eq, cal)
        st = {"n": len(picked), "avg": float(rets.mean()),
              "win": float((rets > 0).mean()),
              "med": float(np.median(rets)),
              "pf": float(pm["total"]), "dd": float(pm["mdd"])}
        stats[n] = st
        print(f"每日 Top{n}: {len(picked):5d} 笔  单笔 {st['avg']*100:+6.2f}%  胜率 {st['win']*100:4.1f}%"
              f"  组合 {st['pf']*100:+6.1f}%  回撤 {st['dd']*100:.1f}%")

    print("== 阈值筛选（当日最多 3 只）==")
    for T in (80, 90, 95, 100, 105, 110):
        sub = [r for r in rows_s if r["sc"] >= T]
        p3, r3 = evaluate(sub, 3)
        if not len(r3):
            continue
        eq = R.build_equity(p3, cal, idxclose, dict(R.BASE, slots=3))
        pm = R.metrics_from_equity(eq, cal)
        print(f"  分≥{T}: {len(sub):5d} 笔候选  取 {len(p3):4d} 笔  单笔 {r3.mean()*100:+6.2f}%"
              f"  胜率 {(r3>0).mean()*100:4.1f}%  组合 {pm['total']*100:+7.1f}%  回撤 {pm['mdd']*100:5.1f}%")

    # 只在「大盘强 + 板块 ≥75」的日子里选 Top3
    strong = [r for r in rows_s if r["regime"] == "强" and (r["sec_score"] or 0) >= 75]
    picked3, rets3 = evaluate(strong, 3)
    print(f"强市×板块≥75 Top3: {len(picked3)} 笔  单笔 {(rets3.mean() if len(rets3) else 0)*100:+6.2f}%"
          f"  胜率 {(rets3>0).mean()*100 if len(rets3) else 0:4.1f}%")

    OUT_JSON.write_text(json.dumps(
        {"base": {"n": len(rows_s), "avg": float(np.mean([r['ret'] for r in rows_s]))},
         "score_buckets": [{"key": k, "n": n, "avg": a, "win": w}
                           for k, n, a, m, w, h in buckets(
                               rows_s, lambda r: f"{int(r['sc']//10)*10}-{int(r['sc']//10)*10+9}")],
         "topn": {str(k): v for k, v in stats.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 最近 30 个交易日的精选，供页面核对
    recent = sorted({r["signal_date"] for r in rows_s})[-30:]
    latest = {}
    by_day = defaultdict(list)
    for r in rows_s:
        by_day[r["signal_date"]].append(r)
    for d in recent:
        lst = sorted(by_day.get(d, []), key=lambda x: (-x["sc"], x["code"]))
        latest[d] = [{"code": x["code"], "name": x["name"], "industry": x["industry"],
                      "score": x["sc"], "ret": round(x["ret"] * 100, 2),
                      "regime": x["regime"], "sec": x["sec_label"],
                      "sec_score": x["sec_score"]} for x in lst[:5]]
    (ROOT / "picks_recent.json").write_text(json.dumps(latest, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print("→ rank.json / picks_recent.json")


if __name__ == "__main__":
    main()
