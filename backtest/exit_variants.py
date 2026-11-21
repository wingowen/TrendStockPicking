#!/usr/bin/env python3
"""出场规则在同一组合引擎下的对比。

v1 的「25% 移动止盈最优点」是在等权口径上选出来的，规则卡却用 6 仓路径背书。
这里把出场方案放回同一个组合框架（10 仓、按距突破位升序、现金 1.5%、
净值回撤 10%/15% 降档），样本内 + walk-forward 样本外各看一遍。

用法：
    .venv-bt/bin/python exit_variants.py
输出：
    exit_variants.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import engine_v2 as E
import run_backtest as R

ROOT = Path(__file__).resolve().parent

VARIANTS = [
    ("移动止盈25% + 120日", dict(trail=0.25, max_hold=120, level_stop=False), "risk"),
    ("纯持有120日", dict(trail=0.99, max_hold=120, level_stop=False), "equal"),
    ("纯持有180日", dict(trail=0.99, max_hold=180, level_stop=False), "equal"),
    ("移动止盈35% + 120日", dict(trail=0.35, max_hold=120, level_stop=False), "risk"),
    ("移动止盈25% + 180日", dict(trail=0.25, max_hold=180, level_stop=False), "risk"),
]


def aux_map():
    """从 features.csv 取「信号日可知」的辅助因子，按 (code, signal_date) 对齐。"""
    with open(E.FEATURES, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out[(r["code"], r["signal_date"])] = {
            "regime": r["regime"] or "",
            "sec_score": float(r["sec_score"]) if r["sec_score"] not in ("", "None") else None,
            "sec_consec": int(float(r["sec_consec"])) if r["sec_consec"] not in ("", "None") else 0,
            "days_above": int(float(r["days_above"])),
            "vol_ratio": float(r["vol_ratio"]),
            "fwd20": float(r["fwd20"]) if r["fwd20"] not in ("", "None") else None,
            "bench20": float(r["bench20"]) if r["bench20"] not in ("", "None") else None,
        }
    return out


def rebuild(exit_cfg, aux):
    cfg = dict(R.BASE)
    cfg.update(exit_cfg)
    stocks, data, idx = R.load_data()
    out = []
    for code, b in data.items():
        if len(b["dates"]) < R.WARMUP_BARS:
            continue
        for s in R.find_signals(b, cfg):
            k = s["signal_idx"]
            if b["dates"][k] < R.EVAL_START:
                continue
            t = R.simulate_trade(b, s, cfg)
            if not t:
                continue
            a = aux.get((code, s["signal_date"]))
            if a is None:
                continue
            t.update({k2: v for k2, v in a.items()})
            t["code"] = code
            t["name"] = stocks.get(code, {}).get("name", code)
            t["gain"] = s["gain"]
            out.append(t)
    out.sort(key=lambda t: (t["entry_date"], t["code"]))
    return out


def main():
    dates, bench = E.load_calendar()
    aux = aux_map()
    rows0 = E.load_features()
    closes, lot_price = E.prep(dates, rows0)
    results = {}

    for name, exit_cfg, mode in VARIANTS:
        rows = rebuild(exit_cfg, aux)
        print(f"[{name}] 信号 {len(rows)}")
        base = dict(slots=10, mode=mode, risk_pct=0.02, trail=exit_cfg["trail"],
                    daily_cap=3, cost=0.004, cash_yield=0.015,
                    dd_soft=0.10, dd_hard=0.15, min_score=None)
        rank = {(r["code"], r["entry_date"]): -r["gain"] for r in rows}
        eq, m = E.run_cfg(rows, dates, closes, lot_price, base, rank)
        wf_full, wf = E.walk_forward(rows, dates, closes, lot_price, base)
        wf_m = E.metrics(wf_full, dates) if wf else {}
        results[name] = {"in": m, "oos": wf_m, "folds": wf}
        print(f"  样本内 累计{pct(m.get('total'))} 年化{pct(m.get('cagr'))} "
              f"回撤{pct(m.get('mdd'))} Sharpe{m.get('sharpe',0):.2f} 成交{m.get('fills')} "
              f"仓位{m.get('avg_exposure',0)*100:.0f}%")
        print(f"  样本外 累计{pct(wf_m.get('total'))} 年化{pct(wf_m.get('cagr'))} "
              f"回撤{pct(wf_m.get('mdd'))}  " +
              " ".join(f"{y}:{pct(r)}" for y, r, *_ in wf))

    (ROOT / "exit_variants.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("→ exit_variants.json")


def pct(v, d=1):
    return "—" if v is None else f"{v*100:+.{d}f}%"


if __name__ == "__main__":
    main()
