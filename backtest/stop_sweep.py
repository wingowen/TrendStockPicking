#!/usr/bin/env python3
"""止损方案对照：结构止损位置 × 固定止损 × 移动止盈 × 时间止损

信号集固定为面板口径（突破 + 量比 1.5~2 + 站稳 3 日 ≥3%），只换出场规则，
对比单笔净期望 / 胜率 / 平均亏损 / 组合收益 / 最大回撤。

用法：
    .venv-bt/bin/python stop_sweep.py
输出：stops.html、stops.json
"""
import json
from pathlib import Path

import numpy as np

from run_backtest import (BASE, EVAL_START, WARMUP_BARS, build_equity, build_equal_weight,
                          load_data, metrics_from_equity, find_signals)

OUT_HTML = Path(__file__).resolve().parent / "stops.html"
OUT_JSON = Path(__file__).resolve().parent / "stops.json"


def simulate(bars, sig, cfg):
    """扩展版出场：支持结构止损位置、固定止损、移动止盈、时间止损"""
    k = sig["signal_idx"]
    n = len(bars["c"])
    if k + 1 >= n:
        return None
    e = k + 1
    entry = bars["o"][e]
    if entry <= 0 or entry / bars["c"][k] - 1 > cfg["gap_max"]:
        return None

    level = sig["level"]
    stop_line = level * cfg["stop_mult"] if cfg["stop_mult"] else None
    hard_line = entry * (1 - cfg["hard_stop"]) if cfg["hard_stop"] else None
    ma_n = cfg.get("ma_stop") or 0
    ma = None
    if ma_n:
        c = bars["c"]
        cs = np.concatenate(([0.0], np.cumsum(c)))
        ma = np.full(len(c), np.nan)
        ma[ma_n - 1:] = (cs[ma_n:] - cs[0:len(c) - ma_n + 1]) / ma_n
    peak = bars["c"][e]
    exit_idx, reason = None, "数据结束"
    for i in range(e, n):
        peak = max(peak, bars["c"][i])
        if stop_line and bars["c"][i] < stop_line:
            exit_idx, reason = i + 1, "结构止损"
        elif hard_line and bars["c"][i] < hard_line:
            exit_idx, reason = i + 1, "固定止损"
        elif ma is not None and not np.isnan(ma[i]) and bars["c"][i] < ma[i]:
            exit_idx, reason = i + 1, "跌破均线"
        elif bars["c"][i] < peak * (1 - cfg["trail"]):
            exit_idx, reason = i + 1, "移动止盈"
        elif i - e + 1 >= cfg["max_hold"]:
            exit_idx, reason = i + 1, "持仓到期"
        if exit_idx is not None:
            break

    if exit_idx is None or exit_idx >= n:
        exit_idx = n - 1
        exit_price = bars["c"][exit_idx]
        if reason != "数据结束":
            reason = "数据结束"
    else:
        exit_price = bars["o"][exit_idx]

    net = exit_price / entry - 1 - cfg["cost"]
    return {"code": sig["code"], "name": sig.get("name", ""), "signal_date": sig["signal_date"],
            "entry_date": bars["dates"][e], "entry": float(entry),
            "exit_date": bars["dates"][exit_idx], "exit": float(exit_price),
            "hold": exit_idx - e + 1, "ret": float(net), "reason": reason,
            "mfe": float(np.max(bars["c"][e:exit_idx + 1]) / entry - 1),
            "mae": float(np.min(bars["c"][e:exit_idx + 1]) / entry - 1),
            "_bars": bars, "_entry_i": e}


CONFIGS = [
    # 名称, stop_mult(相对突破位), hard_stop(entry 起), trail, max_hold
    ("当前口径 破位止损 0%",            1.00, None, 0.12, 120),
    ("提前 1% 止损（位上 1%）",          1.01, None, 0.12, 120),
    ("提前 2% 止损（位上 2%）",          1.02, None, 0.12, 120),
    ("缓冲 2%（跌破位下 2%）",           0.98, None, 0.12, 120),
    ("缓冲 3%",                        0.97, None, 0.12, 120),
    ("缓冲 5%",                        0.95, None, 0.12, 120),
    ("不设结构止损（等价 -12% 硬止损）",   None, None, 0.12, 120),
    ("不设结构止损 + 移动 18%",          None, None, 0.18, 120),
    ("破位 + 固定 -6%",                 1.00, 0.06, 0.12, 120),
    ("破位 + 固定 -8%",                 1.00, 0.08, 0.12, 120),
    ("破位 + 固定 -10%",                1.00, 0.10, 0.12, 120),
    ("破位 + 时间 20 日",               1.00, None, 0.12, 20),
    ("破位 + 时间 40 日",               1.00, None, 0.12, 40),
    ("破位 + 时间 60 日",               1.00, None, 0.12, 60),
    ("破位 + 移动 8%",                  1.00, None, 0.08, 120),
    ("破位 + 移动 18%",                 1.00, None, 0.18, 120),
    ("破位 + 时间 20 + 移动 8%",        1.00, None, 0.08, 20),
    ("破位 + 时间 40 + 移动 8%",        1.00, None, 0.08, 40),
    ("MA20 止损（无结构止损）",          None, None, 0.30, 120),
    ("MA20 止损 + 移动 12%",            None, None, 0.12, 120),
    ("MA50 止损（无结构止损）",          None, None, 0.30, 120),
    ("不设止损 + 移动 22%",             None, None, 0.22, 120),
    ("不设止损 + 移动 25%",             None, None, 0.25, 120),
    ("纯持有 120 日（几乎不止损）",        None, None, 0.50, 120),
]


def main():
    stocks, data, idx = load_data()
    cfg0 = dict(BASE)
    print("[signal] 生成信号", flush=True)
    signals = []
    for code, b in data.items():
        if len(b["dates"]) < WARMUP_BARS:
            continue
        for s in find_signals(b, cfg0):
            if s["signal_date"] >= EVAL_START:
                signals.append(dict(s, code=code, name=stocks.get(code, {}).get("name", "")))
    print(f"[signal] {len(signals)} 笔", flush=True)

    idx_dates = idx["上证指数"]["dates"]
    bench = idx["上证指数"]["close"]
    # 大盘状态：上证收盘 vs MA20（非弱 = 站上 MA20）
    ma20 = np.full(len(bench), np.nan)
    ma20[19:] = np.convolve(bench, np.ones(20) / 20, mode="valid")
    ok_map = {d: (not np.isnan(ma20[i])) and bench[i] >= ma20[i]
              for i, d in enumerate(idx_dates)}

    rows = []
    all_trades, eqs = {}, {}
    for name, sm, hard, trail, hold in CONFIGS:
        ma_stop = 20 if name.startswith("MA20") else (50 if name.startswith("MA50") else None)
        cfg = dict(cfg0, stop_mult=sm, hard_stop=hard, trail=trail, max_hold=hold,
                   ma_stop=ma_stop)
        trades = []
        for sig in signals:
            t = simulate(data[sig["code"]], sig, cfg)
            if t:
                trades.append(t)
        if not trades:
            continue
        all_trades[name] = trades
        rets = np.array([t["ret"] for t in trades])
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        ew = build_equal_weight(trades, idx_dates, cfg)[0]
        eqs[name] = ew
        pm = metrics_from_equity(ew, idx_dates)
        stop_share = sum(1 for t in trades if t["reason"] == "结构止损") / len(trades)
        rows.append({
            "name": name, "stop_mult": sm, "hard": hard, "trail": trail, "hold": hold,
            "n": len(trades), "avg": float(rets.mean()), "median": float(np.median(rets)),
            "win": float(len(wins) / len(trades)), "avg_loss": float(losses.mean()) if len(losses) else 0,
            "avg_win": float(wins.mean()) if len(wins) else 0,
            "stop_share": float(stop_share),
            "avg_hold": float(np.mean([t["hold"] for t in trades])),
            "ew_ret": pm.get("total"), "ew_dd": pm.get("mdd"),
        })
        print(f"  {name:<26} 单笔 {rets.mean()*100:6.2f}%  胜率 {len(wins)/len(trades)*100:4.1f}%  "
              f"组合 {(pm.get('total') or 0)*100:7.1f}%  回撤 {(pm.get('mdd') or 0)*100:5.1f}%",
              flush=True)

    # 第二步：在最优出场上叠加大盘过滤
    extra = [("最优出场 + 大盘非弱", None, None, 0.18, 120, True),
             ("最优出场 + 大盘弱", None, None, 0.18, 120, False)]
    for name, sm, hard, trail, hold, only_ok in extra:
        cfg = dict(cfg0, stop_mult=sm, hard_stop=hard, trail=trail, max_hold=hold,
                   ma_stop=None)
        trades = []
        for sig in signals:
            if ok_map.get(sig["signal_date"]) == only_ok:
                t = simulate(data[sig["code"]], sig, cfg)
                if t:
                    trades.append(t)
        if not trades:
            continue
        rets = np.array([t["ret"] for t in trades])
        wins, losses = rets[rets > 0], rets[rets <= 0]
        ew = build_equal_weight(trades, idx_dates, cfg)[0]
        pm = metrics_from_equity(ew, idx_dates)
        rows.append({"name": name, "stop_mult": sm, "hard": hard, "trail": trail,
                     "hold": hold, "n": len(trades), "avg": float(rets.mean()),
                     "median": float(np.median(rets)), "win": float(len(wins) / len(trades)),
                     "avg_loss": float(losses.mean()) if len(losses) else 0,
                     "avg_win": float(wins.mean()) if len(wins) else 0,
                     "stop_share": 0.0,
                     "avg_hold": float(np.mean([t["hold"] for t in trades])),
                     "ew_ret": pm.get("total"), "ew_dd": pm.get("mdd")})
        print(f"  {name:<26} 单笔 {rets.mean()*100:6.2f}%  胜率 {len(wins)/len(trades)*100:4.1f}%"
              f"  {len(trades)} 笔  等权 {(pm.get('total') or 0)*100:7.1f}%", flush=True)

    # 分年度：看结论是不是只靠某一年
    years = sorted({d[:4] for d in idx_dates if d >= EVAL_START})
    yearly = {}
    for nm in ("当前口径 破位止损 0%", "不设结构止损 + 移动 18%", "不设止损 + 移动 25%",
               "纯持有 120 日（几乎不止损）"):
        eq = eqs.get(nm)
        if eq is None:
            continue
        y = {}
        for yr in years:
            idxs = [i for i, d in enumerate(idx_dates) if d[:4] == yr]
            if len(idxs) < 2:
                continue
            y[yr] = float(eq[idxs[-1]] / eq[idxs[0]] - 1)
        yearly[nm] = y
    print("[yearly] " + json.dumps(yearly, ensure_ascii=False))
    rows.sort(key=lambda r: -(r["avg"] or -9))
    OUT_JSON.write_text(json.dumps({"configs": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    render(rows, yearly)
    print("→ stops.html / stops.json")


def render(rows, yearly):
    def pct(x, d=2):
        return "—" if x is None else f"{x*100:.{d}f}%"

    tr = "\n".join(
        f"<tr><td class=l>{r['name']}</td><td>{r['n']}</td><td>{pct(r['avg'])}</td>"
        f"<td>{pct(r['median'])}</td><td>{pct(r['win'],1)}</td><td>{pct(r['avg_loss'])}</td>"
        f"<td>{pct(r['avg_win'])}</td><td>{pct(r['stop_share'],1)}</td>"
        f"<td>{r['avg_hold']:.1f}</td><td><b>{pct(r['ew_ret'],1)}</b></td>"
        f"<td>{pct(r['ew_dd'],1)}</td></tr>" for r in rows)
    years = sorted({y for v in yearly.values() for y in v})
    yhead = "".join(f"<th>{y}</th>" for y in years)

    def py(v):
        return "—" if v is None else f"{v*100:.1f}%"

    yrows = "\n".join(
        f"<tr><td class=l>{nm}</td>" + "".join(f"<td>{py(v.get(y))}</td>" for y in years) + "</tr>"
        for nm, v in yearly.items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>止损方案对照</title>
<style>
body{{font:14px/1.7 -apple-system,"PingFang SC",sans-serif;margin:32px;color:#1a1a1a}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#888;font-size:12px;margin-bottom:18px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 9px;border-bottom:1px solid #eee;text-align:right;font-size:13px}}
th{{background:#fafafa;font-weight:600;color:#444}} td.l,th.l{{text-align:left}}
tr:hover td{{background:#fcfcfc}}
</style></head><body>
<h1>止损方案对照</h1>
<div class="sub">信号集：突破 + 量比 1.5~2.0 + 站稳 3 日且幅度 ≥3%（主板，2020-01-01 起）。
单笔按 0.4% 往返成本计，次日开盘成交。等权收益 = 所有信号等额参与的净值曲线（衡量规则本身，不含仓位上限）。</div>
<table><thead><tr><th class=l>方案</th><th>笔数</th><th>单笔净期望</th><th>中位</th><th>胜率</th>
<th>平均亏损</th><th>平均盈利</th><th>结构止损占比</th><th>平均持仓</th>
<th>等权收益</th><th>等权最大回撤</th></tr></thead><tbody>
{tr}
</tbody></table>
<h1 style="margin-top:32px">分年度收益（等权净值）</h1>
<div class="sub">看结论是不是只靠某一年。2026 为不完整年度（截至 2026-11-20）。</div>
<table><thead><tr><th class=l>方案</th>{yhead}</tr></thead><tbody>
{yrows}
</tbody></table></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
