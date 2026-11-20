#!/usr/bin/env python3
"""折中出场：破位只减半仓，剩余半仓继续 25% 移动止盈 / 120 日。

对照：
  A  现状：不砍破位，全程 25% 移动止盈 + 120 日
  P  破位减半：收盘跌破突破位 → 次日开盘卖 50%；剩余 50% 继续 25% 止盈 / 120 日
  Q  破位减 1/3（更轻）
  R  破位减 2/3（更重）
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import run_backtest as rb
from run_backtest import (BASE, EVAL_START, WARMUP_BARS, load_data, find_signals,
                          trade_stats, metrics_from_equity, build_equal_weight, svg_curve)
from exit_compare import simulate

ROOT = Path(__file__).resolve().parent


def simulate_partial(bars, sig, cfg, cut=0.5):
    """破位卖 cut 比例的仓位，剩余继续 25% 止盈 / 120 日。返回两段式结果。"""
    k = sig["signal_idx"]
    n = len(bars["c"])
    if k + 1 >= n:
        return None
    e = k + 1
    entry = bars["o"][e]
    gap = entry / bars["c"][k] - 1
    if entry <= 0 or gap > cfg["gap_max"] or gap < -cfg.get("gap_min", 0):
        return None
    level = sig["level"]
    peak = bars["c"][e]
    half_k = None          # 破位触发日（次日开盘减仓）
    exit_idx = None
    reason = "数据结束"
    for i in range(e, n):
        c = bars["c"][i]
        peak = max(peak, c)
        if half_k is None and c < level and cut > 0:
            half_k = i
        if c < peak * (1 - cfg["trail"]):
            exit_idx, reason = i + 1, "移动止盈"
        elif i - e + 1 >= cfg["max_hold"]:
            exit_idx, reason = i + 1, "持仓到期"
        if exit_idx is not None:
            break
    if exit_idx is None or exit_idx >= n:
        exit_idx, exit_px, reason = n - 1, bars["c"][n - 1], "数据结束"
    else:
        exit_px = bars["o"][exit_idx]
    # 减仓价：破位次日开盘；若与最终出场同日则同价
    if half_k is not None:
        hk = half_k + 1
        half_px = bars["o"][hk] if hk < n else exit_px
        half_idx = min(hk, n - 1)
    else:
        half_px, half_idx = None, None
    kept = 1.0 - (cut if half_k is not None else 0.0)
    gross = (cut * (half_px / entry - 1) if half_px else 0.0) + kept * (exit_px / entry - 1)
    return {**sig, "entry_date": bars["dates"][e], "entry": entry,
            "exit_date": bars["dates"][exit_idx], "exit": exit_px,
            "half_date": bars["dates"][half_idx] if half_idx is not None else None,
            "half_px": half_px, "half_idx": half_idx, "kept": kept, "cut": cut,
            "hold": exit_idx - e + 1, "ret": gross - cfg["cost"], "reason": reason,
            "mfe": float(np.max(bars["c"][e:exit_idx + 1]) / entry - 1),
            "mae": float(np.min(bars["c"][e:exit_idx + 1]) / entry - 1),
            "half_ret": (half_px / entry - 1) if half_px else None}


def build_equity_partial(trades, idx_dates, cfg):
    """6 仓 + 部分减仓的组合净值。"""
    date_pos = {d: i for i, d in enumerate(idx_dates)}
    n = len(idx_dates)
    cash = cfg["capital"]
    positions, pend = [], {}
    for tr in trades:
        di = date_pos.get(tr["entry_date"])
        if di is not None:
            pend.setdefault(di, []).append(tr)
    eq = np.zeros(n)
    for di, d in enumerate(idx_dates):
        keep = []
        for p in positions:
            # 减仓（只做一次）
            if p["half_idx"] == di and not p["half_done"]:
                cash += p["half_shares"] * p["half_px"]
                p["half_done"] = True
            # 全部平仓
            if p["exit_idx"] == di:
                cash += (p["shares"] - (p["half_shares"] if p["half_done"] else 0)) * p["exit_px"]
            else:
                keep.append(p)
        positions = keep
        mv = 0.0
        for p in positions:
            b = p["bars"]
            i = p["i"]
            while i + 1 < len(b["dates"]) and b["dates"][i + 1] <= d:
                i += 1
            p["i"] = i
            mv += (p["shares"] - (p["half_shares"] if p["half_done"] else 0)) * b["c"][i]
        held = {p["code"] for p in positions}
        for tr in sorted(pend.get(di, []), key=lambda x: x["code"]):
            if len(positions) >= cfg["slots"] or cash <= 0:
                break
            if tr["code"] in held:
                continue
            buy_px = tr["entry"] * (1 + cfg["cost"] / 2)
            budget = min((cash + mv) / cfg["slots"], cash)
            if cfg.get("risk_pct"):
                budget = min((cash + mv) * cfg["risk_pct"] / max(cfg["trail"], 1e-6), budget)
            shares = int(budget / buy_px / 100) * 100
            if shares <= 0:
                continue
            half = int(shares * tr["cut"] / 100) * 100 if tr["half_idx"] is not None else 0
            cash -= shares * buy_px
            held.add(tr["code"])
            positions.append({"code": tr["code"], "shares": shares, "half_shares": half,
                              "half_done": False, "half_idx": date_pos.get(tr["half_date"]) if tr["half_date"] else None,
                              "half_px": tr["half_px"] * (1 - cfg["cost"] / 2) if tr["half_px"] else None,
                              "exit_idx": date_pos.get(tr["exit_date"], n - 1),
                              "exit_px": tr["exit"] * (1 - cfg["cost"] / 2),
                              "bars": tr["_bars"], "i": tr["_entry_i"]})
            mv += shares * buy_px
            tr["_used"] = True
        eq[di] = cash + mv
    return eq


def main():
    cfg = dict(BASE)
    stocks, data, idx = load_data()
    bench = idx["沪深300"]
    dates = bench["dates"]
    sigs = []
    for code, b in data.items():
        if len(b["dates"]) < WARMUP_BARS:
            continue
        for sig in find_signals(b, cfg):
            if b["dates"][sig["signal_idx"]] >= EVAL_START:
                sigs.append((code, b, sig))
    print(f"[sig] {len(sigs)}", flush=True)

    variants = [("A", "现状：不砍破位 + 25% 止盈", None),
                ("P", "破位减半（50%）", 0.5),
                ("Q", "破位减 1/3", 1 / 3),
                ("R", "破位减 2/3", 2 / 3)]
    out = []
    for key, label, cut in variants:
        trs = []
        for code, b, sig in sigs:
            t = simulate(b, sig, cfg, "base") if cut is None else simulate_partial(b, sig, cfg, cut)
            if not t:
                continue
            t["code"] = code
            t["_bars"] = b
            t["_entry_i"] = sig["signal_idx"] + 1
            trs.append(t)
        trs.sort(key=lambda t: (t["entry_date"], t["code"]))
        si = next(i for i, d in enumerate(dates) if d >= EVAL_START)
        if cut is None:
            eq = rb.build_equity(trs, dates, bench["close"], cfg)
            eqw, _ = build_equal_weight(trs, dates, cfg)
        else:
            eq = build_equity_partial(trs, dates, cfg)
            eqw, _ = build_equal_weight(trs, dates, cfg)
        m = metrics_from_equity(eq[si:], dates[si:])
        mw = metrics_from_equity(eqw[si:], dates[si:])
        st = trade_stats(trs)
        cuts = sum(1 for t in trs if t.get("half_idx") is not None)
        out.append({"key": key, "label": label, "port": m, "eqw": mw, "stats": st,
                    "cuts": cuts, "curve": eq[si:]})
        print(f"{key} {label:26s} 6仓{m['total']*100:+7.2f}% 回撤{m['mdd']*100:6.2f}% "
              f"Sharpe{m['sharpe']:5.2f} | 等权{mw['total']*100:+7.2f}% | 单笔{st['avg']*100:+6.2f}% "
              f"胜率{st['win']*100:4.1f}% 持有{st['hold']:.0f}天 减仓{cuts}笔", flush=True)

    bench_curve = bench["close"][si:].astype(float)
    bench_curve = bench_curve / bench_curve[0]

    def pct(v): return f"{v*100:+.2f}%"
    def cls(v): return "pos" if v > 0 else "neg"

    rows = ""
    for r in out:
        m, st, mw = r["port"], r["stats"], r["eqw"]
        rows += (f"<tr><td><b>{r['key']}</b> {r['label']}</td>"
                 f"<td class='num'>{st['n']}</td><td class='num'>{st['win']*100:.1f}%</td>"
                 f"<td class='num {cls(st['avg'])}'>{pct(st['avg'])}</td><td class='num'>{st['hold']:.0f}</td>"
                 f"<td class='num'>{r['cuts']}</td>"
                 f"<td class='num {cls(m['total'])}'>{pct(m['total'])}</td>"
                 f"<td class='num {cls(m['mdd'])}'>{pct(m['mdd'])}</td><td class='num'>{m['sharpe']:.2f}</td></tr>")
    curves = ""
    for r in out:
        n = min(len(r["curve"]), len(bench_curve))
        curves += (f"<div class='box'><div class='cap'>{r['key']} · {r['label']}</div>"
                   f"{svg_curve(r['curve'][:n].tolist(), bench_curve[:n].tolist(), dates[si:si+n], '6 仓实盘（红）')}</div>")

    best = max(out[1:], key=lambda r: r["port"]["sharpe"]) if len(out) > 1 else out[0]
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>破位减仓折中方案</title><style>
:root{{--ink:#2C2C2A;--ink2:#5F5E5A;--ink3:#888780;--line:#e3e5e8;--up:#C43A2E;--down:#1D7A4C}}
*{{box-sizing:border-box;margin:0;padding:0}}body{{background:#f5f6f8;color:var(--ink);font:14px/1.65 -apple-system,"PingFang SC",sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:30px 24px 70px}}h1{{font-size:24px;font-weight:650}}
h2{{font-size:15px;font-weight:600;margin:26px 0 10px}}.sub{{color:var(--ink3);font-size:12px;margin-top:6px}}
.box{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{text-align:left;font-size:11px;color:var(--ink3);font-weight:500;padding:8px;border-bottom:1px solid var(--line);background:#fafbfc}}
td{{padding:8px;border-bottom:1px solid #f2f3f5}}td.num,th.num{{text-align:right}}
.pos{{color:var(--up)}}.neg{{color:var(--down)}}.cap{{font-size:12px;color:var(--ink3);margin-bottom:6px}}
.note{{font-size:12.5px;color:var(--ink2);line-height:1.9}}.note li{{margin-left:18px}}
</style></head><body><div class="wrap">
<h1>破位减仓 · 折中出场方案</h1>
<div class="sub">2020 起 · {len(data)} 只主板 · 6 仓 / 等权 · 成本 0.4% · 生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<h2>结果</h2>
<div class="box"><table><thead><tr><th>口径</th><th class="num">信号数</th><th class="num">胜率</th><th class="num">单笔净期望</th>
<th class="num">平均持有</th><th class="num">减仓笔数</th><th class="num">6仓累计</th><th class="num">6仓回撤</th>
<th class="num">Sharpe</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="sub">减仓笔数 = 实际触发过破位减仓的笔数。</div>
<h2>净值曲线</h2>{curves}
<h2>读法</h2>
<div class="box note"><ul>
<li>减半仓 = 破位那次只卖 50%，剩下的 50% 继续按 25% 移动止盈走，直到回撤 25% 或满 120 天。</li>
<li>如果折中档能把 6 仓做到接近基准，就说明“破位要处理”这个直觉可以通过减仓而不是清仓来实现。</li>
<li>减仓比例越大越接近“破位即卖”（越差），越小越接近基准（越好）——看曲线在哪一档拐弯。</li>
</ul></div>
<div class="sub">脚本 <code>backtest/partial_exit.py</code></div>
</div></body></html>"""
    (ROOT / "partial_exit.html").write_text(html, encoding="utf-8")
    (ROOT / "partial_exit.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "curve"} for r in out],
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"→ partial_exit.html / partial_exit.json | 折中档最优 {best['key']} {best['label']}")


if __name__ == "__main__":
    main()
