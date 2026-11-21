#!/usr/bin/env python3
"""突破趋势跟踪 · 回测引擎

在主板全市场日线（东方财富公开数据，后复权）上复现「突破 + 量能 + 站稳」筛选，
按「信号次日开盘买入、结构/回撤止盈止损」做逐笔与组合回测，并输出 HTML 报告。

用法：
    .venv-bt/bin/python run_backtest.py                # 基准参数
    .venv-bt/bin/python run_backtest.py --sweep        # 附参数敏感性
    .venv-bt/bin/python run_backtest.py --trail 0.10   # 改移动止盈
"""
import argparse
from bisect import bisect_left
import csv
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DB = ROOT / "market.db"
OUT_HTML = ROOT / "report.html"
OUT_TRADES = ROOT / "trades.csv"
OUT_JSON = ROOT / "summary.json"

EVAL_START = "2020-01-01"
WARMUP_BARS = 70

BASE = dict(
    L=20,            # 突破位窗口：前 L 日最高收盘
    W=8,             # 突破发生在最近 W 个交易日内
    vol_min=1.5,     # 突破日量比下限
    vol_max=2.0,     # 突破日量比上限
    hold_min=3,      # 站稳最少交易日
    gain_min=0.03,   # 站稳幅度下限
    trail=0.25,      # 移动止盈回撤（止损方案实测：紧止损是主要亏损源，放宽到 25%）
    max_hold=120,    # 最长持仓（交易日）
    cost=0.004,      # 单次往返成本（佣金+印花+滑点）
    gap_max=0.095,   # 次日开盘涨幅超此值视为涨停买不进
    gap_min=0.03,    # 次日开盘低开超此值放弃（隔夜反转，回测被拦单笔 -3.32%）
    level_stop=False,# 不再用「跌破突破位」止损（58% 持仓被噪声洗出）
    slots=6,         # 组合同时持仓上限
    capital=50000.0,
    risk_pct=0.02,   # 单笔风险预算：账户权益的 2%（按 trail 距离折算仓位）
)


# ---------------- 数据 ----------------

def load_data():
    print("[load] 打开数据库", flush=True)
    con = sqlite3.connect(DB)
    stocks = {r[0]: {"name": r[1], "list_date": r[2]}
              for r in con.execute("select code, name, list_date from stocks")}
    data, cur_code, buf = {}, None, []

    for code, date, o, c, h, l, v in con.execute(
            "select code, date, open, close, high, low, volume from bars "
            "order by code, date"):
        if code.startswith("IDX_"):
            continue
        if code != cur_code:
            if cur_code:
                data[cur_code] = _pack(buf)
            cur_code, buf = code, []
        buf.append((date, o, c, h, l, v))
    if cur_code:
        data[cur_code] = _pack(buf)

    idx = {}
    for name in ("沪深300", "上证指数"):
        rows = list(con.execute(
            "select date, close from bars where code=? order by date", (f"IDX_{name}",)))
        idx[name] = {"dates": [r[0] for r in rows],
                     "close": np.array([r[1] for r in rows], dtype=float)}
    con.close()
    print(f"[load] 完成 {len(data)} 只", flush=True)
    for code in list(data):
        if code not in stocks:
            stocks[code] = {"name": code, "list_date": ""}
    return stocks, data, idx


def _pack(rows):
    return {
        "dates": [r[0] for r in rows],
        "o": np.array([r[1] for r in rows], dtype=float),
        "c": np.array([r[2] for r in rows], dtype=float),
        "h": np.array([r[3] for r in rows], dtype=float),
        "l": np.array([r[4] for r in rows], dtype=float),
        "v": np.array([r[5] for r in rows], dtype=float),
    }


# ---------------- 信号 ----------------

def find_signals(bars, cfg):
    """返回该股全部信号：dict(breakout_idx, signal_idx, level, vol_ratio, ...)"""
    c, v = bars["c"], bars["v"]
    n = len(c)
    L, W, hold, gain_min = cfg["L"], cfg["W"], cfg["hold_min"], cfg["gain_min"]
    if n < L + 6 + hold + W:
        return []

    # 前 L 日最高收盘（不含当日）
    win = np.lib.stride_tricks.sliding_window_view(c, L)
    rollmax = np.full(n, np.nan)
    rollmax[L:] = win.max(axis=1)[: n - L]

    # 前 5 日均量（不含当日）
    cs = np.concatenate(([0.0], np.cumsum(v)))
    volavg = np.full(n, np.nan)
    volavg[5:] = (cs[5:n] - cs[0:n - 5]) / 5.0

    out = []
    t = L + 5
    while t < n - 1:
        level = rollmax[t]
        prev_level = rollmax[t - 1] if t - 1 >= L else np.nan
        # 只取「首次突破」：前一日尚未站上自己的突破位
        if (np.isnan(level) or c[t] <= level
                or (not np.isnan(prev_level) and c[t - 1] > prev_level)):
            t += 1
            continue
        vr = v[t] / volavg[t] if volavg[t] and volavg[t] > 0 else np.nan
        if not np.isnan(vr) and cfg["vol_min"] <= vr <= cfg["vol_max"]:
            # 找第一个满足「站稳 hold 天且幅度达标」的观察日
            found = -1
            for k in range(t + hold - 1, min(t + W, n)):
                if np.any(c[t:k + 1] <= level):
                    break
                if c[k] / level - 1 >= gain_min:
                    found = k
                    break
            if found >= 0:
                out.append({
                    "breakout_idx": t, "signal_idx": found, "level": float(level),
                    "vol_ratio": float(vr), "gain": float(c[found] / level - 1),
                    "breakout_date": bars["dates"][t],
                    "signal_date": bars["dates"][found],
                })
                t = found + 1
            else:
                t += 1
            continue
        t += 1
    return out


# ---------------- 逐笔 ----------------

def simulate_trade(bars, sig, cfg):
    k = sig["signal_idx"]
    n = len(bars["c"])
    if k + 1 >= n:
        return None
    e = k + 1
    entry_open = bars["o"][e]
    gap = entry_open / bars["c"][k] - 1
    # 1e-9 容差：gap=开盘/昨收-1 的浮点除法会让恰好 ±3%/9.5% 漂移一个 ULP；
    # 规则是「超过」才放弃，边界值应放行；与 build_panel_from_db 同口径。
    if (entry_open <= 0 or gap > cfg["gap_max"] + 1e-9
            or gap < -cfg.get("gap_min", 0) - 1e-9):
        return None   # 大幅高开买不进 / 大幅低开不接

    level = sig["level"]
    entry = entry_open
    peak = bars["c"][e]
    exit_idx = None
    reason = "数据结束"
    for i in range(e, n):
        peak = max(peak, bars["c"][i])
        if cfg.get("level_stop", True) and bars["c"][i] < level:
            exit_idx, reason = i + 1, "跌破突破位"
        elif bars["c"][i] < peak * (1 - cfg["trail"]):
            exit_idx, reason = i + 1, "移动止盈"
        elif i - e + 1 >= cfg["max_hold"]:
            exit_idx, reason = i + 1, "持仓到期"
        if exit_idx is not None:
            break
    if exit_idx is None or exit_idx >= n:
        exit_idx = n - 1
        if reason == "数据结束":
            exit_price = bars["c"][exit_idx]
        else:
            exit_price = bars["c"][n - 1]
    else:
        exit_price = bars["o"][exit_idx]

    gross = exit_price / entry - 1
    net = gross - cfg["cost"]
    return {
        **sig,
        "code": None, "entry_date": bars["dates"][e], "entry": entry,
        "exit_date": bars["dates"][exit_idx], "exit": exit_price,
        "hold": exit_idx - e + 1, "ret": net, "reason": reason,
        "mfe": float(np.max(bars["c"][e:exit_idx + 1]) / entry - 1),
        "mae": float(np.min(bars["c"][e:exit_idx + 1]) / entry - 1),
    }


def forward_returns(bars, sig, bench, cfg):
    k = sig["signal_idx"]
    e = k + 1
    n = len(bars["c"])
    entry = bars["o"][e]
    out = {}
    for f in (5, 10, 20, 60):
        j = e + f - 1
        out[f"f{f}"] = float(bars["c"][j] / entry - 1) if j < n else None
    out["entry_date"] = bars["dates"][e]
    return out


# ---------------- 组合 ----------------

def build_equity(trades, idx_dates, bench_close, cfg):
    """事件驱动 + 每日盯市（用个股收盘序列）。"""
    date_pos = {d: i for i, d in enumerate(idx_dates)}
    n = len(idx_dates)
    cash = cfg["capital"]
    positions, pend = [], {}
    for tr in trades:
        di = date_pos.get(tr["entry_date"])
        if di is not None:
            pend.setdefault(di, []).append(tr)

    eq = np.zeros(n)
    fills = 0
    for di, d in enumerate(idx_dates):
        keep = []
        for p in positions:
            if p["exit_idx"] == di:
                cash += p["shares"] * p["exit_price"]
            else:
                keep.append(p)
        positions = keep
        # 先按当日收盘盯市（已有持仓）
        mv = 0.0
        for p in positions:
            b = p["bars"]
            i = p["i"]
            while i + 1 < len(b["dates"]) and b["dates"][i + 1] <= d:
                i += 1
            p["i"] = i
            mv += p["shares"] * b["c"][i]
        # 再按当前权益的 1/slots 建仓
        held = {p["code"] for p in positions}
        # 按代码固定顺序，避免用未来收益挑单（look-ahead）
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
            cash -= shares * buy_px
            held.add(tr["code"])
            positions.append({
                "code": tr["code"], "shares": shares,
                "exit_idx": date_pos.get(tr["exit_date"], n - 1),
                "exit_price": tr["exit"] * (1 - cfg["cost"] / 2), "bars": tr["_bars"],
                "i": tr["_entry_i"], "date": d,
            })
            mv += shares * buy_px
            tr["_used"] = True
            fills += 1
        eq[di] = cash + mv
    build_equity.last_fills = fills
    return eq


def build_equal_weight(trades, idx_dates, cfg):
    """等权全信号组合：每个信号等额参与，不设仓位上限，衡量筛选本身的 edge。"""
    n = len(idx_dates)
    date_pos = {d: i for i, d in enumerate(idx_dates)}
    daily = [[] for _ in range(n)]
    conc = [0] * n
    for tr in trades:
        b = tr["_bars"]
        e = tr["_entry_i"]
        de = date_pos.get(tr["entry_date"])
        if de is None:
            continue
        # 注意：退出日在「个股 bar 序列」中的位置，不是日历位置
        xi = bisect_left(b["dates"], tr["exit_date"])
        if xi >= len(b["dates"]) or b["dates"][xi] != tr["exit_date"]:
            xi = len(b["dates"]) - 1
        if xi <= e:
            continue
        entry = tr["entry"] * (1 + cfg["cost"] / 2)
        exitp = tr["exit"] * (1 - cfg["cost"] / 2)
        prev = entry
        for j in range(e, xi + 1):
            di = date_pos.get(b["dates"][j])
            if di is None:
                continue
            px = b["c"][j] if j < xi else exitp
            r = px / prev - 1
            daily[di].append(r)
            conc[di] += 1
            prev = px
    eq = np.zeros(n)
    eq[0] = cfg["capital"]
    for i in range(1, n):
        r = float(np.mean(daily[i])) if daily[i] else 0.0
        eq[i] = eq[i - 1] * (1 + r)
    return eq, float(np.mean([c for c in conc if c]))


# ---------------- 指标 ----------------

def metrics_from_equity(eq, idx_dates):
    eq = np.asarray(eq, float)
    rets = eq[1:] / eq[:-1] - 1
    total = eq[-1] / eq[0] - 1
    years = max((len(eq) - 1) / 244, 1e-9)
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1).min()
    vol = rets.std() * math.sqrt(244) if rets.size > 1 else 0
    sharpe = (rets.mean() * 244) / vol if vol else 0
    return {"total": total, "cagr": cagr, "mdd": dd, "sharpe": sharpe, "vol": vol}


def trade_stats(trades):
    rets = np.array([t["ret"] for t in trades])
    if not len(rets):
        return {}
    wins, losses = rets[rets > 0], rets[rets <= 0]
    return {
        "n": len(rets), "win": len(wins) / len(rets),
        "avg": rets.mean(), "median": float(np.median(rets)),
        "avg_win": wins.mean() if len(wins) else 0,
        "avg_loss": losses.mean() if len(losses) else 0,
        "pf": (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf"),
        "hold": statistics.fmean([t["hold"] for t in trades]),
        "best": rets.max(), "worst": rets.min(),
    }


# ---------------- 报告 ----------------

def svg_curve(strategy, bench, dates, title):
    W, H, P = 980, 300, 42
    strategy = np.asarray(strategy, dtype=float)
    bench = np.asarray(bench, dtype=float)
    sy = strategy / strategy[0]
    by = bench / bench[0]
    lo = min(sy.min(), by.min())
    hi = max(sy.max(), by.max())
    def xy(i, v):
        x = P + i * (W - 2 * P) / max(len(sy) - 1, 1)
        y = H - P - (v - lo) / ((hi - lo) or 1) * (H - 2 * P)
        return x, y
    s_pts = " ".join(f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}" for i, v in enumerate(sy))
    b_pts = " ".join(f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}" for i, v in enumerate(by))
    lines = []
    for frac in (0, .25, .5, .75, 1):
        v = lo + frac * (hi - lo)
        y = xy(0, v)[1]
        lines.append(f'<line x1="{P}" x2="{W-P}" y1="{y:.0f}" y2="{y:.0f}" stroke="#eceef1"/>'
                     f'<text x="6" y="{y+4:.0f}" font-size="10" fill="#9a9992">{(v-1)*100:+.0f}%</text>')
    step = max(len(dates) // 6, 1)
    xlabels = "".join(
        f'<text x="{xy(i,lo)[0]:.0f}" y="{H-18}" font-size="10" fill="#9a9992" text-anchor="middle">{dates[i][:7]}</text>'
        for i in range(0, len(dates), step))
    return f"""<svg viewBox="0 0 {W} {H}" style="width:100%;height:300px">
{''.join(lines)}
<polyline points="{b_pts}" fill="none" stroke="#b9bcc2" stroke-width="1.6"/>
<polyline points="{s_pts}" fill="none" stroke="#C43A2E" stroke-width="2.2"/>
{xlabels}
<text x="{W-P}" y="18" font-size="11" fill="#C43A2E" text-anchor="end">{title}（红）</text>
</svg>"""


def render_report(cfg, stats, port, bench_m, curve_svg, yearly, exits, fwd, sweep,
                  eqw, avg_conc, curve_eqw, meta):
    def pct(v): return "—" if v is None else f"{v*100:+.2f}%"
    def cls(v): return "pos" if (v or 0) > 0 else "neg"
    rows_y = "".join(
        f"<tr><td>{y}</td><td class='num'>{pct(v['ew'])}</td><td class='num'>{pct(v['s'])}</td>"
        f"<td class='num'>{pct(v['b'])}</td><td class='num'>{pct(v['ew']-v['b'])}</td>"
        f"<td class='num'>{v['n']}</td></tr>"
        for y, v in yearly)
    rows_sweep = "".join(
        f"<tr><td>{name}</td><td class='num'>{v['n']}</td><td class='num'>{v['win']*100:.1f}%</td>"
        f"<td class='num'>{pct(v['avg'])}</td><td class='num'>{v['hold']:.0f}</td>"
        f"<td class='num'>{pct(v['ew_total'])}</td><td class='num'>{pct(v['total'])}</td>"
        f"<td class='num'>{pct(v['mdd'])}</td></tr>"
        for name, v in sweep)
    rows_fwd = "".join(
        f"<tr><td>{k} 个交易日</td><td class='num'>{pct(v['avg'])}</td>"
        f"<td class='num'>{pct(v['bench'])}</td><td class='num'>{pct(v['excess'])}</td>"
        f"<td class='num'>{v['win']*100:.1f}%</td></tr>" for k, v in fwd.items())
    tot = sum(v["n"] for v in exits.values()) or 1
    rows_exit = "".join(f"<tr><td>{k}</td><td class='num'>{v['n']}</td>"
                        f"<td class='num'>{v['n']/tot*100:.1f}%</td>"
                        f"<td class='num'>{pct(v['avg'])}</td></tr>"
                        for k, v in exits.items())

    # 结论：按等权口径自动表述
    ew_excess = eqw["total"] - bench_m["total"]
    verdict = (
        f"等权全信号组合 {pct(eqw['total'])}（年化 {pct(eqw['cagr'])}，最大回撤 {pct(eqw['mdd'])}），"
        f"同期沪深300 {pct(bench_m['total'])}，超额 {pct(ew_excess)}；"
        f"单笔净期望 {pct(stats['avg'])}、胜率 {stats['win']*100:.1f}%、盈亏比 {stats['pf']:.2f}。"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>突破趋势跟踪 · 回测报告</title>
<style>
:root{{--ink:#2C2C2A;--ink2:#5F5E5A;--ink3:#888780;--line:#e3e5e8;--up:#C43A2E;--down:#1D7A4C}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#f5f6f8;color:var(--ink);font:14px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:30px 24px 70px}}
h1{{font-size:26px;font-weight:650;letter-spacing:-.02em}}
h2{{font-size:15px;font-weight:600;margin:28px 0 10px}}
.sub{{color:var(--ink3);font-size:12px;margin-top:6px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:20px 0}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.card .k{{font-size:11px;color:var(--ink3)}}
.card .v{{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}}
.card .s{{font-size:11px;color:var(--ink2)}}
.pos{{color:var(--up)}} .neg{{color:var(--down)}}
.box{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;font-size:11px;color:var(--ink3);font-weight:500;padding:8px 10px;border-bottom:1px solid var(--line);background:#fafbfc}}
td{{padding:8px 10px;border-bottom:1px solid #f2f3f5}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
.note{{font-size:12px;color:var(--ink2);line-height:1.9}}
.note li{{margin-left:18px}}
.tag{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;background:#EEEDFE;color:#534AB7;margin-right:6px}}
.verdict{{background:#fff;border:1px solid var(--line);border-left:3px solid #C43A2E;border-radius:10px;padding:14px 18px;margin:18px 0;font-size:13.5px}}
</style></head><body><div class="wrap">
<h1>突破趋势跟踪 · 回测报告</h1>
<div class="sub">{meta['period']} · 主板 {meta['stocks']} 只 · 信号 {stats['n']} 笔 · 数据：东方财富日线（后复权）</div>
<div class="verdict">{verdict}</div>
<h2>主口径：等权全信号（每个信号等额买入，不受仓位上限约束，衡量筛选本身的 edge）</h2>
<div class="cards">
  <div class="card"><div class="k">等权累计收益</div><div class="v {cls(eqw['total'])}">{pct(eqw['total'])}</div><div class="s">年化 {pct(eqw['cagr'])}</div></div>
  <div class="card"><div class="k">最大回撤</div><div class="v {cls(eqw['mdd'])}">{pct(eqw['mdd'])}</div><div class="s">Sharpe {eqw['sharpe']:.2f} · 波动 {eqw['vol']*100:.0f}%</div></div>
  <div class="card"><div class="k">单笔净期望</div><div class="v {cls(stats['avg'])}">{pct(stats['avg'])}</div><div class="s">胜率 {stats['win']*100:.1f}% · 盈亏比 {stats['pf']:.2f}</div></div>
  <div class="card"><div class="k">等权口径同时持仓</div><div class="v">{avg_conc:.0f} 只</div><div class="s">中位单笔 {pct(stats['median'])} · 平均持有 {stats['hold']:.0f} 天</div></div>
</div>
<div class="box">{curve_eqw}</div>
<div class="sub">沪深300 同期：{pct(bench_m['total'])}，最大回撤 {pct(bench_m['mdd'])}</div>

<h2>次口径：{meta['slots']} 仓实盘（每仓 1/{meta['slots']} 权益，同股不重复，容量受限）</h2>
<div class="cards">
  <div class="card"><div class="k">实盘口径累计</div><div class="v {cls(port['total'])}">{pct(port['total'])}</div><div class="s">年化 {pct(port['cagr'])}</div></div>
  <div class="card"><div class="k">最大回撤</div><div class="v {cls(port['mdd'])}">{pct(port['mdd'])}</div><div class="s">Sharpe {port['sharpe']:.2f}</div></div>
  <div class="card"><div class="k">沪深300</div><div class="v {cls(bench_m['total'])}">{pct(bench_m['total'])}</div><div class="s">最大回撤 {pct(bench_m['mdd'])}</div></div>
  <div class="card"><div class="k">实际成交 / 信号</div><div class="v">{meta['used']} / {stats['n']}</div><div class="s">容量只吃下 {meta['used']/stats['n']*100:.1f}%；{meta['unfillable']} 笔高开买不进</div></div>
</div>
<div class="box">{curve_svg}</div>

<h2>年度表现</h2>
<div class="box"><table><thead><tr><th>年份</th><th class="num">等权全信号</th><th class="num">{meta['slots']} 仓实盘</th>
<th class="num">沪深300</th><th class="num">等权超额</th><th class="num">信号数</th></tr></thead><tbody>{rows_y}</tbody></table></div>

<h2>信号前瞻收益（不设卖出规则，验证筛选本身有没有 edge）</h2>
<div class="box"><table><thead><tr><th>持有期</th><th class="num">平均收益</th><th class="num">同期沪深300</th>
<th class="num">超额</th><th class="num">胜率</th></tr></thead><tbody>{rows_fwd}</tbody></table></div>

<h2>退出方式分布</h2>
<div class="box"><table><thead><tr><th>退出原因</th><th class="num">笔数</th><th class="num">占比</th>
<th class="num">平均收益</th></tr></thead><tbody>{rows_exit}</tbody></table></div>

<h2>参数敏感性</h2>
<div class="box"><table><thead><tr><th>参数组</th><th class="num">信号数</th><th class="num">胜率</th>
<th class="num">平均收益</th><th class="num">平均持仓</th><th class="num">等权收益</th>
<th class="num">实盘收益</th><th class="num">实盘回撤</th></tr></thead>
<tbody>{rows_sweep}</tbody></table></div>

<h2>口径与局限</h2>
<div class="box note">
<span class="tag">基准参数</span>突破位=前 {cfg['L']} 日最高收盘；量比 {cfg['vol_min']}–{cfg['vol_max']} 倍；站稳 ≥{cfg['hold_min']} 日、幅度 ≥{cfg['gain_min']*100:.0f}%；突破窗口 {cfg['W']} 日；结构止损 {'开' if cfg.get('level_stop', True) else '关'}；移动止盈 {cfg['trail']*100:.0f}%；最长持仓 {cfg['max_hold']} 日；成本 {cfg['cost']*100:.2f}%/往返。
<ul>
<li>股票池为当前仍在交易的主板股票，已退市/长期停牌标的不在样本内，存在生存者偏差，实际结果通常应下调。</li>
<li>ST 状态只按当前名称剔除，历史 ST 期间未还原。</li>
<li>成交假设：信号次日开盘买入，开盘涨幅超过 9.5% 视为买不进；卖出按触发次日开盘。未考虑涨跌停封板、成交量约束和冲击成本。</li>
<li>价格采用后复权，除权日不会造成假突破；分红再投资未单独计息。</li>
<li>同一天多个信号、仓位已满时按代码顺序先到先得，不做额外择优，避免用未来收益挑单。</li>
<li>回测只检验规则的历史统计特征，不构成投资建议。</li>
</ul></div>

<div class="sub">逐笔明细：trades.csv　·　生成 {meta['generated']}</div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    for k in ("L", "W", "hold_min", "max_hold", "slots"):
        ap.add_argument(f"--{k}", type=int)
    for k in ("vol_min", "vol_max", "gain_min", "trail", "cost"):
        ap.add_argument(f"--{k}", type=float)
    a = ap.parse_args()
    cfg = dict(BASE)
    for k, v in vars(a).items():
        if v is not None and k in cfg:
            cfg[k] = v

    stocks, data, idx = load_data()
    bench = idx["沪深300"]
    dates_all = bench["dates"]
    date_pos = {d: i for i, d in enumerate(dates_all)}

    # 信号 + 逐笔
    all_trades = []
    blind = 0
    import time
    t0 = time.time()
    for code, b in data.items():
        if len(b["dates"]) < WARMUP_BARS:
            continue
        for sig in find_signals(b, cfg):
            k = sig["signal_idx"]
            if b["dates"][k] < EVAL_START:
                continue
            tr = simulate_trade(b, sig, cfg)
            if not tr:
                blind += 1
                continue
            if tr["code"] is None:
                tr["code"] = code
                tr["name"] = stocks.get(code, {}).get("name", code)
                tr["_bars"] = b
                tr["_entry_i"] = k + 1
            tr.update({k2: v for k2, v in forward_returns(b, sig, bench, cfg).items()})
            all_trades.append(tr)

    print(f"[signal] {len(all_trades)} 笔，跳过 {blind}，耗时 {time.time()-t0:.1f}s", flush=True)
    all_trades.sort(key=lambda t: (t["entry_date"], t["code"]))
    stats = trade_stats(all_trades)
    if not stats:
        sys.exit("无交易")

    print("[equity] 组合净值计算中", flush=True)
    port_eq = build_equity(all_trades, dates_all, bench["close"], cfg)
    print("[equity] 完成", flush=True)
    used = sum(1 for t in all_trades if t.get("_used"))
    start_i = date_pos[all_trades[0]["entry_date"]]
    port_eq_view = port_eq[start_i:]
    port = metrics_from_equity(port_eq_view, dates_all[start_i:])
    bench_window = bench["close"][start_i:]
    bench_curve = bench_window / bench_window[0] * port_eq_view[0]
    n = min(len(port_eq_view), len(bench_curve))
    bench_m = metrics_from_equity(bench_curve[:n], dates_all[start_i:start_i + n])
    curve = svg_curve(port_eq_view[:n].tolist(), bench_curve[:n].tolist(),
                      dates_all[start_i:start_i + n], "风险预算实盘口径（6 仓上限 · 单笔 2% 风险）")

    # 等权全信号组合：衡量筛选本身 edge（不设仓位上限）
    eqw, avg_conc = build_equal_weight(all_trades, dates_all, cfg)
    eqw_m = metrics_from_equity(eqw[start_i:], dates_all[start_i:])
    curve_eqw = svg_curve(eqw[start_i:start_i + n].tolist(), bench_curve[:n].tolist(),
                          dates_all[start_i:start_i + n], "等权全信号")

    # 年度
    yearly = []
    for y in sorted({t["entry_date"][:4] for t in all_trades}):
        ts = [t["ret"] for t in all_trades if t["entry_date"][:4] == y]
        i0 = next((i for i, d in enumerate(dates_all) if d.startswith(y)), None)
        i1 = next((i for i, d in enumerate(dates_all) if d.startswith(str(int(y) + 1))), len(dates_all))
        if i0 is None:
            continue
        s = port_eq[i1 - 1] / port_eq[i0] - 1
        ew = eqw[i1 - 1] / eqw[i0] - 1
        b = bench["close"][i1 - 1] / bench["close"][i0] - 1
        yearly.append((y, {"s": s, "ew": ew, "b": b, "n": len(ts)}))

    # 退出分布
    exits = {}
    for t in all_trades:
        e = exits.setdefault(t["reason"], {"n": 0, "sum": 0.0})
        e["n"] += 1
        e["sum"] += t["ret"]
    exits = {k: {"n": v["n"], "avg": v["sum"] / v["n"], "sum": v["sum"]} for k, v in exits.items()}

    # 前瞻
    fwd = {}
    for f in (5, 10, 20, 60):
        vals = [t[f"f{f}"] for t in all_trades if t.get(f"f{f}") is not None]
        if not vals:
            continue
        bs = []
        for t in all_trades:
            if t.get(f"f{f}") is None:
                continue
            i0 = date_pos.get(t["entry_date"])
            if i0 is None or i0 + f - 1 >= len(bench["close"]):
                continue
            bs.append(bench["close"][i0 + f - 1] / bench["close"][i0] - 1)
        avg = statistics.fmean(vals)
        bavg = statistics.fmean(bs) if bs else 0
        fwd[f] = {"avg": avg, "bench": bavg, "excess": avg - bavg,
                  "win": sum(1 for v in vals if v > 0) / len(vals)}

    # 参数敏感性
    sweep = []
    if a.sweep:
        variants = [("基准（结构止损+12%止盈）", {}),
                    ("无止盈（仅结构止损）", {"trail": 0.99}),
                    ("仅12%止盈（不设结构止损）", {"level_stop": False}),
                    ("仅20%止盈（不设结构止损）", {"level_stop": False, "trail": 0.20}),
                    ("固定持有20日", {"level_stop": False, "trail": 0.99, "max_hold": 20}),
                    ("固定持有60日", {"level_stop": False, "trail": 0.99, "max_hold": 60}),
                    ("无止盈+250日上限", {"trail": 0.99, "max_hold": 250})]
        for L in (10, 30):
            variants.append((f"突破位 {L} 日", {"L": L}))
        variants += [("量比 1.5–2.5", {"vol_max": 2.5}), ("站稳 2 日", {"hold_min": 2}),
                     ("站稳 5 日", {"hold_min": 5}), ("幅度 ≥2%", {"gain_min": 0.02}),
                     ("幅度 ≥5%", {"gain_min": 0.05}), ("成本 0.2%", {"cost": 0.002}),
                     ("突破当日买（不等站稳）", {"hold_min": 1, "gain_min": 0.0}),
                     ("突破当日买+纯12%止盈", {"hold_min": 1, "gain_min": 0.0, "level_stop": False})]
        for name, delta in variants:
            vcfg = {**cfg, **delta, "level_stop": delta.get("level_stop", cfg.get("level_stop", True))}
            ts, bl = [], 0
            for code, b in data.items():
                if len(b["dates"]) < WARMUP_BARS:
                    continue
                for sig in find_signals(b, vcfg):
                    if b["dates"][sig["signal_idx"]] < EVAL_START:
                        continue
                    tr = simulate_trade(b, sig, vcfg)
                    if not tr:
                        bl += 1
                        continue
                    tr["code"] = code
                    tr["_bars"] = b
                    tr["_entry_i"] = sig["signal_idx"] + 1
                    ts.append(tr)
            st = trade_stats(ts)
            eq = build_equity(ts, dates_all, bench["close"], vcfg)
            pm = metrics_from_equity(eq, dates_all)
            eq2, _ = build_equal_weight(ts, dates_all, vcfg)
            pm2 = metrics_from_equity(eq2, dates_all)
            sweep.append((name, {"n": st["n"], "win": st["win"], "avg": st["avg"],
                                 "hold": st["hold"], "total": pm["total"], "mdd": pm["mdd"],
                                 "ew_total": pm2["total"], "ew_mdd": pm2["mdd"]}))

    # 输出
    with open(OUT_TRADES, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["代码", "名称", "突破日", "信号日", "买入日", "买入价", "卖出日", "卖出价",
                    "收益", "持仓天数", "量比", "突破位", "退出原因", "MFE", "MAE"])
        for t in all_trades:
            w.writerow([t["code"], t.get("name", ""), t["breakout_date"], t["signal_date"],
                        t["entry_date"], f"{t['entry']:.2f}", t["exit_date"], f"{t['exit']:.2f}",
                        f"{t['ret']*100:.2f}%", t["hold"], f"{t['vol_ratio']:.2f}",
                        f"{t['level']:.2f}", t["reason"], f"{t['mfe']*100:.1f}%", f"{t['mae']*100:.1f}%"])

    summary = {"config": cfg, "signals": stats, "portfolio": port,
               "benchmark": bench_m, "yearly": yearly, "equal_weight": eqw_m,
               "avg_concurrent": avg_conc,
               "forward": {str(k): v for k, v in fwd.items()},
               "exits": exits, "sweep": sweep,
               "unfillable": blind, "stocks": len(data)}
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    period = f"{all_trades[0]['entry_date']} → {all_trades[-1]['exit_date']}"
    html = render_report(cfg, stats, port, bench_m, curve, yearly, exits, fwd, sweep, eqw_m, avg_conc, curve_eqw,
                         {"period": period, "stocks": len(data), "slots": cfg["slots"],
                          "unfillable": blind, "used": used,
                          "generated": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")})
    OUT_HTML.write_text(html, encoding="utf-8")

    print(f"信号 {stats['n']} 笔 | 胜率 {stats['win']*100:.1f}% | 平均 {stats['avg']*100:+.2f}% | "
          f"盈亏比 {stats['pf']:.2f} | 平均持仓 {stats['hold']:.0f} 天")
    print(f"等权全信号 {eqw_m['total']*100:+.1f}%（年化 {eqw_m['cagr']*100:+.1f}%，回撤 {eqw_m['mdd']*100:.1f}%，Sharpe {eqw_m['sharpe']:.2f}）")
    print(f"实盘口径 {port['total']*100:+.1f}%（年化 {port['cagr']*100:+.1f}%）| 最大回撤 {port['mdd']*100:.1f}% | "
          f"Sharpe {port['sharpe']:.2f} | 基准 {bench_m['total']*100:+.1f}%")
    print(f"→ {OUT_HTML.name} / {OUT_TRADES.name} / {OUT_JSON.name}")


if __name__ == "__main__":
    main()
