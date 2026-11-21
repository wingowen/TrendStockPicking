#!/usr/bin/env python3
"""突破趋势跟踪 · 统一组合引擎 v2

解决的问题（v1 的四类漏洞）：
1. 路径不可复现：v1 用代码字母序抢仓，+48.4% 只是众多路径里的上沿。
   v2 改成「按排序分填仓」，并把「随机填仓 / 年度 bootstrap / 去掉前 k 笔」
   都算出来，报区间而不是单点。
2. 实盘口径 ≠ 回测口径：v2 用一套参数跑完选股 → 排序 → 每日上限 → 仓位上限，
   回测与面板走同一条路径；分数权重用 walk-forward 拟合，样本外验证。
3. 资金约束用错价格：整手与资金约束改用「折回当前不复权尺度」的价格
   （后复权价 ÷ 今日复权因子），收益仍用后复权总回报口径。
4. 没有组合层风控：加入净值回撤降档（软阈值减半仓 / 硬阈值停止新开仓）
   与空仓现金收益（货币基金年化）。

用法：
    .venv-bt/bin/python engine_v2.py                # 全量实验 + 报告
    .venv-bt/bin/python engine_v2.py --quick        # 只跑基准对比
输出：
    report_v2.html / report_v2.json
"""
import argparse
import csv
import json
import math
import sqlite3
import statistics
from bisect import bisect_left
from pathlib import Path

import numpy as np

import run_backtest as R

ROOT = Path(__file__).resolve().parent
DB = ROOT / "market.db"
FEATURES = ROOT / "features.csv"
OUT_HTML = Path("/tmp/report_v2_scratch.html")
OUT_JSON = Path("/tmp/report_v2_scratch.json")

EVAL_START = "2020-01-02"
TRADING_DAYS = 244

# ---------------- v3 冻结口径（唯一对外口径） ----------------
# 排序分只用「训练期 2020-01-01 ~ 2021-12-31」拟合，系数在代码里硬编码冻结：
# 换参数必须重跑 run_v3.py 并更新这里的常量，否则测试 test_engine_v2 会失败。
V3_TRAIN = ("2020-01-01", "2022-01-01")
V3_MU = (0.420605, 0.066976, 3.508365, 1.730955)      # 强1/弱-1, gain, days_above, vol_ratio
V3_SD = (0.735744, 0.039762, 1.037386, 0.144746)
V3_W = (0.012879, 0.001884, -0.005449, 0.001036, 0.002682)   # 含截距

V3_SPEC = dict(
    slots=10, mode="equal", daily_cap=3, cost=0.004, cash_yield=0.015,
    dd_soft=0.10, dd_hard=0.15, dd_peak_window=252,
    trail=0.25, max_hold=120, regime_filter=("震荡", "弱"), min_score=None,
    capital=50000.0,
)


def score_v3(r):
    """v3 排序分：强市 + 贴近突破位 + 站稳久 + 量比 1.65-1.8。
    剔除的两个因子：板块评分（训练期 75 分最优、验证期不再单调）、
    板块连续流入（训练期方向与权重相反，验证期三档无差且历史仅 120 日真值）。"""
    g = 1.0 if r["regime"] == "强" else (-1.0 if r["regime"] == "弱" else 0.0)
    x = (g,
         min(r["gain"], 0.20),
         min(max(r["days_above"], 3), 9),
         r["vol_ratio"])
    z = [(v - m) / s for v, m, s in zip(x, V3_MU, V3_SD)]
    return V3_W[0] + sum(w * zi for w, zi in zip(V3_W[1:], z))


# ---------------- 数据 ----------------

def load_features():
    if not Path(FEATURES).exists():
        raise SystemExit(f"缺少因子缓存 {FEATURES}，先跑 build_features.py（约 5 分钟）")
    with open(FEATURES, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("entry", "level", "vol_ratio", "gain", "ret", "exit", "mfe", "mae"):
            r[k] = float(r[k])
        for k in ("days_above", "hold"):
            r[k] = int(float(r[k]))
        for k in ("sec_score", "sec_consec", "sec_flow5", "sec_mom20", "fwd20", "bench20"):
            r[k] = None if r[k] in ("", "None") else float(r[k])
        r["score1"] = score_v1(r)
    return rows


def load_calendar():
    con = sqlite3.connect(DB)
    dates = [r[0] for r in con.execute(
        "select date from bars where code='IDX_沪深300' order by date")]
    bench = np.array([r[0] for r in con.execute(
        "select close from bars where code='IDX_沪深300' order by date")], dtype=float)
    con.close()
    return dates, bench


def load_close_series(codes):
    """只取用到的股票日线收盘（后复权），用于逐日盯市。"""
    con = sqlite3.connect(DB)
    q = ",".join("?" * len(codes))
    series = {}
    for code, date, close in con.execute(
            f"select code,date,close from bars where code in ({q}) order by code,date", codes):
        series.setdefault(code, []).append((date, close))
    con.close()
    return series


def load_adj_factor():
    """adj(t)/raw(t) 的今日值：用 bars_raw 与 bars 的重叠区间算每只票的复权因子。
    历史价的「折回当前尺度」= adj(t)/factor_now，用于整手与资金约束。"""
    con = sqlite3.connect(DB)
    raw = {}
    for code, date, c in con.execute("select code,date,close from bars_raw"):
        raw[(code, date)] = c
    adj = {}
    for code, date, c in con.execute("select code,date,close from bars"):
        if code.startswith("IDX_"):
            continue
        adj[(code, date)] = c
    con.close()
    fac = {}
    for (code, date), rp in raw.items():
        ap = adj.get((code, date))
        if ap and rp and rp > 0:
            fac[code] = ap / rp          # 逐日算，取最后一个即为「今日因子」
    return fac


# ---------------- 排序分 ----------------

def score_v1(r):
    """v1 规则卡的排序分（沿用，作为对照）。"""
    s = 0.0
    s += {"强": 40, "弱": 5}.get(r["regime"], -20)
    s += {75: 25, 100: 10, 50: 5}.get(int(r["sec_score"]) if r["sec_score"] is not None else -1, 0)
    c = r["sec_consec"] or 0
    s += 12 if c >= 3 else (6 if c >= 1 else 0)
    d = r["days_above"]
    s += 15 if 6 <= d <= 8 else (10 if d == 5 else (5 if d <= 4 else 5))
    g = r["gain"]
    s += 25 if g < 0.05 else (12 if g < 0.08 else (0 if g < 0.12 else -25))
    v = r["vol_ratio"]
    s += 10 if v < 1.8 else (6 if v >= 1.9 else 2)
    return round(s, 1)


FEATS = ("regime_num", "sec_score", "sec_consec", "days_above", "gain", "vol_ratio")


def feature_matrix(rows, sec_mean=None):
    """把当日可知因子拼成矩阵（标准化用训练集均值方差）。"""
    out = []
    for r in rows:
        out.append([
            1.0 if r["regime"] == "强" else (-1.0 if r["regime"] == "弱" else 0.0),
            (r["sec_score"] or 0) / 100.0,
            min(max(r["sec_consec"] or 0, 0), 5),
            min(max(r["days_above"], 3), 9),
            min(r["gain"], 0.20),
            r["vol_ratio"],
        ])
    X = np.array(out, dtype=float)
    if sec_mean is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        return (X - mu) / sd, mu, sd
    mu, sd = sec_mean
    return (X - mu) / sd


def fit_score(train):
    """在训练集上拟合「20 日超额收益」的线性排序模型。只用训练集，避免未来函数。"""
    usable = [r for r in train if r["fwd20"] is not None and r["bench20"] is not None]
    if len(usable) < 200:
        return None
    X, mu, sd = feature_matrix(usable)
    X = np.column_stack([np.ones(len(X)), X])
    y = np.array([r["fwd20"] - r["bench20"] for r in usable], dtype=float)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "n": len(usable)}


def apply_score(rows, model):
    if model is None:
        return np.zeros(len(rows))
    X = feature_matrix(rows, (np.array(model["mu"]), np.array(model["sd"])))
    X = np.column_stack([np.ones(len(X)), X])
    return X @ np.array(model["w"])


# ---------------- 组合模拟 ----------------

def simulate(trades, dates, params, rank_scores=None):
    """统一组合引擎。

    trades: 已按 entry_date 排序的信号（含 entry/exit 后复权价、exit_date）
    params:
        slots          并发上限
        mode           'risk'（风险预算）| 'equal'（1/slots 等额）
        risk_pct       风险预算占比
        trail          移动止盈距离（只用于折算仓位与真实风险，出场已由信号给定）
        daily_cap      每日最多新开仓数
        min_score      最低排序分（None 表示不限）
        cash_yield     空仓现金年化收益
        dd_soft/dd_hard 净值回撤降档（软：半仓；硬：停止新开仓）
        lot_price      每只票的「折回当前尺度」价格函数
        cost           单边成本
    """
    dp = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    capital = params.get("capital", 50000.0)
    cash = capital
    positions = []          # dict(code, units, exit_idx, exit_px)
    pending = {}
    for t in trades:
        di = dp.get(t["entry_date"])
        if di is not None:
            pending.setdefault(di, []).append(t)

    eq = np.zeros(n)
    filled = []             # 真正成交的信号（此前 avg_trade 误用全部候选，已修）
    eq_hist = []            # 用于滚动峰值（避免全历史峰值把策略永久锁死）
    dd_win = int(params.get("dd_peak_window", 252))
    fills = 0
    skipped_lot = 0
    exposure = np.zeros(n)
    last_close = {}
    closes = params["closes"]
    lot_price = params["lot_price"]

    for di, d in enumerate(dates):
        # 1) 开盘按信号卖出
        keep = []
        for p in positions:
            if p["exit_idx"] == di:
                cash += p["units"] * p["exit_px"] * (1 - params["cost"] / 2)
            else:
                keep.append(p)
        positions = keep

        # 2) 用「昨日收盘」权益定仓（不能用当日收盘给当日开盘定价）
        mv = 0.0
        for p in positions:
            i = p["i"]
            b = closes[p["code"]][0]
            while i + 1 < len(b) and b[i + 1] < d:
                i += 1
            p["i"] = i
            mv += p["units"] * closes[p["code"]][1][i]
        equity = cash + mv
        eq_hist.append(equity)
        peak_ref = max(eq_hist[-dd_win:]) if eq_hist else capital
        dda = equity / peak_ref - 1 if peak_ref > 0 else 0
        if params.get("dd_hard") is not None and dda <= -params["dd_hard"]:
            size_factor = 0.0
        elif params.get("dd_soft") is not None and dda <= -params["dd_soft"]:
            size_factor = 0.5
        else:
            size_factor = 1.0

        # 3) 按排序分填仓
        cand = pending.get(di, [])
        if rank_scores is not None and cand:
            ordv = [(-rank_scores.get((t["code"], t["entry_date"]), -1e18), t["code"]) for t in cand]
            cand = [t for _, t in sorted(zip(ordv, cand), key=lambda x: x[0])]
        else:
            cand = sorted(cand, key=lambda t: t["code"])
        new = 0
        held = {p["code"] for p in positions}
        for t in cand:
            if len(positions) >= params["slots"] or cash <= 0 or new >= params["daily_cap"]:
                break
            if t["code"] in held:
                continue
            if params.get("min_score") is not None:
                if (rank_scores or {}).get((t["code"], t["entry_date"]), -1e18) < params["min_score"]:
                    continue
            if params.get("regime_filter") and t["regime"] in params["regime_filter"]:
                continue
            if size_factor <= 0:
                break
            if params["mode"] == "equal":
                weight = 1.0 / params["slots"]
            else:
                weight = min(params["risk_pct"] / max(params["trail"], 1e-6), 1.0 / params["slots"])
            budget = min(equity * weight * size_factor, cash)
            px_lot = lot_price(t)
            lots = int(budget / (px_lot * 100)) if px_lot > 0 else 0
            if lots <= 0:
                skipped_lot += 1
                continue
            used = lots * 100 * px_lot        # 元
            cash -= used * (1 + params["cost"] / 2)
            units = used / t["entry"]         # 后复权口径的份额
            positions.append({
                "code": t["code"], "units": units,
                "exit_idx": dp.get(t["exit_date"], n - 1),
                "exit_px": t["exit"],
                "i": bisect_left(closes[t["code"]][0], t["entry_date"]),
            })
            held.add(t["code"])
            new += 1
            fills += 1
            filled.append(t)

        # 4) 收盘盯市
        mv = 0.0
        for p in positions:
            i = p["i"]
            b = closes[p["code"]][0]
            while i + 1 < len(b) and b[i + 1] <= d:
                i += 1
            p["i"] = i
            mv += p["units"] * closes[p["code"]][1][i]
        # 5) 空仓现金收益
        cash *= 1 + params.get("cash_yield", 0.0) / TRADING_DAYS
        eq[di] = cash + mv
        exposure[di] = mv / eq[di] if eq[di] > 0 else 0

    return {"eq": eq, "fills": fills, "skipped_lot": skipped_lot,
            "filled": filled, "avg_exposure": float(exposure.mean())}


# ---------------- 指标 ----------------

def metrics(eq, dates, start=EVAL_START):
    i0 = next((i for i, d in enumerate(dates) if d >= start), 0)
    e = np.asarray(eq[i0:], float)
    if len(e) < 3 or e[0] <= 0:
        return {}
    r = e[1:] / e[:-1] - 1
    total = e[-1] / e[0] - 1
    years = max((len(e) - 1) / TRADING_DAYS, 1e-9)
    cagr = (e[-1] / e[0]) ** (1 / years) - 1 if e[-1] > 0 else -1
    peak = np.maximum.accumulate(e)
    mdd = float((e / peak - 1).min())
    vol = float(r.std() * math.sqrt(TRADING_DAYS)) if r.size > 1 else 0
    sharpe = float(r.mean() * TRADING_DAYS / vol) if vol else 0
    tstat = float(r.mean() / r.std() * math.sqrt(len(r))) if r.std() else 0
    underwater = int((e / peak - 1 < 0).sum())
    return {"total": total, "cagr": cagr, "mdd": mdd, "sharpe": sharpe, "vol": vol,
            "tstat": tstat, "underwater_days": underwater, "days": len(e)}


def trade_stats(trades):
    r = np.array([t["ret"] for t in trades], dtype=float)
    if not len(r):
        return {}
    return {"n": len(r), "avg": float(r.mean()), "med": float(np.median(r)),
            "win": float((r > 0).mean())}


# ---------------- 实验 ----------------

def prep(dates, rows):
    codes = sorted({r["code"] for r in rows})
    raw_series = load_close_series(codes)
    closes = {}
    for c, lst in raw_series.items():
        closes[c] = ([d for d, _ in lst], np.array([v for _, v in lst], float))
    fac = load_adj_factor()

    # 折回当前不复权尺度的价格
    adj_by_code = {}
    for r in rows:
        adj_by_code.setdefault(r["code"], {})[r["entry_date"]] = r["entry"]
    lot_px = {}
    for code, dd in adj_by_code.items():
        f = fac.get(code)
        for d, p in dd.items():
            lot_px[(code, d)] = (p / f) if f else p
    return closes, (lambda t: lot_px.get((t["code"], t["entry_date"]), t["entry"]))


def run_cfg(rows, dates, closes, lot_price, params, rank=None, start=None):
    start = start or EVAL_START
    tr = [r for r in rows if r["entry_date"] >= start]
    sim = simulate(tr, dates, dict(params, closes=closes, lot_price=lot_price), rank)
    m = metrics(sim["eq"], dates, start=start)
    st = trade_stats(sim["filled"])
    m.update({"fills": sim["fills"], "skipped_lot": sim["skipped_lot"],
              "avg_exposure": sim["avg_exposure"], "avg_trade": st.get("avg"),
              "win": st.get("win"), "signals": len(tr)})
    return sim["eq"], m


def walk_forward(rows, dates, closes, lot_price, params, folds=(2022, 2023, 2024, 2025, 2026)):
    """逐年样本外：第 y 年只用 y 年之前的数据拟合排序权重，当年交易，收益连乘。

    每年年初重置资金（不跨年持仓），把各年收益拼接成一条连续的样本外净值。
    """
    full = np.ones(len(dates))
    cum = 1.0
    out = []
    for y in folds:
        train = [r for r in rows if EVAL_START <= r["entry_date"] < f"{y}-01-01"]
        model = fit_score(train)
        test = [r for r in rows if f"{y}-01-01" <= r["entry_date"] < f"{y+1}-01-01"]
        if not test or model is None:
            continue
        sc = {(r["code"], r["entry_date"]): s for r, s in zip(test, apply_score(test, model))}
        sim = simulate(test, dates, dict(params, closes=closes, lot_price=lot_price), sc)
        i0 = next((i for i, d in enumerate(dates) if d >= f"{y}-01-01"), None)
        if i0 is None:
            continue
        i1 = next((i for i, d in enumerate(dates) if d >= f"{y+1}-01-01"), len(dates))
        e = sim["eq"][i0:i1]
        if e[0] <= 0:
            continue
        e = e / e[0]
        full[i0:i1] = cum * e
        cum *= e[-1]
        out.append((str(y), float(e[-1] - 1), sim["fills"], len(test)))
    return full, out


def bootstrap_years(rows, dates, closes, lot_price, params, rank, n=200, seed=7):
    """年度 bootstrap：按年重采样本，给出年化分布。衡量「是否只靠某一年」。"""
    rnd = np.random.default_rng(seed)
    years = sorted({r["entry_date"][:4] for r in rows if r["entry_date"] >= EVAL_START})
    by_year = {y: [r for r in rows if r["entry_date"][:4] == y] for y in years}
    res = []
    for _ in range(n):
        pick = rnd.choice(years, size=len(years), replace=True)
        yrs = []
        for y in pick:
            yrs.extend(by_year[y])
        eq, _ = run_cfg(yrs, dates, closes, lot_price, params, rank)
        m = metrics(eq, dates)
        if m:
            res.append((m["total"], m["cagr"], m["mdd"]))
    a = np.array(res)
    return {"total_med": float(np.median(a[:, 0])), "total_p5": float(np.percentile(a[:, 0], 5)),
            "total_p95": float(np.percentile(a[:, 0], 95)),
            "cagr_med": float(np.median(a[:, 1])), "cagr_p5": float(np.percentile(a[:, 1], 5)),
            "cagr_p95": float(np.percentile(a[:, 1], 95)),
            "mdd_med": float(np.median(a[:, 2]))}


def shuffle_paths(rows, dates, closes, lot_price, params, n=30, seed=11):
    """无排序时随机填仓的路径分布（说明 v1 的数字为什么不可复现）。"""
    rnd = np.random.default_rng(seed)
    vals = []
    for i in range(n):
        sc = {(r["code"], r["entry_date"]): rnd.random() for r in rows}
        eq, _ = run_cfg(rows, dates, closes, lot_price, params, sc)
        m = metrics(eq, dates)
        vals.append((m["total"], m["cagr"], m["mdd"]))
    a = np.array(vals)
    return {"cagr_med": float(np.median(a[:, 1])), "cagr_p5": float(np.percentile(a[:, 1], 5)),
            "cagr_p95": float(np.percentile(a[:, 1], 95)),
            "positive_frac": float((a[:, 0] > 0).mean())}


# ---------------- 报告 ----------------

def pct(v, d=1):
    return "—" if v is None else f"{v*100:+.{d}f}%"


def render(res, wf, bs, sh, meta):
    rows_cfg = "".join(
        f"<tr><td>{name}</td><td class='num'>{m.get('fills','')}</td>"
        f"<td class='num'>{pct(m.get('total'))}</td><td class='num'>{pct(m.get('cagr'))}</td>"
        f"<td class='num'>{pct(m.get('mdd'))}</td><td class='num'>{m.get('sharpe',0):.2f}</td>"
        f"<td class='num'>{pct(m.get('avg_trade'))}</td><td class='num'>{m.get('win',0)*100:.1f}%</td>"
        f"<td class='num'>{m.get('avg_exposure',0)*100:.0f}%</td></tr>"
        for name, m in res)
    rows_wf = "".join(
        f"<tr><td>{y}</td><td class='num'>{pct(r)}</td><td class='num'>{f}</td>"
        f"<td class='num'>{n}</td></tr>" for y, r, f, n in wf)
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>突破跟踪 · 统一引擎 v2</title><style>
body{{background:#f5f6f8;color:#2c2c2a;font:14px/1.65 -apple-system,"PingFang SC",sans-serif;margin:0}}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 24px 70px}}
h1{{font-size:24px;margin:0 0 6px}} h2{{font-size:15px;margin:26px 0 10px}}
.sub{{color:#888780;font-size:12px}}
.box{{background:#fff;border:1px solid #e3e5e8;border-radius:12px;padding:16px 18px;margin:12px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;font-size:11px;color:#888780;font-weight:500;padding:8px 10px;border-bottom:1px solid #e3e5e8}}
td{{padding:8px 10px;border-bottom:1px solid #f2f3f5}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
.note{{font-size:12.5px;color:#5f5e5a;line-height:1.9}} .note li{{margin-left:18px}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}}
.card{{background:#fff;border:1px solid #e3e5e8;border-radius:12px;padding:12px 14px}}
.card .k{{font-size:11px;color:#888780}} .card .v{{font-size:20px;font-weight:650}}
</style></head><body><div class="wrap">
<h1>突破趋势跟踪 · 统一组合引擎 v2</h1>
<div class="sub">{meta['period']} · 信号 {meta['n']} 笔 · 一套参数跑完「选股→排序→每日上限→仓位上限→风控」</div>
<h2>一、口径对比（同一引擎，只换填仓规则）</h2>
<div class="box"><table><thead><tr><th>口径</th><th class="num">成交</th><th class="num">累计</th>
<th class="num">年化</th><th class="num">最大回撤</th><th class="num">Sharpe</th>
<th class="num">单笔均值</th><th class="num">胜率</th><th class="num">平均仓位</th></tr></thead>
<tbody>{rows_cfg}</tbody></table></div>
<h2>二、逐年样本外（权重只用当年之前的数据拟合）</h2>
<div class="box"><table><thead><tr><th>年份</th><th class="num">当年收益</th>
<th class="num">成交</th><th class="num">候选</th></tr></thead><tbody>{rows_wf}</tbody></table></div>
<h2>三、稳健性</h2>
<div class="box note"><ul>
<li>随机填仓 30 次：年化中位 {pct(sh['cagr_med'])}，5–95% 区间 {pct(sh['cagr_p5'])} ~ {pct(sh['cagr_p95'])}，为正比例 {sh['positive_frac']*100:.0f}%。</li>
<li>年度 bootstrap {bs.get('n',200)} 次：累计中位 {pct(bs['total_med'])}，5–95% 区间 {pct(bs['total_p5'])} ~ {pct(bs['total_p95'])}；年化中位 {pct(bs['cagr_med'])}。</li>
<li>样本外各年收益见上表；任何一个模型如果只在某一年有效，这一栏会直接暴露。</li>
</ul></div>
<h2>四、v2 改了什么</h2>
<div class="box note"><ul>
<li>填仓规则从「代码字母序」改为「排序分降序」，与面板的每日精选一致。</li>
<li>排序权重用 walk-forward 线性拟合（20 日超额收益），不再手写分档打分。</li>
<li>整手与资金约束改用「折回当前不复权尺度」的价格，收益仍按后复权总回报。</li>
<li>新增净值回撤降档（软/硬阈值）与空仓现金收益，二者都可以单独关掉做对照。</li>
<li>所有结论报区间（随机路径 / 年度 bootstrap / 样本外），不再报单一路径的点估计。</li>
</ul></div>
<div class="sub">生成 {meta['generated']}</div></div></body></html>"""


def main():
    """草稿诊断台：只用于探索「排序分/并发/闸门」的敏感性，**不是官方数字**。

    官方口径与对外数字唯一来源是 run_v3.py（spec_v3.json）。
    输出写到 /tmp，避免草稿数字留在仓库里被误读。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print("[草稿] 本脚本含全样本拟合对照，数字不可对外引用；官方口径见 run_v3.py")

    rows = load_features()
    dates, bench = load_calendar()
    closes, lot_price = prep(dates, rows)
    print(f"[data] 信号 {len(rows)}，日历 {len(dates)} 天")

    base = dict(slots=6, mode="risk", risk_pct=0.02, trail=0.25, daily_cap=3,
                cost=0.004, cash_yield=0.0, dd_soft=None, dd_hard=None, min_score=None)
    res = []

    # 1) v1 代码顺序（复现头条）
    eq, m = run_cfg(rows, dates, closes, lot_price, base, None)
    res.append(("v1 代码顺序 6 仓（头条口径）", m))

    # 2) v1 排序分 + 95 线 + 每日≤3
    sc1 = {(r["code"], r["entry_date"]): r["score1"] for r in rows}
    eq, m = run_cfg(rows, dates, closes, lot_price, dict(base, min_score=95), sc1)
    res.append(("v1 手写排序分 ≥95 · 6 仓", m))

    # 3) 只按「距突破位」升序（最朴素的单调规则）
    sc_gain = {(r["code"], r["entry_date"]): -r["gain"] for r in rows}
    eq, m = run_cfg(rows, dates, closes, lot_price, base, sc_gain)
    res.append(("按距突破位升序 · 6 仓", m))

    # 4) 全样本拟合的排序模型（对照，用来暴露过拟合）
    model_all = fit_score(rows)
    sc_all = {(r["code"], r["entry_date"]): s for r, s in
              zip(rows, apply_score(rows, model_all))}
    eq, m = run_cfg(rows, dates, closes, lot_price, base, sc_all)
    res.append(("全样本拟合排序分 · 6 仓（过拟合对照）", m))

    # 5) 放宽并发：10 仓（降低路径依赖）
    for slots in (10, 14):
        eq, m = run_cfg(rows, dates, closes, lot_price,
                        dict(base, slots=slots, daily_cap=3), sc_all)
        res.append((f"全样本排序分 · {slots} 仓", m))

    # 6) 加现金收益
    eq, m = run_cfg(rows, dates, closes, lot_price,
                    dict(base, cash_yield=0.015), sc_all)
    res.append(("+ 空仓现金 1.5%", m))

    # 7) 加净值回撤降档
    eq, m = run_cfg(rows, dates, closes, lot_price,
                    dict(base, dd_soft=0.10, dd_hard=0.15), sc_all)
    res.append(("+ 回撤 10%/15% 降档", m))

    # 8) 组合：现金 + 降档 + 10 仓
    chosen = dict(base, slots=10, cash_yield=0.015, dd_soft=0.10, dd_hard=0.15)
    eq, m = run_cfg(rows, dates, closes, lot_price, chosen, sc_all)
    res.append(("v2 推荐口径（10 仓+现金+降档）", m))

    # 样本外
    wf_full, wf = walk_forward(rows, dates, closes, lot_price, chosen)
    wf_metrics = metrics(wf_full, dates) if wf else {}

    sh = shuffle_paths(rows, dates, closes, lot_price, base, n=10 if a.quick else 30)
    bs = bootstrap_years(rows, dates, closes, lot_price, chosen, sc_all, n=60 if a.quick else 200)

    meta = {"n": len(rows), "period": f"{min(r['entry_date'] for r in rows)} → "
            f"{max(r['exit_date'] for r in rows)}",
            "generated": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}
    OUT_HTML.write_text(render(res, wf, bs, sh, meta), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "configs": {k: v for k, v in res}, "walk_forward": wf, "wf_metrics": wf_metrics,
        "bootstrap": bs, "shuffle": sh, "chosen": chosen}, ensure_ascii=False, indent=1,
        default=float), encoding="utf-8")

    print("\n口径".ljust(34), "成交  累计     年化    回撤    Sharpe  单笔    胜率   仓位")
    for name, m in res:
        print(f"{name:<34}{m.get('fills',0):>4}  {pct(m.get('total'),1):>8} "
              f"{pct(m.get('cagr'),2):>8} {pct(m.get('mdd'),1):>7} "
              f"{m.get('sharpe',0):>6.2f} {pct(m.get('avg_trade'),2):>7} "
              f"{m.get('win',0)*100:>5.1f}% {m.get('avg_exposure',0)*100:>4.0f}%")
    print("\n样本外:", " ".join(f"{y}:{pct(r)}" for y, r, *_ in wf))
    print("样本外汇总:", pct(wf_metrics.get("total")), pct(wf_metrics.get("cagr")),
          pct(wf_metrics.get("mdd")) if wf_metrics else "")
    print("随机填仓:", sh)
    print("年度 bootstrap:", bs)
    print(f"→ {OUT_HTML.name} / {OUT_JSON.name}")


if __name__ == "__main__":
    main()
