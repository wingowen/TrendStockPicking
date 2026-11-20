#!/usr/bin/env python3
"""从本地 market.db 生成突破面板数据（不再依赖问财接口）。

- 信号判定用后复权日线（除权不制造假突破），展示用不复权收盘（与行情软件一致）
- 一次扫描最近 N 个交易日，生成 breakout-data/<YYYYMMDD>.json 并汇总 breakout-data.js
- 左侧日期菜单来自交易日历，点哪天看哪天

用法：
    python3 build_panel_from_db.py                 # 最近 30 个交易日
    python3 build_panel_from_db.py --days 60
    python3 build_panel_from_db.py --date 2026-11-19
"""
import argparse
import json
import sys
import sqlite3
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backtest"))
from sector_trend import build_industry_indices, snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parent
DB = ROOT / "backtest" / "market.db"
OUT_DIR = ROOT / "breakout-data"
OUT_JS = ROOT / "breakout-data.js"

LOOKBACK = 20      # 突破位 = 前 N 日最高收盘
BREAK_WIN = 8      # 突破发生在近 N 个交易日内
VOL_MIN, VOL_MAX = 1.5, 2.0
HOLD_MIN = 3
GAIN_MIN = 0.03
SERIES_BARS = 60
HISTORY_START = "2025-01-01"


def load(con):
    stocks = {r[0]: {"name": r[1], "industry": r[2] or "", "market": r[3]}
              for r in con.execute("select code,name,industry,market from stocks")}
    bars, cur, buf = {}, None, []
    for code, d, o, c, h, l, v, amt in con.execute(
            "select code,date,open,close,high,low,volume,amount from bars "
            "where code not like 'IDX_%' and date>=? order by code,date", (HISTORY_START,)):
        if code != cur:
            if cur:
                bars[cur] = pack(buf)
            cur, buf = code, []
        buf.append((d, o, c, h, l, v, amt))
    if cur:
        bars[cur] = pack(buf)

    raw = {}
    for code, d, c in con.execute(
            "select code,date,close from bars_raw where date>=? order by code,date", (HISTORY_START,)):
        raw.setdefault(code, {})[d] = c

    idx = {}
    for name in ("上证指数", "深证成指", "创业板指", "沪深300"):
        rows = list(con.execute("select date,close,amount from bars where code=? order by date",
                                (f"IDX_{name}",)))
        idx[name] = {"dates": [r[0] for r in rows],
                     "close": [r[1] for r in rows],
                     "amount": [r[2] for r in rows]}
    return stocks, bars, raw, idx


def pack(rows):
    return {"dates": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "close": [r[2] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[4] for r in rows],
            "vol": [r[5] for r in rows],
            "amount": [r[6] for r in rows]}


def ymd(d):
    return d.replace("-", "")


def evaluate(b, k, raw_map):
    """对个股在 k 位置（bar 下标）判定状态；返回 (record, reasons) 或 None。"""
    c, v, dates = b["close"], b["vol"], b["dates"]
    if k < LOOKBACK + BREAK_WIN + 5:
        return None

    # 找「首个仍然有效」的突破：收盘站上前 LOOKBACK 日最高收盘、前一日还在该位置下方，
    # 且从突破日到今天一直没跌回突破位下方。取最早的那个，站稳天数才不会被后续新高重置
    i = level = None
    start = max(LOOKBACK, k - BREAK_WIN + 1)
    for t in range(start, k + 1):
        lv = max(c[t - LOOKBACK:t])
        if c[t] > lv and c[t - 1] <= lv and all(x > lv for x in c[t:k + 1]):
            i, level = t, lv
            break
    if i is None:
        return None

    vr = v[i] / (sum(v[i - 5:i]) / 5) if i >= 5 and sum(v[i - 5:i]) > 0 else None
    days_above = k - i + 1

    # 不复权换算：用突破日的复权/不复权比例把价格折回真实价位
    ratio = 1.0
    if raw_map:
        rd = raw_map.get(dates[i])
        if rd:
            ratio = rd / c[i]

    # 回踩：突破后价格缩量回落靠近突破位（最低价触及突破位上方 3% 以内），
    # 但收盘不破位；随后某日收盘重新站上「突破日到回踩日」的最高价 = 回踩确认
    retest = {"state": "none"}
    if i < k:
        zone = level * 1.02
        pull = next((t for t in range(i + 1, k + 1)
                     if b["low"][t] <= zone and c[t] > level), None)
        if pull is not None:
            shrink = v[pull] / v[i] if v[i] else None   # 回踩量 / 突破日量，<1 为缩量
            hi = max(b["high"][i:pull + 1])
            conf = next((s2 for s2 in range(pull + 1, k + 1) if c[s2] > hi), None)
            retest = {"state": "confirmed" if conf else "pull",
                      "date": ymd(dates[pull]),
                      "low": round(b["low"][pull] * ratio, 2),
                      "shrink": round(shrink, 2) if shrink else None,
                      "confirm_date": ymd(dates[conf]) if conf else None}
    level_raw = level * ratio
    close_raw = raw_map.get(dates[k], c[k] * ratio) if raw_map else c[k] * ratio
    breakout_close_raw = raw_map.get(dates[i], c[i] * ratio) if raw_map else c[i] * ratio
    gain = close_raw / level_raw - 1

    series = {}
    lo = max(0, k - SERIES_BARS + 1)
    for j in range(lo, k + 1):
        series[ymd(dates[j])] = round(raw_map.get(dates[j], c[j] * ratio), 2) if raw_map \
            else round(c[j] * ratio, 2)

    rec = {
        "code": None, "name": None, "industry": "", "board": "主板",
        "breakout_date": ymd(dates[i]), "level": round(level_raw, 2),
        "close": round(close_raw, 2), "gain": round(gain * 100, 2),
        "days_above": days_above, "vol_ratio": round(vr, 2) if vr else None,
        "breakout_vol": int(v[i] * 100), "avg_vol5": int(sum(v[i - 5:i]) / 5 * 100),
        "series": series, "breakout_close": round(breakout_close_raw, 2),
        "amount": int(b["amount"][i]), "retest": retest,
        "vol_ok": bool(vr is not None and VOL_MIN <= vr <= VOL_MAX),
        "stage": days_above,
        "announcements": [],
    }
    reasons = []
    if vr is None or not (VOL_MIN <= vr <= VOL_MAX):
        reasons.append(f"突破日量比 {vr:.2f} 倍，不在 {VOL_MIN}-{VOL_MAX} 倍区间" if vr else "突破日量比缺失")
    if days_above < HOLD_MIN:
        reasons.append(f"站稳突破位仅 {days_above} 个交易日，少于 {HOLD_MIN} 天")
    if gain < GAIN_MIN:
        reasons.append(f"距突破位仅 {gain*100:.2f}%，低于 {GAIN_MIN*100:.0f}%")
    return rec, reasons


def regime_for(idx, date):
    """大盘状态：上证指数相对 MA20/MA60 的位置 + 20/60 日动量 + 成交额量能。"""
    o = idx.get("上证指数")
    if not o:
        return None
    ds, c, a = o["dates"], o["close"], o["amount"]
    j = bisect_left(ds, date)
    if j >= len(ds) or ds[j] != date:
        return None
    ma20 = sum(c[max(0, j - 19):j + 1]) / min(20, j + 1)
    ma60 = sum(c[max(0, j - 59):j + 1]) / min(60, j + 1)
    mom20 = c[j] / c[j - 20] - 1 if j >= 20 else 0.0
    mom60 = c[j] / c[j - 60] - 1 if j >= 60 else 0.0
    base = sum(a[max(0, j - 20):j]) / max(1, min(20, j))
    volr = a[j] / base if base else None
    above20, above60 = c[j] > ma20, c[j] > ma60
    if above20 and above60 and mom20 > 0:
        label, note = "强", "指数在 20/60 日线上方且 20 日动量为正，顺势环境"
    elif (not above20) and (not above60):
        label, note = "弱", "指数跌到 20/60 日线下方，逆势环境，突破失败率显著上升"
    else:
        label, note = "震荡", "均线与动量不一致，信号需板块强度配合"
    score = 50 + (20 if above20 else -20) + (15 if above60 else -15) + (15 if mom20 > 0 else -15)
    score = max(0, min(100, score))
    return {"label": label, "note": note, "score": score,
            "close": round(c[j], 2), "ma20": round(ma20, 2), "ma60": round(ma60, 2),
            "mom20": round(mom20 * 100, 2), "mom60": round(mom60 * 100, 2),
            "vol_ratio": round(volr, 2) if volr else None,
            "above_ma20": above20, "above_ma60": above60}


PICK_MIN = 95   # 精选分数线：回测里 分≥95 且每日最多 3 只，单笔 +4.71%、胜率 44.9%
PICK_N = 3

TRAIL = 0.25       # 移动止盈：收盘从持仓峰值回撤 25% → 次日开盘卖
MAX_HOLD = 120     # 最长持有交易日
GAP_MAX = 0.095    # 信号次日开盘高开 >9.5% 放弃入场（涨停买不进）
GAP_MIN = 0.03     # 信号次日开盘低开 >3% 放弃入场（隔夜情绪反转，回测：被拦单笔 -3.32%）
CLOSED_WIN = 20    # 「近期平仓」回看窗口（交易日）
POS_SLOTS = 6      # 持仓池上限：最多同时持有 6 只（与原基准 6 仓口径一致，不做行业互斥）


def simulate_positions(stocks, bars, raw, idx, targets):
    """按策略模拟虚拟持仓池：最多同时持有 6 只（6 仓口径），不做资金分配。

    入场：某 breakout 首次满足全部确认条件（量比 1.5-2 / 站稳≥3 天 / 距突破位≥3%）
          的次日开盘买入；高开 >9.5% 或低开 >3% 放弃；同一 breakout 只做一次；
          满 6 只时跳过新信号（先入为主，同日多信号按代码顺序）。
          不做行业互斥——专项回测显示它会明显减分（6 仓 +46.24% → -8.00%）。
    出场：收盘从持仓期间最高收盘回撤 25% → 次日开盘卖；或持有满 120 个交易日 → 次日开盘卖。
    收益用后复权价计算（除权不失真），展示价按最新复权系数折回。未扣双边约 0.4% 成本。
    """
    cal = idx["上证指数"]["dates"]
    tindex = {d: t for t, d in enumerate(cal)}
    # 最新一日复权系数：全部展示字段统一用它折算（前复权式），保证与日K、彼此可比
    latest_ratio = {}
    for code, b in bars.items():
        rm = raw.get(code, {})
        for t in range(len(b["dates"]) - 1, -1, -1):
            d0 = b["dates"][t]
            if rm.get(d0) and b["close"][t]:
                latest_ratio[code] = rm[d0] / b["close"][t]
                break
    targets = [d for d in targets if d in tindex]
    pos_by_date = {d: [] for d in targets}
    closed_by_date = {d: [] for d in targets}
    reg_cache = {}

    def reg_label(date):
        if date not in reg_cache:
            r = regime_for(idx, date)
            reg_cache[date] = r["label"] if r else "—"
        return reg_cache[date]

    def day_ratio(rm, ds, c, j):
        r = rm.get(ds[j])
        if r and c[j]:
            return r / c[j]
        for t in range(j - 1, -1, -1):
            r = rm.get(ds[t])
            if r and c[t]:
                return r / c[t]
        return 1.0

    def ind_key(code):
        return stocks.get(code, {}).get("industry") or ("__" + code)

    # ---- 第一遍：逐票扫出全部「首次确认」入场事件（与 evaluate 同口径）----
    entries_by_date = {}
    for code, b in bars.items():
        ds, c, o, v = b["dates"], b["close"], b["open"], b["vol"]
        n = len(ds)
        if n < LOOKBACK + BREAK_WIN + 6:
            continue
        rm = raw.get(code, {})
        bi = blv = None
        last_signaled = None
        for k in range(n):
            if bi is not None and c[k] <= blv:
                bi = None
            if bi is None and k >= LOOKBACK:
                lv = max(c[k - LOOKBACK:k])
                if c[k] > lv and c[k - 1] <= lv:
                    vr = v[k] / (sum(v[k - 5:k]) / 5) if k >= 5 and sum(v[k - 5:k]) > 0 else None
                    if vr is not None and VOL_MIN <= vr <= VOL_MAX:
                        bi, blv = k, lv
            if bi is None:
                continue
            span = k - bi + 1
            if not (HOLD_MIN <= span <= BREAK_WIN) or ds[bi] == last_signaled:
                continue
            ratio = (rm[ds[bi]] / c[bi]) if rm.get(ds[bi]) else 1.0
            close_raw = rm.get(ds[k], c[k] * ratio)
            if close_raw / (blv * ratio) - 1 < GAIN_MIN:
                continue
            last_signaled = ds[bi]
            if k + 1 < n:
                gap = o[k + 1] / c[k] - 1
                if -GAP_MIN <= gap <= GAP_MAX:
                    entries_by_date.setdefault(ds[k + 1], []).append(
                        (code, {"entry_k": k + 1, "signal_date": ds[k],
                                "breakout": ds[bi], "level_adj": blv}))
    for d in entries_by_date:
        entries_by_date[d].sort(key=lambda x: x[0])   # 同日按代码顺序，避免未来函数

    # ---- 第二遍：按日历联合推进（6 仓上限）----
    active = {}          # code -> position state
    all_closed = []      # 已平仓交易
    # 必须从 HISTORY_START 起模拟（早于面板窗口的持仓也要在池子里）
    for d in cal:
        if d < HISTORY_START:
            continue
        # 开盘：先执行卖出（释放行业），再执行入场
        for code in list(active):
            pos = active[code]
            b = bars[code]
            ds = b["dates"]
            j = pos["j"]
            while j + 1 < len(ds) and ds[j + 1] <= d:
                j += 1
            pos["j"] = j
            if ds[j] != d:
                continue
            if pos.get("exit_k") == j:
                pos["exit_px"] = b["open"][j]
                pos["exit_date"] = d
                pos["hold_days"] = j - pos["entry_k"]
                all_closed.append(pos)
                del active[code]
        for code, ev in entries_by_date.get(d, []):
            if code in active or len(active) >= POS_SLOTS:
                continue
            ik = ind_key(code)
            b = bars[code]
            j = ev["entry_k"]
            if b["dates"][j] != d:
                continue
            active[code] = {**ev, "code": code, "ind": ik, "j": j,
                            "entry_date": d, "entry_px": b["open"][j],
                            "peak": b["open"][j], "exit_k": None,
                            "sig_k": None, "reason": None}
        # 收盘：更新峰值、检查出场
        for code, pos in active.items():
            b = bars[code]
            j = pos["j"]
            if b["dates"][j] != d:
                continue
            ck = b["close"][j]
            pos["peak"] = max(pos["peak"], ck)
            if pos["exit_k"] is None:
                hold_days = j - pos["entry_k"] + 1
                if ck <= pos["peak"] * (1 - TRAIL):
                    pos["exit_k"], pos["sig_k"], pos["reason"] = j + 1, j, "trail"
                elif hold_days >= MAX_HOLD:
                    pos["exit_k"], pos["sig_k"], pos["reason"] = j + 1, j, "maxhold"
        # 快照
        if d in pos_by_date:
            for code, pos in active.items():
                b = bars[code]
                ds, c = b["dates"], b["close"]
                j = pos["j"]
                if ds[j] != d:
                    continue
                info = stocks.get(code, {"name": code, "industry": ""})
                rday = latest_ratio.get(code, 1.0)
                pk = pos["peak"]
                pending = pos["exit_k"] is not None and j == pos["exit_k"] - 1
                pos_by_date[d].append({
                    "code": code, "name": info["name"], "industry": info["industry"],
                    "entry_date": ymd(pos["entry_date"]), "signal_date": ymd(pos["signal_date"]),
                    "breakout_date": ymd(pos["breakout"]),
                    "entry_px": round(pos["entry_px"] * rday, 2),
                    "level": round(pos["level_adj"] * rday, 2),
                    "regime_entry": reg_label(pos["signal_date"]),
                    "close": round(c[j] * rday, 2),
                    "pnl": round((c[j] / pos["entry_px"] - 1) * 100, 2),
                    "hold_days": j - pos["entry_k"] + 1,
                    "peak_px": round(pk * rday, 2),
                    "drawdown": round((c[j] / pk - 1) * 100, 2),
                    "trail_px": round(pk * (1 - TRAIL) * rday, 2),
                    "exit_pending": pending,
                    "exit_reason": pos["reason"] if pending else None})
            pos_by_date[d].sort(key=lambda x: -x["pnl"])

    # 近期平仓（按日历回看 CLOSED_WIN 个交易日）
    for d in targets:
        t = tindex[d]
        lo = cal[max(0, t - CLOSED_WIN + 1)]
        rows = []
        for tr in all_closed:
            if not (lo <= tr["exit_date"] <= d):
                continue
            code = tr["code"]
            b = bars[code]
            info = stocks.get(code, {"name": code, "industry": ""})
            rx = latest_ratio.get(code, 1.0)
            rows.append({
                "code": code, "name": info["name"], "industry": info["industry"],
                "entry_date": ymd(tr["entry_date"]), "signal_date": ymd(tr["signal_date"]),
                "breakout_date": ymd(tr["breakout"]),
                "entry_px": round(tr["entry_px"] * rx, 2),
                "level": round(tr["level_adj"] * rx, 2),
                "regime_entry": reg_label(tr["signal_date"]),
                "exit_date": ymd(tr["exit_date"]),
                "exit_px": round(tr["exit_px"] * rx, 2),
                "close": round(tr["exit_px"] * rx, 2),
                "pnl": round((tr["exit_px"] / tr["entry_px"] - 1) * 100, 2),
                "hold_days": tr["hold_days"], "reason": tr["reason"]})
        rows.sort(key=lambda x: (-int(x["exit_date"]), -x["pnl"]))
        closed_by_date[d] = rows
    return pos_by_date, closed_by_date


def rank_score(rec, sec, reg_label):
    """排序分：权重按 7 年回测分层实测的单笔期望校准（只用信号当日可知信息）。
    回踩状态不进分（回测里“回踩确认后入场”并不优于直接买，且会引入未来函数）。"""
    s = 0.0
    s += {"强": 40, "弱": 5}.get(reg_label or "", -20)   # 震荡 -20
    ss = sec["score"] if sec else 0
    s += {75: 25, 100: 10, 50: 5}.get(ss, 0)
    c = (sec["consec"] if sec else 0) or 0
    s += 12 if c >= 3 else (6 if c >= 1 else 0)
    d = rec["days_above"]
    s += 15 if 6 <= d <= 8 else (10 if d == 5 else 5)
    g = (rec["gain"] or 0) / 100.0   # rec["gain"] 存的是百分数
    s += 25 if g < 0.05 else (12 if g < 0.08 else (0 if g < 0.12 else -25))
    v = rec["vol_ratio"] or 1.75
    s += 10 if v < 1.8 else (6 if v >= 1.9 else 2)
    return round(s, 1)


def market_snapshot(stocks, bars, idx, date):
    out = {"indices": {}}
    for name, o in idx.items():
        ds = o["dates"]
        j = bisect_left(ds, date)
        if j >= len(ds) or ds[j] != date:
            continue
        prev = o["close"][j - 1] if j else o["close"][j]
        out["indices"][name] = {
            "code": {"上证指数": "000001.SH", "深证成指": "399001.SZ",
                     "创业板指": "399006.SZ", "沪深300": "000300.SH"}[name],
            "close": round(o["close"][j], 2),
            "chg": round(o["close"][j] / prev - 1, 4) * 100 if prev else 0,
            "amount": o["amount"][j],
        }
    up = down = limit_up = 0
    for code, b in bars.items():
        ds = b["dates"]
        j = bisect_left(ds, date)
        if j <= 0 or j >= len(ds) or ds[j] != date:
            continue
        chg = b["close"][j] / b["close"][j - 1] - 1
        if chg > 0:
            up += 1
        elif chg < 0:
            down += 1
        if chg >= 0.098:
            limit_up += 1
    out["breadth"] = {"up": up, "down": down, "limit_up": limit_up}
    return out


def build_day(date, stocks, bars, raw, idx, ind_series, real_flow=None, prev_codes=None):
    passed, rejected, watch = [], [], []
    reg = regime_for(idx, date)
    sec_rank = snapshot(ind_series, date, real_flow)
    top_sectors = sorted(
        [{"industry": k, "label": v["label"], "score": v["score"],
          "mom20": round(v["mom20"], 2), "flow5": round(v["flow5"] * 100, 1),
          "flow_src": v.get("flow_src", "proxy"), "consec": v["consec"]}
         for k, v in sec_rank.items()],
        key=lambda x: (-x["score"], -x["flow5"]))[:6]
    for code, b in bars.items():
        ds = b["dates"]
        if not ds or ds[-1] < date:
            continue
        j = bisect_left(ds, date)
        if j >= len(ds) or ds[j] != date:
            continue
        r = evaluate(b, j, raw.get(code, {}))
        if not r:
            continue
        rec, reasons = r
        info = stocks.get(code, {"name": code, "industry": ""})
        rec.update({"code": code, "name": info["name"], "industry": info["industry"]})
        sec = sec_rank.get(info["industry"])
        rec["sector"] = ({"label": sec["label"], "score": sec["score"],
                          "mom20": round(sec["mom20"], 2),
                          "flow5": round(sec["flow5"] * 100, 1),
                          "consec": sec["consec"]} if sec else None)
        rec["regime_ok"] = (reg["label"] != "弱") if reg else True
        if reasons:
            # 刚突破、量能达标、只是还没满 3 天 → 进观察池（不用等确认就能盯上）
            if rec["vol_ok"] and rec["days_above"] < HOLD_MIN:
                rec["rank_score"] = rank_score(rec, sec, reg["label"] if reg else "")
                rec["stage"] = rec["days_above"]
                watch.append(rec)
            else:
                rec.pop("series", None)   # 淘汰表不需要迷你走势
                rec.pop("announcements", None)
                rec.pop("retest", None)
                rejected.append({**rec, "reasons": reasons})
        else:
            rec["score"] = round(rank_score(rec, sec, reg["label"] if reg else ""))
            rec["rank_score"] = rank_score(rec, sec, reg["label"] if reg else "")
            rec["is_new"] = 0 if not prev_codes else (0 if code in prev_codes else 1)
            passed.append(rec)

    passed.sort(key=lambda x: -(x.get("rank_score") or 0))
    for i, p in enumerate(passed, 1):
        p["pick_rank"] = i if i <= PICK_N else 0
    watch.sort(key=lambda x: (-len(x["breakout_date"]), -x["days_above"], -x["rank_score"]))
    watch_new = sum(1 for w in watch if w["breakout_date"] == ymd(date))
    picks = [p for p in passed[:PICK_N] if p["rank_score"] >= PICK_MIN]
    picks_meta = {"threshold": PICK_MIN, "qualified": len(picks),
                  "candidates": sum(1 for p in passed if p["rank_score"] >= PICK_MIN)}
    rejected.sort(key=lambda x: -(x["gain"] or 0))
    screened = len(passed) + len(rejected)
    evaluated = len(passed) + sum(1 for r in rejected
                                  if r["vol_ratio"] and VOL_MIN <= r["vol_ratio"] <= VOL_MAX)
    snap = market_snapshot(stocks, bars, idx, date)
    return {
        "date": ymd(date), "date_cn": date,
        "market": snap,
        "regime": reg,
        "sectors": top_sectors,
        "funnel": {"screened": screened, "evaluated": evaluated,
                   "passed": len(passed), "rejected": len(rejected),
                   "confirmed": sum(1 for p in passed if p["retest"]["state"] == "confirmed")},
        "passed": passed, "rejected": rejected[:200],
        "picks": picks, "picks_meta": picks_meta,
        "watch": watch, "watch_meta": {"total": len(watch), "new_today": watch_new},
        "meta": {"rules": {"lookback": LOOKBACK, "break_win": BREAK_WIN, "vol_min": VOL_MIN,
                           "vol_max": VOL_MAX, "hold_min": HOLD_MIN, "gain_min": GAIN_MIN},
                 "source": "东方财富日线（本地 market.db）"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--date", help="只生成某一天，如 2026-11-19")
    ap.add_argument("--intraday", action="store_true",
                    help="允许生成当日盘中快照（未收盘，量比按当前成交量，只供盘中参考）")
    a = ap.parse_args()
    if not DB.exists():
        raise SystemExit(f"缺少 {DB}，先跑 backtest/fetch_data.py")

    con = sqlite3.connect(DB)
    stocks, bars, raw, idx = load(con)
    con.close()
    print("[sector] 计算板块等权指数", flush=True)
    bars_cv = {c: {"dates": b["dates"], "c": b["close"], "v": b["vol"]}
               for c, b in bars.items()}
    ind_series = build_industry_indices(
        bars_cv, {c: v["industry"] for c, v in stocks.items()}, idx["上证指数"]["dates"])

    # 板块真实主力净流入（只有近 120 日），能对上名字的行业优先用真实值
    print("[flow] 抓板块真实主力净流入", flush=True)
    try:
        from board_flow import load_real_flow, norm
        inds = sorted({v["industry"] for v in stocks.values() if v["industry"]})
        raw_flow = load_real_flow(inds)
        real_flow = {name: raw_flow[norm(name)] for name in ind_series
                     if norm(name) in raw_flow}
        print(f"[flow] 真实覆盖 {len(real_flow)}/{len(ind_series)} 个板块", flush=True)
    except Exception as e:
        real_flow = {}
        print(f"[flow] 抓取失败，退回代理口径：{e}", flush=True)

    cal = idx["上证指数"]["dates"]
    targets = [a.date] if a.date else cal[-a.days:]
    # 盘中抓的当日 K 线是未走完的半成品，收盘前不生成当日快照
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if today in targets and (now.hour, now.minute) < (15, 5) and not a.intraday:
        targets = [d for d in targets if d != today]
        print(f"{today} 尚未收盘，跳过当日快照")

    OUT_DIR.mkdir(exist_ok=True)
    days = {}
    prev_codes, new_codes = None, set()
    for date in targets:
        d = build_day(date, stocks, bars, raw, idx, ind_series, real_flow, prev_codes)
        prev_codes = {p["code"] for p in d["passed"]}
        if a.intraday and date == today:
            d["intraday"] = True
        if not d["passed"] and not d["rejected"]:
            continue
        days[ymd(date)] = d
        (OUT_DIR / f"{ymd(date)}.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{date}  候选 {d['funnel']['screened']:4d}  通过 {d['funnel']['passed']:3d}  "
              f"淘汰 {d['funnel']['rejected']:4d}", flush=True)

    # 虚拟持仓池：按策略从 HISTORY_START 起模拟，不限仓位/资金
    print("[pos] 模拟策略持仓池", flush=True)
    used_dates = [d for d in targets if ymd(d) in days]
    pos_by_date, closed_by_date = simulate_positions(stocks, bars, raw, idx, used_dates)
    for d in used_dates:
        day = days[ymd(d)]
        ps, cl = pos_by_date[d], closed_by_date[d]
        day["positions"] = ps
        day["positions_meta"] = {
            "count": len(ps),
            "avg_pnl": round(sum(p["pnl"] for p in ps) / len(ps), 2) if ps else 0,
            "winners": sum(1 for p in ps if p["pnl"] > 0),
            "pending_exit": sum(1 for p in ps if p["exit_pending"]),
            "trail": TRAIL, "max_hold": MAX_HOLD, "since": HISTORY_START,
        }
        day["closed_recent"] = cl
        (OUT_DIR / f"{ymd(d)}.json").write_text(
            json.dumps(day, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{d}  持仓 {len(ps):3d}  待卖出 {day['positions_meta']['pending_exit']}  "
              f"近{CLOSED_WIN}日平仓 {len(cl)}", flush=True)

    # 日K数据：给所有进过 主表/观察池/持仓池 的票带最近 60 根 OHLCV（统一按最新复权系数折回，与持仓表同口径）
    kcodes = set()
    for d in days.values():
        kcodes.update(x["code"] for x in d["passed"])
        kcodes.update(x["code"] for x in d.get("watch", []))
        kcodes.update(x["code"] for x in d.get("positions", []))
        kcodes.update(x["code"] for x in d.get("closed_recent", []))
    klines = {}
    for code in kcodes:
        b = bars.get(code)
        if not b:
            continue
        rm = raw.get(code, {})
        n = len(b["dates"])
        lo = max(0, n - 60)
        last_ratio = None
        for j in range(n - 1, -1, -1):
            r = rm.get(b["dates"][j])
            if r and b["close"][j]:
                last_ratio = r / b["close"][j]
                break
        if last_ratio is None:
            last_ratio = 1.0
        K = {"dates": [], "o": [], "h": [], "l": [], "c": [], "v": []}
        for j in range(lo, n):
            dt = b["dates"][j]
            K["dates"].append(ymd(dt))
            for k1, k2 in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")):
                K[k2].append(round(b[k1][j] * last_ratio, 2))
            K["v"].append(int(b["vol"][j]))
        klines[code] = K
    print(f"[kline] {len(klines)} 只 × 60 根", flush=True)

    all_dates = sorted({k for k in days})
    js = {"source": "东方财富日线 · 本地 market.db（后复权算信号；展示价按最新复权系数折回，前复权式）",
          "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
          "calendar": [ymd(d) for d in cal[-120:]],
          "dates": all_dates,
          "days": days,
          "klines": klines}
    OUT_JS.write_text("window.BREAKOUT_DATA = " +
                      json.dumps(js, ensure_ascii=False, separators=(",", ":")) + ";\n",
                      encoding="utf-8")
    print(f"生成 {len(days)} 个交易日 → {OUT_JS.name}")


if __name__ == "__main__":
    main()
