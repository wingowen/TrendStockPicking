#!/usr/bin/env python3
"""板块趋势：不用涨幅榜，用板块等权指数的均线位置 + 20 日动量 + 多日资金净流入。

- 板块指数：成分股日收益等权平均后累乘（后复权），≥5 只成分股才计
- 资金流代理：上涨日成交额记正、下跌日记负，净额 / 总成交额 = 资金净流入比
- 强：站上 MA20 且站上 MA60 且 20 日动量 > 0，且近 5 日资金净流入比 > 0
"""
from bisect import bisect_left

import numpy as np


def build_industry_indices(bars, industry, calendar, min_members=5):
    """返回 {行业: {"dates":..., "idx":..., "flow":..., "consec": int}}"""
    col = {d: i for i, d in enumerate(calendar)}
    names = sorted({v for v in industry.values() if v and v.strip()})
    rid = {n: i for i, n in enumerate(names)}
    n_ind, n_date = len(names), len(calendar)
    rsum = np.zeros((n_ind, n_date))
    cnt = np.zeros((n_ind, n_date))
    sflow = np.zeros((n_ind, n_date))
    svol = np.zeros((n_ind, n_date))

    for code, b in bars.items():
        ind = industry.get(code)
        if not ind or not ind.strip():
            continue
        r = rid[ind]
        ds, c, v = b["dates"], b["c"], b["v"]
        for j in range(1, len(ds)):
            p = col.get(ds[j])
            if p is None:
                continue
            ret = c[j] / c[j - 1] - 1
            rsum[r, p] += ret
            cnt[r, p] += 1
            sflow[r, p] += v[j] * (1 if ret > 0 else (-1 if ret < 0 else 0))
            svol[r, p] += v[j]

    out = {}
    for name, r in rid.items():
        idx = np.full(n_date, np.nan)
        flow = np.full(n_date, np.nan)
        level = 100.0
        started = False
        for p in range(n_date):
            if cnt[r, p] >= min_members:
                if not started:
                    idx[p] = level
                    started = True
                else:
                    level *= 1 + rsum[r, p] / cnt[r, p]
                    idx[p] = level
            elif started:
                idx[p] = level
            if svol[r, p] > 0:
                flow[p] = sflow[r, p] / svol[r, p]
            elif started:
                flow[p] = 0.0
        if not started:
            continue
        # 连续净流入天数
        consec = np.zeros(n_date, dtype=int)
        for p in range(n_date):
            if flow[p] is not None and not np.isnan(flow[p]) and flow[p] > 0:
                consec[p] = (consec[p - 1] if p else 0) + 1
        out[name] = {"dates": calendar, "idx": idx, "flow": flow, "consec": consec}
    return out


def snapshot(ind_series, date, real_flow=None):
    """返回该日全行业状态 {行业: {...}}

    real_flow: {行业: {日期: 主力净流入净占比}}，有就优先用真实主力资金，
    没有就退回代理口径（上涨日成交额记正、下跌日记负）。
    """
    out = {}
    for name, o in ind_series.items():
        ds = o["dates"]
        p = bisect_left(ds, date)
        if p >= len(ds) or ds[p] != date or np.isnan(o["idx"][p]):
            continue
        idx = o["idx"]
        ma20 = np.nanmean(idx[max(0, p - 19):p + 1])
        ma60 = np.nanmean(idx[max(0, p - 59):p + 1])
        mom20 = idx[p] / idx[p - 20] - 1 if p >= 20 and idx[p - 20] > 0 else 0.0
        mom60 = idx[p] / idx[p - 60] - 1 if p >= 60 and idx[p - 60] > 0 else 0.0
        flow5 = np.nanmean(o["flow"][max(0, p - 4):p + 1])
        flow_src = "proxy"
        consec = int(o["consec"][p])
        real = (real_flow or {}).get(name)
        if real:
            win = [x for x in ds[max(0, p - 4):p + 1] if x in real]
            if len(win) >= 4:
                flow5 = float(np.mean([real[x] for x in win]))
                flow_src = "real"
                consec = 0
                j = p
                while j >= 0 and ds[j] in real and real[ds[j]] > 0:
                    consec += 1
                    j -= 1
        above20 = idx[p] > ma20
        above60 = idx[p] > ma60
        score = ((25 if above20 else 0) + (25 if above60 else 0)
                 + (25 if mom20 > 0 else 0) + (25 if flow5 > 0 else 0))
        label = ("强" if (score >= 75 and flow5 > 0)
                 else ("弱" if score <= 25 else "中性"))
        out[name] = {"label": label, "score": score, "mom20": float(mom20 * 100),
                     "mom60": float(mom60 * 100), "flow5": float(flow5),
                     "flow_src": flow_src, "consec": consec,
                     "above20": bool(above20), "above60": bool(above60)}
    return out
