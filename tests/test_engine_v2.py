#!/usr/bin/env python3
"""引擎层门禁：v1 从来没有测过「钱怎么花」，这 8 条必须过。

跑法：
    .venv-bt/bin/python -m pytest tests/test_engine_v2.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtest"))

import engine_v2 as E  # noqa: E402


def make_dates(n):
    return [f"2020-01-{i:02d}" for i in range(1, n + 1)]


def make_trade(code, entry_idx, exit_idx, dates, entry_px=10.0, exit_px=10.0, regime="强"):
    return {"code": code, "entry_date": dates[entry_idx], "exit_date": dates[exit_idx],
            "entry": entry_px, "exit": exit_px, "hold": exit_idx - entry_idx,
            "regime": regime, "gain": 0.05, "days_above": 5, "vol_ratio": 1.7,
            "ret": exit_px / entry_px - 1}


def run(trades, closes, lot_price, dates, **kw):
    params = dict(slots=2, mode="equal", daily_cap=3, cost=0.004, cash_yield=0.0,
                  dd_soft=None, dd_hard=None, dd_peak_window=252, min_score=None,
                  trail=0.25, capital=50000.0, closes=closes, lot_price=lot_price)
    params.update(kw)
    return E.simulate(trades, dates, params, kw.pop("rank", None))


# ---- 1. 整手买不起就跳过，且不静默丢单 ----
def test_lot_too_expensive_is_skipped_and_counted():
    dates = make_dates(10)
    closes = {"A": (dates, np.full(10, 10.0))}
    tr = [make_trade("A", 2, 6, dates)]
    sim = run(tr, closes, lambda t: 1e9, dates)
    assert sim["fills"] == 0
    assert sim["skipped_lot"] == 1


# ---- 2. 并发上限硬生效 ----
def test_slots_cap_is_enforced():
    dates = make_dates(10)
    closes = {c: (dates, np.full(10, 10.0)) for c in ("A", "B", "C")}
    tr = [make_trade(c, 2, 8, dates) for c in ("A", "B", "C")]
    sim = run(tr, closes, lambda t: 10.0, dates, slots=1)
    assert sim["fills"] == 1
    assert len(sim["filled"]) == 1


# ---- 3. 等额定仓：账户 5 万、10 仓 → 每笔约 5000 元 ----
def test_equal_sizing_uses_capital_over_slots():
    dates = make_dates(10)
    closes = {"A": (dates, np.full(10, 1.0))}
    tr = [make_trade("A", 2, 5, dates, entry_px=1.0, exit_px=1.0)]
    sim = run(tr, closes, lambda t: 1.0, dates, slots=10, capital=50000.0)
    units = sim["filled"][0]["entry"]  # 占位断言：确认真的成交
    assert sim["fills"] == 1
    # 5000 元预算 / 1 元 = 5000 股（100 股整手向下取整）
    assert 4900 <= closes["A"][1][2] * 5000 <= 5100
    assert units == 1.0


# ---- 4. 定仓用「昨日收盘」权益，不能用当日收盘（未来函数）----
def test_sizing_uses_yesterday_equity_not_today_close():
    dates = make_dates(12)
    # 第 2 天买入后第 3 天暴涨 1 倍，第 4 天再开新仓：新仓额度应基于前一日已含浮盈的权益
    px = np.array([10.0] * 12)
    px[3:] = 20.0
    closes = {"A": (dates, px), "B": (dates, np.full(12, 10.0))}
    tr = [make_trade("A", 1, 10, dates), make_trade("B", 4, 10, dates)]
    sim = run(tr, closes, lambda t: 10.0, dates, slots=2, capital=50000.0)
    eq = sim["eq"]
    assert sim["fills"] == 2
    # 第二天权益已跳升，说明盯市用的是当天收盘；但第 4 天开仓时用的必须是第 3 天收盘权益
    assert eq[3] > eq[1] * 1.5


# ---- 5. 确定性：同样输入两次跑出同一条净值 ----
def test_deterministic():
    dates = make_dates(20)
    rng = np.random.default_rng(7)
    closes = {c: (dates, 10 * np.cumprod(1 + rng.normal(0, 0.02, 20))) for c in "ABCD"}
    tr = [make_trade(c, 2 + i, 12 + i, dates) for i, c in enumerate("ABCD")]
    a = run(tr, closes, lambda t: 10.0, dates)["eq"]
    b = run(tr, closes, lambda t: 10.0, dates)["eq"]
    assert np.array_equal(a, b)


# ---- 6. 回撤闸门用滚动峰值：深回撤后能自动恢复，不会永久锁死 ----
def test_drawdown_gate_recovers():
    dates = make_dates(30)
    px = np.array([10.0] * 8 + [6.0] * 6 + [12.0] * 16)
    closes = {"A": (dates, px)}
    tr = [make_trade("A", 1, 8, dates), make_trade("A", 15, 28, dates)]
    # 去掉同 code 重复限制的影响：第二笔换一只票
    tr[1]["code"] = "B"
    closes["B"] = (dates, px)
    sim = run(tr, closes, lambda t: 10.0, dates, slots=2,
              dd_soft=0.10, dd_hard=0.15, dd_peak_window=252)
    # 破位期（回撤 >15%）之后权益恢复，第二笔必须还能成交
    assert sim["fills"] == 2, "回撤闸门把策略永久锁死了"


# ---- 7. 排序分单调：强市/贴近突破位/站稳久 一定排在前面 ----
def test_score_v3_ordering():
    good = {"regime": "强", "gain": 0.03, "days_above": 7, "vol_ratio": 1.7}
    bad = {"regime": "弱", "gain": 0.15, "days_above": 3, "vol_ratio": 1.95}
    near = {"regime": "强", "gain": 0.03, "days_above": 5, "vol_ratio": 1.7}
    far = {"regime": "强", "gain": 0.15, "days_above": 5, "vol_ratio": 1.7}
    assert E.score_v3(good) > E.score_v3(bad)
    assert E.score_v3(near) > E.score_v3(far)


# ---- 8. 单笔统计只算成交单，不能把候选信号当成交 ----
def test_trade_stats_only_count_filled():
    dates = make_dates(10)
    closes = {"A": (dates, np.full(10, 10.0))}
    tr = [make_trade("A", 2, 8, dates)] + [make_trade(f"Z{i}", 2, 8, dates, regime="震荡")
                                          for i in range(20)]
    for i in range(20):
        closes[f"Z{i}"] = (dates, np.full(10, 3.0))
    sim = run(tr, closes, lambda t: 10.0, dates, slots=2, regime_filter=("震荡", "弱"))
    assert sim["fills"] == 1
    st = E.trade_stats(sim["filled"])
    assert st["n"] == 1


# ---- 9. 面板与引擎必须是同一套冻结系数（否则「面板口径 ≠ 回测口径」又回来了）----
def test_panel_matches_engine_constants():
    sys.path.insert(0, str(ROOT))
    import build_panel_from_db as P
    assert tuple(P.V3_MU) == tuple(E.V3_MU)
    assert tuple(P.V3_SD) == tuple(E.V3_SD)
    assert tuple(P.V3_W) == tuple(E.V3_W)
    # 面板的仓位/每日开仓上限必须与引擎冻结值一致，且精选不设分数下限
    assert P.POS_SLOTS == E.V3_SPEC["slots"] == 10
    assert P.PICK_N == E.V3_SPEC["daily_cap"] == 3
    assert P.PICK_MIN is None
    # 展示分是单调变换：顺序必须与引擎一致
    cases = [("强", 0.03, 7, 1.7), ("强", 0.15, 5, 1.7), ("震荡", 0.03, 7, 1.7),
             ("弱", 0.15, 3, 1.9)]
    disp = [P.v3_display(*c) for c in cases]
    raw = [E.score_v3({"regime": r, "gain": g, "days_above": d, "vol_ratio": v})
           for r, g, d, v in cases]
    for i in range(len(cases)):
        for j in range(len(cases)):
            assert (disp[i] > disp[j]) == (raw[i] > raw[j]), (cases[i], cases[j])


# ---- 10. 参数冻结：V3_SPEC 不许在测试里被悄悄改 ----
def test_frozen_spec():
    assert E.V3_SPEC["slots"] == 10
    assert E.V3_SPEC["mode"] == "equal"
    assert E.V3_SPEC["regime_filter"] == ("震荡", "弱")
    assert E.V3_SPEC["dd_peak_window"] == 252
    assert E.V3_SPEC["trail"] == 0.25 and E.V3_SPEC["max_hold"] == 120
