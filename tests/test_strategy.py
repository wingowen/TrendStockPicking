# -*- coding: utf-8 -*-
"""P0 修复最小回归集。

运行： .venv-bt/bin/python -m pytest tests/test_strategy.py -v
不读 market.db、不联网：K 线用 fixture 合成，网络点用 monkeypatch 打桩。

全文符号表：
  c        后复权收盘价序列（list[float]，长度 n）
  o        open，开盘价序列（list[float]）
  v        volume，成交量序列（list[float]）
  k        当前 bar 下标（int，0-based）
  t        候选突破日下标（int，0-based）
  L        LOOKBACK = 20，突破位取前 L 个交易日最高收盘（交易日）
  W        BREAK_WIN = 8，突破后需在 W 个交易日内完成站稳（交易日）
  lv       level = max(c[t-L:t])，突破位（价格）
  prev_lv  max(c[t-L-1:t-1])，前一日的突破位（价格）
  vr       volume ratio = v[t] / mean(v[t-5:t])，突破日量比（无量纲）
  gap      次日开盘涨幅 = o[k+1]/c[k] - 1（比例，0.03 = 3%）
  GAP_MAX  +0.095，高开超过 9.5% 放弃；GAP_MIN 0.03，低开超过 3% 放弃
  TTL      CACHE_MAX_AGE_DAYS = 1，资金流缓存有效期（自然日）
"""
import json
import re
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))

import build_panel_from_db as panel  # noqa: E402
import board_flow  # noqa: E402
import run_backtest as bt  # noqa: E402

L = panel.LOOKBACK          # 20，前 L 日最高收盘的窗口
HOLD = panel.HOLD_MIN       # 3，最少站稳天数
W = panel.BREAK_WIN         # 8，站稳观察窗口
GAPMAX = panel.GAP_MAX      # 0.095
GAPMIN = panel.GAP_MIN      # 0.03


def _trade_dates(n, start="2025-01-02"):
    """生成 n 个递增工作日作为合成 K 线日期轴（维度：交易日）。"""
    d, out = date.fromisoformat(start), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


@pytest.fixture
def tmp_cache_dir():
    """一次性临时目录；不用 pytest 的 tmp_path，避免沙箱对复用目录误报 EEXIST。"""
    with tempfile.TemporaryDirectory(prefix="wencai_test_") as d:
        yield Path(d)


@pytest.fixture
def synth():
    """合成单边上涨 K 线工厂。

    base=100，首破在 first_breakout=26，之后每根 +step；
    令突破日成交量 = vr * 前 5 日均量(1000)，使量比恰为 vr；
    open 默认 = 前一日收盘（gap=0），gap 用例单独覆盖入场日 open。
    """

    def _make(n=90, first_breakout=26, step=10.0, base=100.0, vr=1.75):
        ds = _trade_dates(n)
        c = [base] * first_breakout + [
            base + step * (k - first_breakout + 1)
            for k in range(first_breakout, n)
        ]
        v = [1000.0] * n
        v[first_breakout] = vr * 1000.0
        o = [c[0]] + [c[k - 1] for k in range(1, n)]
        bars = {
            "dates": ds, "open": o, "close": c, "high": c, "low": c,
            "vol": v, "amount": [1e8] * n,
        }
        idx = {"上证指数": {"dates": ds, "close": [3000.0] * n,
                            "amount": [1e9] * n}}
        return bars, idx

    return _make


def _entry_on_signal(bars, idx, gap):
    """跑 simulate_positions，返回入场日是否成交。

    信号日 = 26（首破）+ 3（站稳）- 1 = 28，入场日 = 29（次日开盘）。
    把入场日 open 设为信号日收盘 *(1+gap)。
    """
    b = {**bars, "open": list(bars["open"])}
    signal_day, entry_day = 28, 29
    b["open"][entry_day] = b["close"][signal_day] * (1 + gap)
    stocks = {"T00001": {"name": "合成股", "industry": "合成行业", "market": "主板"}}
    pos, _closed = panel.simulate_positions(
        stocks, {"T00001": b}, {}, idx, [b["dates"][entry_day]])
    return len(pos[b["dates"][entry_day]]) > 0


# ---------- 正常流 ----------

def test_BRK_001_first_breakout_only_once(synth):
    """TC-BRK-001：单边上涨中只有首日算首次突破（原恒真 bug 的回归）。"""
    c = synth()[0]["close"]
    assert panel.prev_below(c, 26, L) is True
    assert panel.prev_below(c, 27, L) is False
    assert panel.prev_below(c, 28, L) is False


def test_BRK_002_prev_below_short_history(synth):
    """TC-BRK-002：t==L 时历史不足一个 prev 窗口，按首次突破处理。"""
    assert panel.prev_below(synth()[0]["close"], L, L) is True


def test_BRK_003_panel_backtest_signals_aligned(synth):
    """TC-BRK-003：面板 evaluate 与回测 find_signals 的首破日集合一致。"""
    bars, _ = synth(vr=1.75)
    ps = set()
    for k in range(L + W + 5, len(bars["close"])):
        got = panel.evaluate(bars, k, {})
        if got and not got[1]:
            ps.add(got[0]["breakout_date"])
    rb = {
        "c": np.asarray(bars["close"]),
        "v": np.asarray(bars["vol"]),
        "dates": bars["dates"],
    }
    bs = {s["breakout_date"].replace("-", "") for s in bt.find_signals(rb, bt.BASE)}
    assert ps == bs == {bars["dates"][26].replace("-", "")}


@pytest.mark.parametrize("vr, expected", [(1.5, True), (1.75, True), (2.0, True)])
def test_VOL_boundaries_accept(synth, vr, expected):
    """TC-VOL-001~003：量比闭区间 [1.5, 2.0] 全部放行。"""
    rec, _ = panel.evaluate(synth(vr=vr)[0], L + W + 5, {})
    assert rec["vol_ok"] is expected


def test_GAP_within_range(synth):
    """TC-GAP-001~002：+9.4% 与 -2.9% 均可入场。"""
    bars, idx = synth()
    assert _entry_on_signal(bars, idx, +0.094) is True
    assert _entry_on_signal(bars, idx, -0.029) is True


def test_TTL_001_fresh_cache_hit(tmp_cache_dir, monkeypatch):
    """TC-TTL-001：新鲜缓存命中时不得联网。"""
    cache = tmp_cache_dir / "fflow_cache.json"
    payload = {
        "want": ["合成行业"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flow": {"合成行业": {"2025-01-02": 0.01}},
    }
    cache.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(board_flow, "CACHE", cache)
    monkeypatch.setattr(board_flow, "board_map", lambda: pytest.fail("命中缓存却联网"))
    assert board_flow.load_real_flow(["合成行业"]) == payload["flow"]


def test_TTL_002_naive_ts_as_utc():
    """TC-TTL-002：无时区戳按 UTC 解释。"""
    naive_utc = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    assert board_flow._cache_fresh(
        {"generated_at": naive_utc},
        board_flow.CACHE_MAX_AGE_DAYS) is True


def test_POS_001_cost_disclaimer_present():
    """TC-POS-001：面板/规则/生成脚本须明示未扣成本、非资金口径。"""
    text = "\n".join((ROOT / n).read_text(encoding="utf-8")
                     for n in ("breakout-panel.html", "rules.html",
                               "build_panel_from_db.py"))
    assert re.search(r"不含成本|未扣.*成本|不扣.*成本", text)
    assert re.search(r"仓位|资金分配", text)


@pytest.mark.parametrize("name", [
    "server_daily.sh", "ci_build.sh", "run_daily.sh", "panel_server.py",
])
def test_TZ_001_scripts_export_tz(name):
    """TC-TZ-001：所有会判定收盘时间的入口都锁定北京时间。"""
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "Asia/Shanghai" in text, name


def test_CACHE_PATH_001_daily_script_path():
    """TC-CACHE-001：每日脚本清理的缓存路径必须与 board_flow.CACHE 一致。"""
    text = (ROOT / "server_daily.sh").read_text(encoding="utf-8")
    assert "backtest/fflow_cache.json" in text


# ---------- 异常流 ----------

def test_BRK_E01_no_repeat(synth):
    """TC-BRK-E01：同一突破在窗口内只产出一个信号日。"""
    bars, _ = synth(vr=1.75)
    seen = set()
    for k in range(L + W + 5, L + W + 18):
        got = panel.evaluate(bars, k, {})
        if got and not got[1]:
            seen.add(got[0]["breakout_date"])
    assert len(seen) <= 1, seen


def test_BRK_E02_fall_below_then_rebreak(synth):
    """TC-BRK-E02：跌破后重新突破仍算首次。"""
    c = [100.0] * 26 + [110.0, 90.0, 95.0, 120.0]
    assert panel.prev_below(c, 29, L) is True


@pytest.mark.parametrize("vr", [1.49, 2.01])
def test_VOL_E01_E02_out_of_range(synth, vr):
    """TC-VOL-E01/E02：量比越界即否决。"""
    rec, _ = panel.evaluate(synth(vr=vr)[0], L + W + 5, {})
    assert rec["vol_ok"] is False


def test_VOL_E03_missing_volume(synth):
    """TC-VOL-E03：前 5 日均量为 0 时量比缺失。"""
    b = synth()[0]
    b["vol"][21:26] = [0.0] * 5
    b["vol"][26] = 1000.0
    rec, _ = panel.evaluate(b, L + W + 5, {})
    assert rec["vol_ok"] is False


def test_GAP_E01_high_open_reject(synth):
    """TC-GAP-E01：高开 +9.6% 放弃。"""
    bars, idx = synth()
    assert _entry_on_signal(bars, idx, +0.096) is False


def test_GAP_E02_low_open_reject(synth):
    """TC-GAP-E02：低开 -3.1% 放弃。"""
    bars, idx = synth()
    assert _entry_on_signal(bars, idx, -0.031) is False


def test_GAP_E03_exact_minus3_boundary(synth):
    """TC-GAP-E03：按规则「超过 3% 才放弃」，恰好 -3.0% 应放行。

    浮点误差会让 130*0.97/130-1 = -0.030000000000000027，
    已在两处 gap 判断加 1e-9 容差（团长裁定 2026-11-21）。
    """
    bars, idx = synth()
    assert _entry_on_signal(bars, idx, -0.030) is True


def test_TTL_E01_expired_refetch(tmp_cache_dir, monkeypatch):
    """TC-TTL-E01：缓存过期 2 天则重抓并刷新 generated_at。"""
    cache = tmp_cache_dir / "fflow_cache.json"
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    cache.write_text(json.dumps({
        "want": ["合成行业"], "generated_at": old, "flow": {},
    }), encoding="utf-8")
    monkeypatch.setattr(board_flow, "CACHE", cache)
    calls = {"n": 0}

    def fake_map():
        calls["n"] += 1
        return {"合成行业": "BK0001"}

    monkeypatch.setattr(board_flow, "board_map", fake_map)
    monkeypatch.setattr(board_flow, "board_flow",
                        lambda bk: {"2025-01-03": 0.02})
    out = board_flow.load_real_flow(["合成行业"])
    assert calls["n"] == 1
    assert out == {"合成行业": {"2025-01-03": 0.02}}
    assert "generated_at" in json.loads(cache.read_text(encoding="utf-8"))


@pytest.mark.parametrize("cache", [
    {},
    {"generated_at": "not-a-time"},
    {"generated_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
])
def test_TTL_E02_E03_invalid_or_missing(cache):
    """TC-TTL-E02/E03：无戳、非法戳、过期戳均视为失效。"""
    assert board_flow._cache_fresh(cache, board_flow.CACHE_MAX_AGE_DAYS) is False
