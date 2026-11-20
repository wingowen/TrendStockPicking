#!/usr/bin/env python3
"""在回测里加上「大盘闸门 / 板块强度 / 回踩确认」三层，量化它们到底加不加分。

复用 run_backtest.py 的数据加载、信号识别、逐笔模拟与前瞻收益，不改原口径。

用法：
    ../.venv-bt/bin/python analyze_factors.py          # 全历史
    ../.venv-bt/bin/python analyze_factors.py --from 2022-01-01
输出：factors.html / factors.json / trades_factors.csv
"""
import argparse
import json
import sqlite3
import sys
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_backtest as R  # noqa: E402
from sector_trend import build_industry_indices, snapshot  # noqa: E402

OUT_HTML = ROOT / "factors.html"
OUT_JSON = ROOT / "factors.json"
OUT_CSV = ROOT / "trades_factors.csv"

EVAL_START = R.EVAL_START
RETEST_ZONE = 1.02      # 回踩区间：突破位上方 2% 以内
RETEST_WIN = 15         # 信号后多少个交易日内等回踩
RETEST_CONFIRM_WIN = 20 # 回踩后多少个交易日内等确认


# ---------------- 大盘状态 ----------------

def regime_series(idx):
    ds = idx["上证指数"]["dates"]
    c = idx["上证指数"]["close"]
    a = idx["上证指数"]["amount"]
    n = len(c)
    out = {}
    cs = np.cumsum(np.concatenate(([0.0], c)))
    ca = np.cumsum(np.concatenate(([0.0], a)))
    for j in range(n):
        lo20 = max(0, j - 19)
        ma20 = (cs[j + 1] - cs[lo20]) / (j + 1 - lo20)
        lo60 = max(0, j - 59)
        ma60 = (cs[j + 1] - cs[lo60]) / (j + 1 - lo60)
        mom20 = c[j] / c[j - 20] - 1 if j >= 20 else 0.0
        mom60 = c[j] / c[j - 60] - 1 if j >= 60 else 0.0
        above20, above60 = c[j] > ma20, c[j] > ma60
        if above20 and above60 and mom20 > 0:
            label = "强"
        elif (not above20) and (not above60):
            label = "弱"
        else:
            label = "震荡"
        lo20a = max(0, j - 20)
        base = (ca[j] - ca[lo20a]) / max(1, j - lo20a)
        out[ds[j]] = {"label": label, "score": 50 + (20 if above20 else -20)
                      + (15 if above60 else -15) + (15 if mom20 > 0 else -15),
                      "mom20": round(mom20 * 100, 2),
                      "vol_ratio": round(a[j] / base, 2) if base else None}
    return out


# ---------------- 板块强度 ----------------

# ---------------- 回踩 ----------------

def find_retest(bars, i, k, level, horizon):
    """在 (i, k+horizon] 里找回踩：最低价触及突破位上方 2%、收盘不破位。
    返回 dict(confirm_idx, pull_idx, shrink) 或 None（含回踩未确认，confirm_idx=None）"""
    c, l, h, v = bars["c"], bars["l"], bars["h"], bars["v"]
    n = len(c)
    end = min(k + horizon, n - 1)
    zone = level * RETEST_ZONE
    pull = next((t for t in range(i + 1, end + 1)
                 if l[t] <= zone and c[t] > level), None)
    if pull is None:
        return None
    shrink = float(v[pull] / v[i]) if v[i] else None
    hi = float(h[i:pull + 1].max())
    conf = next((s for s in range(pull + 1, min(pull + 1 + RETEST_CONFIRM_WIN, n))
                 if c[s] > hi), None)
    return {"pull_idx": pull, "confirm_idx": conf, "shrink": shrink,
            "pull_date": bars["dates"][pull],
            "confirm_date": bars["dates"][conf] if conf is not None else None}


def retest_entry(bars, sig, cfg, level):
    """回踩确认后入场（确认日次日开盘），沿用同一套出场规则。"""
    r = find_retest(bars, sig["breakout_idx"], sig["signal_idx"], level, RETEST_WIN)
    if not r or r["confirm_idx"] is None:
        return None, r
    s = r["confirm_idx"]
    if s + 1 >= len(bars["c"]):
        return None, r
    fake = dict(sig, signal_idx=s)
    tr = R.simulate_trade(bars, fake, cfg)
    return tr, r


# ---------------- 统计 ----------------

def group_stats(trades, keyfn):
    buckets = {}
    for t in trades:
        k = keyfn(t)
        if k is None:
            continue
        buckets.setdefault(k, []).append(t)
    rows = []
    for k, lst in buckets.items():
        if len(lst) < 30:
            continue
        rets = np.array([x["ret"] for x in lst])
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        f20 = [x.get("f20") for x in lst if x.get("f20") is not None]
        f60 = [x.get("f60") for x in lst if x.get("f60") is not None]
        rows.append({
            "key": k, "n": len(lst),
            "avg": float(rets.mean()), "med": float(np.median(rets)),
            "win": float(len(wins) / len(rets)),
            "pl": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else None,
            "hold": float(np.mean([x["hold"] for x in lst])),
            "f20": float(np.mean(f20)) if f20 else None,
            "f60": float(np.mean(f60)) if f60 else None,
        })
    return rows


def fmt_pct(v, d=2):
    return "—" if v is None else f"{v*100:+.{d}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=EVAL_START)
    a = ap.parse_args()
    start = a.start

    stocks, bars, idx = R.load_data()
    con = sqlite3.connect(R.DB)
    industry = {r[0]: (r[1] or "") for r in
                con.execute("select code, industry from stocks")}
    con.close()
    con = sqlite3.connect(R.DB)
    idx_amount = [r[0] for r in con.execute(
        "select amount from bars where code='IDX_上证指数' order by date")]
    con.close()
    idx["上证指数"] = dict(idx["上证指数"])
    idx["上证指数"]["amount"] = np.array(
        [idx_amount[i] if i < len(idx_amount) else 0.0
         for i in range(len(idx["上证指数"]["close"]))], dtype=float)
    regime = regime_series(idx)

    print("[signal] 扫描全市场", flush=True)
    t0 = time.time()
    sigs = []
    for code, b in bars.items():
        if len(b["dates"]) < R.WARMUP_BARS:
            continue
        for s in R.find_signals(b, R.BASE):
            k = s["signal_idx"]
            if b["dates"][k] < start:
                continue
            tr = R.simulate_trade(b, s, R.BASE)
            if not tr:
                continue
            tr["code"] = code
            tr["name"] = stocks.get(code, {}).get("name", code)
            tr.update({kk: vv for kk, vv in R.forward_returns(b, s, idx["沪深300"], R.BASE).items()})
            sigs.append((b, s, tr))
    print(f"[signal] {len(sigs)} 笔，{time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    cal = idx["上证指数"]["dates"]
    ind_series = build_industry_indices(bars, industry, cal)
    print(f"[sector] {len(ind_series)} 个板块指数，{time.time()-t1:.1f}s", flush=True)
    need_dates = sorted({s["signal_date"] for _, s, _ in sigs})
    sector = {d: snapshot(ind_series, d) for d in need_dates}
    print(f"[sector] 快照 {len(need_dates)} 个信号日", flush=True)

    trades = []
    retest_entries = []
    for b, s, tr in sigs:
        d = s["signal_date"]
        reg = regime.get(d) or {}
        sec = (sector.get(d) or {}).get(industry.get(tr["code"]) or "")
        sec_pct = sec["score"] if sec else None
        rt = find_retest(b, s["breakout_idx"], s["signal_idx"], s["level"], 0)
        tr.update({"regime": reg.get("label"), "regime_score": reg.get("score"),
                   "sector_pct": sec_pct,
                   "sector_label": sec["label"] if sec else None,
                   "sector_flow": sec["flow5"] if sec else None,
                   "sector_consec": sec["consec"] if sec else None,
                   "retest": "pull" if rt else "none"})
        trades.append(tr)

        # 回踩改造入场（信号后 15 个交易日内等回踩确认）
        if rt is None:
            e, r2 = retest_entry(b, s, R.BASE, s["level"])
            if e:
                e.update({"code": tr["code"], "name": tr["name"],
                          "base_entry_date": tr["entry_date"],
                          "regime": reg.get("label"), "sector_pct": sec_pct,
                          "sector_label": sec["label"] if sec else None,
                          "sector_flow": sec["flow5"] if sec else None,
                          "sector_consec": sec["consec"] if sec else None,
                          "retest": "confirmed"})
                e.update({kk: vv for kk, vv in R.forward_returns(b, dict(s, signal_idx=r2["confirm_idx"]),
                                                                 idx["沪深300"], R.BASE).items()})
                retest_entries.append(e)

    print(f"[trade] 原始 {len(trades)} 笔，回踩确认入场 {len(retest_entries)} 笔", flush=True)

    base = trades
    by_regime = group_stats(base, lambda t: t["regime"])
    by_sector = group_stats(base, lambda t: None if t["sector_label"] is None
                            else "板块" + t["sector_label"])
    by_flow = group_stats(base, lambda t: None if t["sector_consec"] is None else
                          ("连续净流入 ≥3 日" if t["sector_consec"] >= 3 else
                           ("连续净流入 1–2 日" if t["sector_consec"] >= 1 else "当日无净流入")))
    by_score = group_stats(base, lambda t: None if t["sector_pct"] is None
                           else f"板块评分 {t['sector_pct']}")
    by_flowratio = group_stats(base, lambda t: None if t["sector_flow"] is None else
                               ("资金净流入 >10%" if t["sector_flow"] > 0.1 else
                                ("资金净流入 0–10%" if t["sector_flow"] > 0 else
                                 ("资金净流出 0–10%" if t["sector_flow"] > -0.1 else "资金净流出 >10%"))))
    by_regime_x_sector = group_stats(base, lambda t: None if t["sector_label"] is None
                                     else f"{t['regime']} × 板块{t['sector_label']}")

    # 过滤组合
    variants = {
        "原始（全部信号）": base,
        "大盘非弱": [t for t in base if t["regime"] != "弱"],
        "板块强（站均线+动量为正+资金流入）": [t for t in base if t["sector_label"] == "强"],
        "板块非弱": [t for t in base if t["sector_label"] in ("强", "中性")],
        "回踩确认后入场": retest_entries,
        "大盘非弱 + 板块非弱": [t for t in base if t["regime"] != "弱"
                              and t["sector_label"] in ("强", "中性")],
        "三层全开（大盘非弱 + 板块非弱 + 回踩确认）": [
            t for t in retest_entries if t["regime"] != "弱"
            and t["sector_label"] in ("强", "中性")],
    }
    var_rows = []
    for name, lst in variants.items():
        st = group_stats(lst, lambda t: "all")
        if st:
            var_rows.append(dict(st[0], key=name))

    # 回踩入场 vs 同期信号日入场（同代码同信号）
    pairs = []
    base_by_key = {(t["code"], t["breakout_date"]): t for t in base}
    for t in retest_entries:
        b0 = base_by_key.get((t["code"], t["breakout_date"]))
        if b0:
            pairs.append((b0, t))
    pair_stats = []
    if pairs:
        pair_stats.append(("信号日次日买入", [p[0] for p in pairs]))
        pair_stats.append(("回踩确认后买入", [p[1] for p in pairs]))

    data = {"start": start, "signals": len(base), "retest_entries": len(retest_entries),
            "by_regime": by_regime, "by_sector": by_sector, "by_flow": by_flow,
            "by_score": by_score, "by_flowratio": by_flowratio,
            "regime_x_sector": by_regime_x_sector,
            "variants": var_rows,
            "pairs": [dict(group_stats(v, lambda t: "all")[0], key=n) for n, v in pair_stats]}

    # CSV
    import csv
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["代码", "名称", "突破日", "信号日", "买入日", "收益", "持仓天数",
                    "大盘", "板块状态", "板块资金比", "板块连续流入", "回踩", "退出原因", "f20"])
        for t in base:
            w.writerow([t["code"], t["name"], t["breakout_date"], t["signal_date"],
                        t["entry_date"], round(t["ret"], 4), t["hold"], t["regime"],
                        t["sector_label"], None if t["sector_flow"] is None else round(t["sector_flow"], 3),
                        t["sector_consec"], t["retest"], t["reason"],
                        None if t.get("f20") is None else round(t["f20"], 4)])

    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 报告 ----------------
    def table(rows, labels):
        head = ("<tr><th>分组</th><th class='num'>笔数</th><th class='num'>平均净收益</th>"
                "<th class='num'>中位数</th><th class='num'>胜率</th><th class='num'>盈亏比</th>"
                "<th class='num'>20日前瞻</th><th class='num'>60日前瞻</th></tr>")
        body = "".join(
            f"<tr><td>{labels(r)}</td><td class='num'>{r['n']}</td>"
            f"<td class='num {'pos' if r['avg']>0 else 'neg'}'>{fmt_pct(r['avg'])}</td>"
            f"<td class='num'>{fmt_pct(r['med'])}</td><td class='num'>{r['win']*100:.1f}%</td>"
            f"<td class='num'>{r['pl']:.2f}</td><td class='num'>{fmt_pct(r['f20'])}</td>"
            f"<td class='num'>{fmt_pct(r['f60'])}</td></tr>" for r in rows)
        return f"<table>{head}{body}</table>"

    base_stat = group_stats(base, lambda t: "all")[0]
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>三层因子检验</title><style>
body{{font:13px/1.7 -apple-system,"PingFang SC",sans-serif;margin:0;background:#f6f7f9;color:#1f2328}}
.wrap{{max-width:1080px;margin:0 auto;padding:26px 20px 60px}}
h1{{font-size:21px;margin:0 0 4px}} h2{{font-size:15px;margin:26px 0 10px}}
.sub{{color:#67707a;font-size:12px;margin-bottom:18px}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;
 box-shadow:0 1px 2px rgba(0,0,0,.06)}}
th,td{{padding:7px 10px;border-bottom:1px solid #eef0f3;text-align:left;font-size:12px}}
th{{background:#fafbfc;color:#5b636b;font-weight:600}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.pos{{color:#c0392b}} .neg{{color:#1a7f52}}
.note{{background:#fff;border-left:3px solid #4b45a8;padding:10px 14px;border-radius:8px;
 font-size:12.5px;color:#3a4046;margin:14px 0}}
</style></head><body><div class="wrap">
<h1>三层判断到底加不加分</h1>
<div class="sub">口径：{start} 起，全部原始突破信号 {len(base)} 笔 · 同一套出场（跌破突破位止损 / 12% 移动止盈 / 最长 120 日）· 单边成本 0.2%</div>

<div class="note"><b>读法</b>：平均净收益是「信号次日开盘买入、按规则卖出、扣 0.4% 往返成本」后的单笔期望；
20/60 日前瞻是指同一时点买入后持有 20/60 个交易日的平均收益，用来剔除出场规则干扰。
基准：全部信号 {fmt_pct(base_stat['avg'])}、胜率 {base_stat['win']*100:.1f}%、20 日前瞻 {fmt_pct(base_stat['f20'])}。</div>

<h2>一、按大盘状态分层</h2>
{table(by_regime, lambda r: '大盘' + r['key'])}

<h2>二、按板块趋势分层（板块等权指数：均线 + 20 日动量 + 5 日资金净流入）</h2>
{table(by_sector, lambda r: r['key'])}

<h2>二·补、按板块连续资金净流入天数分层</h2>
{table(by_flow, lambda r: r['key'])}

<h2>二·补 2、按板块资金净流入强度分层</h2>
{table(by_flowratio, lambda r: r['key'])}

<h2>二·补 3、按板块趋势评分分层</h2>
{table(by_score, lambda r: r['key'])}

<h2>三、大盘 × 板块 交叉</h2>
{table(by_regime_x_sector, lambda r: r['key'])}

<h2>四、过滤组合对比</h2>
{table(var_rows, lambda r: r['key'])}

<h2>五、回踩确认入场 vs 信号日入场（同一批信号）</h2>
{table([dict(group_stats(v, lambda t: "all")[0], key=n) for n, v in pair_stats], lambda r: r['key'])}
<div class="sub" style="margin-top:8px">“回踩确认后买入”= 信号后 15 个交易日内出现回踩（最低价触及突破位上方 2%、收盘不破位），
之后收盘重新站上回踩前高点，次日开盘买入。</div>

</div></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print("生成", OUT_HTML.name, OUT_JSON.name, OUT_CSV.name)


if __name__ == "__main__":
    main()
