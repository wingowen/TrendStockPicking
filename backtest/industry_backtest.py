#!/usr/bin/env python3
"""行业互斥约束回测：同一行业同时只持一只（先入为主，平仓释放，同日按代码顺序）。

对照四组口径（其余参数与 run_backtest.BASE 完全一致）：
  A 等权全信号（原基准）
  B 等权 + 行业互斥（= 面板「当前持仓」的口径）
  C 6 仓风险预算实盘（原基准）
  D 6 仓风险预算实盘 + 行业互斥

输出 industry.html / industry.json，并在控制台打印对照表。
"""
import json
import sqlite3
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from run_backtest import (BASE, DB, EVAL_START, WARMUP_BARS, load_data, find_signals,
                          simulate_trade, build_equity, build_equal_weight,
                          metrics_from_equity, trade_stats, svg_curve)

ROOT = Path(__file__).resolve().parent


def ind_map():
    con = sqlite3.connect(DB)
    m = {r[0]: (r[1] or ("__" + r[0])) for r in con.execute("select code, industry from stocks")}
    con.close()
    return m


def filter_industry(trades, ind_of):
    """按入场时间顺序执行行业互斥；exit_date 当日开盘卖出即释放行业。"""
    accepted, skipped = [], []
    active, occupied = {}, {}
    for tr in trades:                       # 已按 (entry_date, code) 排序
        d = tr["entry_date"]
        for code in list(active):
            if active[code] <= d:
                ind = ind_of(code)
                if occupied.get(ind) == code:
                    del occupied[ind]
                del active[code]
        code = tr["code"]
        ind = ind_of(code)
        if code in active or ind in occupied:
            skipped.append(tr)
            continue
        accepted.append(tr)
        active[code] = tr["exit_date"]
        occupied[ind] = code
    return accepted, skipped


def build_equity_ind(trades, idx_dates, cfg, ind_of):
    """6 仓风险预算实盘 + 行业互斥（build_equity 的行业互斥版）。"""
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
            if p["exit_idx"] == di:
                cash += p["shares"] * p["exit_price"]
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
            mv += p["shares"] * b["c"][i]
        held = {p["code"] for p in positions}
        held_ind = {ind_of(p["code"]) for p in positions}
        for tr in sorted(pend.get(di, []), key=lambda x: x["code"]):
            if len(positions) >= cfg["slots"] or cash <= 0:
                break
            if tr["code"] in held or ind_of(tr["code"]) in held_ind:
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
            held_ind.add(ind_of(tr["code"]))
            positions.append({
                "code": tr["code"], "shares": shares,
                "exit_idx": date_pos.get(tr["exit_date"], n - 1),
                "exit_price": tr["exit"] * (1 - cfg["cost"] / 2), "bars": tr["_bars"],
                "i": tr["_entry_i"], "date": d,
            })
            mv += shares * buy_px
            tr["_used_ind"] = True
        eq[di] = cash + mv
    return eq


def fragility(trades, topn=50):
    if len(trades) <= topn:
        return None
    rest = sorted(trades, key=lambda t: -t["ret"])[topn:]
    return float(np.mean([t["ret"] for t in rest]))


def main():
    cfg = dict(BASE)
    ind_of_map = ind_map()
    ind_of = lambda c: ind_of_map.get(c, "__" + c)
    stocks, data, idx = load_data()
    bench = idx["沪深300"]
    dates_all = bench["dates"]

    t0 = time.time()
    all_trades = []
    for code, b in data.items():
        if len(b["dates"]) < WARMUP_BARS:
            continue
        for sig in find_signals(b, cfg):
            k = sig["signal_idx"]
            if b["dates"][k] < EVAL_START:
                continue
            tr = simulate_trade(b, sig, cfg)
            if not tr:
                continue
            tr["code"] = code
            tr["_bars"] = b
            tr["_entry_i"] = k + 1
            all_trades.append(tr)
    all_trades.sort(key=lambda t: (t["entry_date"], t["code"]))
    print(f"[signal] {len(all_trades)} 笔，耗时 {time.time()-t0:.1f}s", flush=True)

    acc, skipped = filter_industry(all_trades, ind_of)
    print(f"[ind] 行业互斥后成交 {len(acc)} 笔，跳过 {len(skipped)} 笔", flush=True)

    start_i = next(i for i, d in enumerate(dates_all) if d >= all_trades[0]["entry_date"])
    win_dates = dates_all[start_i:]
    bench_curve = bench["close"][start_i:]
    bench_curve = bench_curve / bench_curve[0]
    bench_m = metrics_from_equity(bench_curve, win_dates)

    results = {}
    curves = {}
    for name, ts in (("A", all_trades), ("B", acc)):
        eq, conc = build_equal_weight(ts, dates_all, cfg)
        m = metrics_from_equity(eq[start_i:], win_dates)
        st = trade_stats(ts)
        results[name] = {"stats": st, "eq": m, "conc": conc,
                         "frag50": fragility(ts)}
        curves[name] = eq[start_i:]
    for name, ts, fn in (("C", all_trades, lambda t: build_equity(t, dates_all, bench["close"], cfg)),
                         ("D", all_trades, lambda t: build_equity_ind(t, dates_all, cfg, ind_of))):
        eq = fn(ts)
        m = metrics_from_equity(eq[start_i:], win_dates)
        used = sum(1 for t in ts if t.get("_used" if name == "C" else "_used_ind"))
        results[name] = {"eq": m, "used": used}
        curves[name] = eq[start_i:]

    # 年度（等权 A vs B）
    eqA, _ = build_equal_weight(all_trades, dates_all, cfg)
    eqB, _ = build_equal_weight(acc, dates_all, cfg)
    yearly = []
    for y in sorted({d[:4] for d in win_dates}):
        i0 = next((i for i, d in enumerate(dates_all) if d.startswith(y)), None)
        i1 = next((i for i, d in enumerate(dates_all) if d.startswith(str(int(y) + 1))), len(dates_all))
        if i0 is None or i0 < start_i:
            continue
        yearly.append((y, {
            "A": eqA[i1 - 1] / eqA[i0] - 1, "B": eqB[i1 - 1] / eqB[i0] - 1,
            "bench": bench["close"][i1 - 1] / bench["close"][i0] - 1,
            "nA": sum(1 for t in all_trades if t["entry_date"].startswith(y)),
            "nB": sum(1 for t in acc if t["entry_date"].startswith(y))}))

    # 被跳过信号的质量
    sk_stats = trade_stats(skipped) if skipped else {}

    # 控制台
    def pct(v): return f"{v*100:+.2f}%"
    print("\n========== 行业互斥对照（2020 起，其余参数同 BASE） ==========")
    for name, label in (("A", "等权全信号（原基准）"), ("B", "等权+行业互斥（面板口径）"),
                        ("C", "6仓实盘（原基准）"), ("D", "6仓实盘+行业互斥")):
        r = results[name]
        m = r["eq"]
        line = f"{label:26s} 累计 {pct(m['total']):>9s}  年化 {pct(m['cagr']):>8s}  回撤 {pct(m['mdd']):>8s}  Sharpe {m['sharpe']:.2f}"
        if "stats" in r:
            st = r["stats"]
            line += f"  | {st['n']} 笔  胜率 {st['win']*100:.1f}%  单笔 {pct(st['avg'])}  并发 {r['conc']:.0f}"
        else:
            line += f"  | 成交 {r['used']} 笔"
        print(line)
    print(f"沪深300 同期 {pct(bench_m['total'])}  回撤 {pct(bench_m['mdd'])}")
    print(f"脆弱性：剔除 top50 后单笔  A {pct(results['A']['frag50'])}  B {pct(results['B']['frag50'])}")
    if sk_stats:
        print(f"被行业互斥跳过的 {sk_stats['n']} 笔：单笔 {pct(sk_stats['avg'])}  胜率 {sk_stats['win']*100:.1f}%")

    # HTML 报告
    def card(k, v, s, cls=""):
        return f'<div class="card"><div class="k">{k}</div><div class="v {cls}">{v}</div><div class="s">{s}</div></div>'
    def cls(v): return "pos" if v > 0 else "neg"
    rows_y = "".join(
        f"<tr><td>{y}</td><td class='num {cls(v['A'])}'>{pct(v['A'])}</td><td class='num {cls(v['B'])}'>{pct(v['B'])}</td>"
        f"<td class='num'>{pct(v['bench'])}</td><td class='num'>{v['nA']}</td><td class='num'>{v['nB']}</td></tr>"
        for y, v in yearly)
    rows_cmp = ""
    for name, label in (("A", "等权全信号（原基准）"), ("B", "等权 + 行业互斥"),
                        ("C", "6 仓实盘（原基准）"), ("D", "6 仓实盘 + 行业互斥")):
        r = results[name]; m = r["eq"]
        st = r.get("stats")
        n = st["n"] if st else r["used"]
        win = f"{st['win']*100:.1f}%" if st else "—"
        avg = pct(st["avg"]) if st else "—"
        conc = f"{r['conc']:.0f}" if st else "≤6"
        rows_cmp += (f"<tr><td>{label}</td><td class='num'>{n}</td>"
                     f"<td class='num'>{win}</td><td class='num'>{avg}</td>"
                     f"<td class='num {cls(m['total'])}'>{pct(m['total'])}</td>"
                     f"<td class='num'>{pct(m['cagr'])}</td>"
                     f"<td class='num {cls(m['mdd'])}'>{pct(m['mdd'])}</td>"
                     f"<td class='num'>{m['sharpe']:.2f}</td>"
                     f"<td class='num'>{conc}</td></tr>")
    curveB = svg_curve(curves["B"].tolist(), bench_curve.tolist(), win_dates, "等权+行业互斥（红）vs 沪深300")
    curveA = svg_curve(curves["A"].tolist(), bench_curve.tolist(), win_dates, "等权全信号（红）vs 沪深300")
    curveD = svg_curve(curves["D"].tolist(), bench_curve.tolist(), win_dates, "6仓实盘+行业互斥（红）vs 沪深300")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>行业互斥约束回测</title><style>
:root{{--ink:#2C2C2A;--ink2:#5F5E5A;--ink3:#888780;--line:#e3e5e8;--up:#C43A2E;--down:#1D7A4C}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#f5f6f8;color:var(--ink);font:14px/1.65 -apple-system,"PingFang SC",sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:30px 24px 70px}}
h1{{font-size:24px;font-weight:650}} h2{{font-size:15px;font-weight:600;margin:26px 0 10px}}
.sub{{color:var(--ink3);font-size:12px;margin-top:6px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.card .k{{font-size:11px;color:var(--ink3)}} .card .v{{font-size:21px;font-weight:650;margin-top:2px}}
.card .s{{font-size:11px;color:var(--ink2)}} .pos{{color:var(--up)}} .neg{{color:var(--down)}}
.box{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;font-size:11px;color:var(--ink3);font-weight:500;padding:8px 10px;border-bottom:1px solid var(--line);background:#fafbfc}}
td{{padding:8px 10px;border-bottom:1px solid #f2f3f5}} td.num,th.num{{text-align:right}}
.verdict{{background:#fff;border:1px solid var(--line);border-left:3px solid #C43A2E;border-radius:10px;padding:14px 18px;margin:16px 0;font-size:13.5px}}
.note{{font-size:12px;color:var(--ink2);line-height:1.9}} .note li{{margin-left:18px}}
</style></head><body><div class="wrap">
<h1>行业互斥约束回测</h1>
<div class="sub">2020-01 起 · 主板 {len(data)} 只 · 参数同 BASE（trail 25% / 120 日 / 成本 0.4%）· 行业=东财分类（{len(set(ind_of_map.values()))} 个）· 生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<div class="verdict">行业互斥把等权池从 {results['A']['stats']['n']} 笔压到 {results['B']['stats']['n']} 笔（并发 {results['A']['conc']:.0f} → {results['B']['conc']:.0f} 只），
等权累计 {pct(results['A']['eq']['total'])} → <b>{pct(results['B']['eq']['total'])}</b>，回撤 {pct(results['A']['eq']['mdd'])} → <b>{pct(results['B']['eq']['mdd'])}</b>；
6 仓实盘口径 {pct(results['C']['eq']['total'])} → <b>{pct(results['D']['eq']['total'])}</b>，回撤 {pct(results['C']['eq']['mdd'])} → <b>{pct(results['D']['eq']['mdd'])}</b>。</div>
<h2>四组口径对照</h2>
<div class="box"><table><thead><tr><th>口径</th><th class="num">笔数</th><th class="num">胜率</th><th class="num">单笔净期望</th>
<th class="num">累计</th><th class="num">年化</th><th class="num">最大回撤</th><th class="num">Sharpe</th><th class="num">平均并发</th></tr></thead>
<tbody>{rows_cmp}</tbody></table></div>
<div class="sub">沪深300 同期 {pct(bench_m['total'])}，最大回撤 {pct(bench_m['mdd'])}。被行业互斥跳过的 {sk_stats.get('n',0)} 笔信号：单笔 {pct(sk_stats.get('avg',0))}、胜率 {sk_stats.get('win',0)*100:.1f}%（对照入选 {pct(results['B']['stats']['avg'])}）。</div>
<h2>净值曲线</h2>
<div class="box">{curveB}</div>
<div class="box">{curveA}</div>
<div class="box">{curveD}</div>
<h2>年度对照（等权口径）</h2>
<div class="box"><table><thead><tr><th>年份</th><th class="num">A 全信号</th><th class="num">B 行业互斥</th><th class="num">沪深300</th><th class="num">A 笔数</th><th class="num">B 笔数</th></tr></thead>
<tbody>{rows_y}</tbody></table></div>
<h2>口径与局限</h2>
<div class="box note"><ul>
<li>行业互斥规则与面板「当前持仓」完全一致：先入为主、平仓当日开盘释放行业、同日同行业按代码顺序取第一只；同一突破只做一次。</li>
<li>等权口径衡量约束对筛选 edge 的影响；6 仓口径衡量对实际执行的影响。两者都含 0.4% 往返成本。</li>
<li>脆弱性：剔除收益最高 50 笔后，A 单笔 {pct(results['A']['frag50'])}、B 单笔 {pct(results['B']['frag50'])}——收益仍高度依赖少数大牛股。</li>
<li>行业分类用当前东财行业（历史行业变更未还原）；股票池存在生存者偏差；未考虑涨跌停封板与冲击成本。</li>
<li>回测只检验规则的历史统计特征，不构成投资建议。</li>
</ul></div>
</div></body></html>"""
    (ROOT / "industry.html").write_text(html, encoding="utf-8")
    out = {"config": cfg, "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
           "benchmark": bench_m, "yearly": yearly, "skipped": sk_stats,
           "n_industries": len(set(ind_of_map.values()))}
    (ROOT / "industry.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("→ industry.html / industry.json")


if __name__ == "__main__":
    main()
