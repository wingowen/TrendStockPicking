#!/usr/bin/env python3
"""出场口径对照：破位止损 / 峰值回撤 25% / 浮盈回撤 25 个百分点。

用户诉求：破位就卖；收益回撤 25% 也卖。这里把「回撤」拆成两种可执行定义：
  T1  价格从峰值回撤 25%（= 现行 trail，相对最高收盘价）
  T2  浮盈从峰值回撤 25 个百分点（只有曾经盈利才触发；从未盈利不砍）
另加 P  收盘跌破突破位就卖（level_stop）

组合出 6 种口径，6 仓实盘 + 等权两种组合，其余参数同 BASE。
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from run_backtest import (BASE, EVAL_START, WARMUP_BARS, load_data, find_signals,
                          trade_stats, metrics_from_equity, build_equal_weight, svg_curve)
import run_backtest as rb

ROOT = Path(__file__).resolve().parent


def simulate(bars, sig, cfg, mode):
    """mode: base / peak25 / gain25 / level / level_peak25 / level_gain25 / levelbuf3 / levelbuf5 / level2d / peak25_hs{N}"""
    hard = 0.0
    level_stop = False
    buf = 0.0
    two_day = False
    kind = "peak25"
    raw = mode
    if "_hs" in mode:
        raw, hs = mode.split("_hs")
        hard = float(hs) / 100.0
    if raw == "base":
        kind = "base"
    elif raw == "gain25":
        kind = "gain25"
    elif raw == "level_gain25":
        kind, level_stop = "gain25", True
    elif raw == "level_peak25":
        kind, level_stop = "peak25", True
    elif raw == "level":
        kind, level_stop = "peak25", True
    elif raw == "levelbuf3":
        kind, buf = "peak25", 0.03
    elif raw == "levelbuf5":
        kind, buf = "peak25", 0.05
    elif raw == "level2d":
        kind, two_day = "peak25", True
    elif raw == "gainrel":
        kind = "gainrel"
    elif raw == "losslevel":
        kind, level_stop = "peak25", True
    k = sig["signal_idx"]
    n = len(bars["c"])
    if k + 1 >= n:
        return None
    e = k + 1
    entry = bars["o"][e]
    if entry <= 0 or entry / bars["c"][k] - 1 > cfg["gap_max"]:
        return None
    level = sig["level"]
    peak = bars["c"][e]
    peak_gain = bars["c"][e] / entry - 1
    exit_idx, reason = None, "数据结束"
    for i in range(e, n):
        c = bars["c"][i]
        peak = max(peak, c)
        peak_gain = max(peak_gain, c / entry - 1)
        cur_gain = c / entry - 1
        if hard and cur_gain <= -hard:
            exit_idx, reason = i + 1, f"硬止损{int(hard*100)}%"
        elif level_stop and c < level and (raw != "losslevel" or cur_gain < 0):
            exit_idx, reason = i + 1, "亏损中跌破突破位" if raw == "losslevel" else "跌破突破位"
        elif buf and c < level * (1 - buf):
            exit_idx, reason = i + 1, f"破位{int(buf*100)}%"
        elif two_day and c < level and i - 1 >= e and bars["c"][i - 1] < level:
            exit_idx, reason = i + 1, "连续2日破位"
        elif kind in ("peak25", "base") and c < peak * (1 - cfg["trail"]):
            exit_idx, reason = i + 1, "峰值回撤25%"
        elif kind == "gain25" and peak_gain > 0 and cur_gain < peak_gain - 0.25:
            exit_idx, reason = i + 1, "浮盈回撤25点"
        elif kind == "gainrel" and peak_gain > 0 and cur_gain < peak_gain * 0.75:
            exit_idx, reason = i + 1, "浮盈回撤25%"
        elif i - e + 1 >= cfg["max_hold"]:
            exit_idx, reason = i + 1, "持仓到期"
        if exit_idx is not None:
            break
    if exit_idx is None or exit_idx >= n:
        exit_idx, exit_price, reason = n - 1, bars["c"][n - 1], "数据结束"
    else:
        exit_price = bars["o"][exit_idx]
    return {**sig, "entry_date": bars["dates"][e], "entry": entry,
            "exit_date": bars["dates"][exit_idx], "exit": exit_price,
            "hold": exit_idx - e + 1, "ret": exit_price / entry - 1 - cfg["cost"],
            "reason": reason,
            "mfe": float(np.max(bars["c"][e:exit_idx + 1]) / entry - 1),
            "mae": float(np.min(bars["c"][e:exit_idx + 1]) / entry - 1)}


def build_equity_simple(trades, idx_dates, cfg):
    """6 仓（等份资金，不含风险预算折算），避免与旧 build_equity 的差异引入噪声。"""
    date_pos = {d: i for i, d in enumerate(idx_dates)}
    cash, n = cfg["capital"], len(idx_dates)
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
        for tr in sorted(pend.get(di, []), key=lambda x: x["code"]):
            if len(positions) >= cfg["slots"] or cash <= 0:
                break
            if tr["code"] in held:
                continue
            buy_px = tr["entry"] * (1 + cfg["cost"] / 2)
            budget = min((cash + mv) / cfg["slots"], cash)
            shares = int(budget / buy_px / 100) * 100
            if shares <= 0:
                continue
            cash -= shares * buy_px
            held.add(tr["code"])
            positions.append({"code": tr["code"], "shares": shares,
                              "exit_idx": date_pos.get(tr["exit_date"], n - 1),
                              "exit_price": tr["exit"] * (1 - cfg["cost"] / 2),
                              "bars": tr["_bars"], "i": tr["_entry_i"], "date": d})
            mv += shares * buy_px
            tr["_used"] = True
        eq[di] = cash + mv
    return eq


def main():
    cfg = dict(BASE)
    cfg["level_stop"] = False
    stocks, data, idx = load_data()
    bench = idx["沪深300"]
    dates_all = bench["dates"]

    modes = [("A", "峰值回撤 25%（原基准）", "base"),
             ("B", "破位即卖（仅破位）", "level"),
             ("C", "破位 + 峰值回撤 25%", "level_peak25"),
             ("D", "浮盈回撤 25 点（不破位）", "gain25"),
             ("E", "破位 + 浮盈回撤 25 点", "level_gain25"),
             ("H", "峰值回撤25% + 硬止损10%", "peak25_hs10"),
             ("I", "峰值回撤25% + 硬止损15%", "peak25_hs15"),
             ("J", "峰值回撤25% + 硬止损8%", "peak25_hs8"),
             ("K", "峰值回撤25% + 破位3%缓冲", "levelbuf3"),
             ("L", "峰值回撤25% + 破位5%缓冲", "levelbuf5"),
             ("M", "峰值回撤25% + 连续2日破位", "level2d"),
             ("F", "浮盈比例回撤 25%（字面口径）", "gainrel"),
             ("G", "亏损中破位才卖", "losslevel")]

    # 预生成信号（各 mode 共用）
    sigs = []
    for code, b in data.items():
        if len(b["dates"]) < WARMUP_BARS:
            continue
        for sig in find_signals(b, cfg):
            if b["dates"][sig["signal_idx"]] >= EVAL_START:
                sigs.append((code, b, sig))
    print(f"[sig] {len(sigs)} 个原始信号", flush=True)

    results = []
    for key, label, mode in modes:
        trades = []
        for code, b, sig in sigs:
            tr = simulate(b, sig, cfg, mode)
            if not tr:
                continue
            tr["code"] = code
            tr["_bars"] = b
            tr["_entry_i"] = sig["signal_idx"] + 1
            trades.append(tr)
        trades.sort(key=lambda t: (t["entry_date"], t["code"]))
        st = trade_stats(trades)
        eq = rb.build_equity(trades, dates_all, bench["close"], cfg)
        start_i = next(i for i, d in enumerate(dates_all) if d >= trades[0]["entry_date"])
        m = metrics_from_equity(eq[start_i:], dates_all[start_i:])
        eqw, conc = build_equal_weight(trades, dates_all, cfg)
        mw = metrics_from_equity(eqw[start_i:], dates_all[start_i:])
        exits = {}
        for t in trades:
            e = exits.setdefault(t["reason"], [0, 0.0])
            e[0] += 1
            e[1] += t["ret"]
        exits = {k: {"n": v[0], "avg": v[1] / v[0]} for k, v in exits.items()}
        used = sum(1 for t in trades if t.get("_used"))
        results.append({"key": key, "label": label, "mode": mode, "stats": st, "port": m,
                        "eqw": mw, "avg_conc": conc, "exits": exits, "used": used,
                        "curve": eq[start_i:], "curve_ew": eqw[start_i:]})
        print(f"{key} {label:28s} 信号{st['n']:6d} 胜率{st['win']*100:5.1f}% 单笔{st['avg']*100:+6.2f}% "
              f"| 6仓{m['total']*100:+7.2f}% 回撤{m['mdd']*100:6.2f}% Sharpe{m['sharpe']:5.2f} "
              f"| 等权{mw['total']*100:+7.2f}% 回撤{mw['mdd']*100:6.2f}%", flush=True)

    start_i = next(i for i, d in enumerate(dates_all) if d >= EVAL_START)
    win_dates = dates_all[start_i:]
    bench_curve = bench["close"][start_i:]
    bench_curve = bench_curve / bench_curve[0]
    bench_m = metrics_from_equity(bench_curve, win_dates)

    def pct(v): return f"{v*100:+.2f}%"
    def cls(v): return "pos" if v > 0 else "neg"

    rows = ""
    for r in results:
        st, m, mw = r["stats"], r["port"], r["eqw"]
        rows += (f"<tr><td><b>{r['key']}</b> {r['label']}</td><td class='num'>{st['n']}</td>"
                 f"<td class='num'>{st['win']*100:.1f}%</td><td class='num {cls(st['avg'])}'>{pct(st['avg'])}</td>"
                 f"<td class='num'>{st['hold']:.0f}</td>"
                 f"<td class='num {cls(m['total'])}'>{pct(m['total'])}</td>"
                 f"<td class='num {cls(m['mdd'])}'>{pct(m['mdd'])}</td><td class='num'>{m['sharpe']:.2f}</td>"
                 f"<td class='num {cls(mw['total'])}'>{pct(mw['total'])}</td>"
                 f"<td class='num {cls(mw['mdd'])}'>{pct(mw['mdd'])}</td></tr>")
    exit_rows = ""
    for r in results:
        ex = "；".join(f"{k} {v['n']}笔 {pct(v['avg'])}" for k, v in r["exits"].items())
        exit_rows += f"<tr><td><b>{r['key']}</b></td><td colspan='2' style='font-size:12px'>{ex}</td></tr>"

    curves = ""
    for r in results:
        n = min(len(r["curve"]), len(bench_curve))
        curves += f"<div class='box'><div class='cap'>{r['key']} · {r['label']}</div>{svg_curve(r['curve'][:n].tolist(), bench_curve[:n].tolist(), win_dates[:n], '6 仓实盘（红）')}</div>"

    a, c = results[0], results[2]
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>出场口径对照</title><style>
:root{{--ink:#2C2C2A;--ink2:#5F5E5A;--ink3:#888780;--line:#e3e5e8;--up:#C43A2E;--down:#1D7A4C}}
*{{box-sizing:border-box;margin:0;padding:0}}body{{background:#f5f6f8;color:var(--ink);font:14px/1.65 -apple-system,"PingFang SC",sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:30px 24px 70px}}h1{{font-size:24px;font-weight:650}}
h2{{font-size:15px;font-weight:600;margin:26px 0 10px}}.sub{{color:var(--ink3);font-size:12px;margin-top:6px}}
.box{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;font-size:11px;color:var(--ink3);font-weight:500;padding:8px 10px;border-bottom:1px solid var(--line);background:#fafbfc}}
td{{padding:8px 10px;border-bottom:1px solid #f2f3f5}}td.num,th.num{{text-align:right}}
.pos{{color:var(--up)}}.neg{{color:var(--down)}}.cap{{font-size:12px;color:var(--ink3);margin-bottom:6px}}
.verdict{{background:#fff;border:1px solid var(--line);border-left:3px solid #C43A2E;border-radius:10px;padding:14px 18px;margin:16px 0;font-size:13.5px}}
.note{{font-size:12px;color:var(--ink2);line-height:1.9}}.note li{{margin-left:18px}}
</style></head><body><div class="wrap">
<h1>出场口径对照 · 破位止损 vs 回撤止盈</h1>
<div class="sub">2020 起 · 主板 {len(data)} 只 · 6 仓 + 等权两种组合 · 成本 0.4% · 生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<div class="verdict">
A（原基准，峰值回撤 25%、不砍破位）：6 仓 {pct(a['port']['total'])}／回撤 {pct(a['port']['mdd'])}，单笔 {pct(a['stats']['avg'])}，平均持有 {a['stats']['hold']:.0f} 天。<br>
C（<b>你要的口径：破位就卖 + 回撤 25%</b>）：6 仓 <b>{pct(c['port']['total'])}</b>／回撤 <b>{pct(c['port']['mdd'])}</b>，单笔 <b>{pct(c['stats']['avg'])}</b>，平均持有 {c['stats']['hold']:.0f} 天，成交 {c['used']} 笔。
</div>
<h2>六种口径（6 仓 = 2% 风险预算口径；等权 = 每笔等额、衡量筛选本身）</h2>
<div class="box"><table><thead><tr><th>口径</th><th class="num">信号数</th><th class="num">胜率</th><th class="num">单笔净期望</th>
<th class="num">平均持有</th><th class="num">6仓累计</th><th class="num">6仓回撤</th><th class="num">Sharpe</th>
<th class="num">等权累计</th><th class="num">等权回撤</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="sub">沪深300 同期 {pct(bench_m['total'])}，回撤 {pct(bench_m['mdd'])}。</div>
<h2>退出原因分布</h2>
<div class="box"><table><thead><tr><th>口径</th><th colspan="2">各出场原因的笔数与平均收益</th></tr></thead><tbody>{exit_rows}</tbody></table></div>
<h2>净值曲线（6 仓实盘口径）</h2>{curves}
<h2>怎么读这份结果</h2>
<div class="box note"><ul>
<li>A 的「回撤 25%」是<b>价格</b>从持仓最高收盘回落 25%，本质是宽幅移动止盈；它有 120 天硬上限兜底，所以持仓最长 120 天。</li>
<li>B 只砍破位：收盘低于突破位即卖，回撤往往只有 3–10 个百分点，属于「小止损」。历史上这是主要亏损来源——突破后回踩到突破位附近是常态，被噪声洗出的概率很高。</li>
<li>C 叠加两者：先被破位砍掉大部分仓位，剩下没砍的再按 25% 回撤止盈。等于把趋势单的容错空间压到最窄。</li>
<li>D/E 用「浮盈回撤 25 个百分点」替代「价格回撤 25%」：只有真正赚到钱之后才启用止盈，没赚钱就一直放着（有 120 天兜底）。这是最贴近你原话「收益回撤 25%」的可执行版本。</li>
</ul></div>
<div class="sub">逐笔可复现：<code>backtest/exit_compare.py</code>　·　回测只检验历史统计特征，不构成投资建议。</div>
</div></body></html>"""
    (ROOT / "exit_compare.html").write_text(html, encoding="utf-8")
    (ROOT / "exit_compare.json").write_text(json.dumps({
        "benchmark": bench_m,
        "results": [{k: v for k, v in r.items() if k not in ("curve", "curve_ew")} for r in results]
    }, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("→ exit_compare.html / exit_compare.json")


if __name__ == "__main__":
    main()
