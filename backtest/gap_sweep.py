#!/usr/bin/env python3
"""向下跳空过滤扫描：信号次日开盘大幅低开时是否该放弃买入。

逻辑：涨停/突破后次日跌停开（gap < -x%），说明隔夜情绪反转，照买往往接飞刀。
扫 0%（基准，不拦）/3%/5%/7%/9%/10%，看 6 仓实盘与等权两种组合，
并统计被拦掉的单子本来能赚多少（判断拦得对不对）。
其余参数同 BASE（trail 25%、120 日、不砍破位、6 仓、2% 风险预算）。
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
                k = sig["signal_idx"]
                if k + 1 < len(b["dates"]):
                    gap = b["o"][k + 1] / b["c"][k] - 1
                    sigs.append((code, b, sig, gap))
    print(f"[sig] {len(sigs)} 个原始信号", flush=True)

    # 先把每笔按基准口径模拟出来，缓存
    all_trades = []
    for code, b, sig, gap in sigs:
        t = simulate(b, sig, cfg, "base")
        if not t:
            continue
        t["code"] = code
        t["_bars"] = b
        t["_entry_i"] = sig["signal_idx"] + 1
        t["gap"] = gap
        all_trades.append(t)
    print(f"[sim] 可执行 {len(all_trades)} 笔", flush=True)

    rows = []
    for th in [0, 3, 5, 7, 9, 10]:
        cut = th / 100.0
        kept = [t for t in all_trades if t["gap"] >= -cut - 1e-9]
        blocked = [t for t in all_trades if t["gap"] < -cut - 1e-9]
        cfg2 = dict(cfg)
        eq = rb.build_equity(kept, dates, bench["close"], cfg2)
        si = next(i for i, d in enumerate(dates) if d >= EVAL_START)
        m = metrics_from_equity(eq[si:], dates[si:])
        st = trade_stats(kept)
        eqw, conc = build_equal_weight(kept, dates, cfg2)
        mw = metrics_from_equity(eqw[si:], dates[si:])
        # 被拦掉那批本来怎样
        bst = trade_stats(blocked) if blocked else {"n": 0, "avg": 0, "win": 0, "hold": 0}
        rows.append({"th": th, "kept": len(kept), "blocked": len(blocked), "port": m,
                     "stats": st, "eqw": mw, "bstats": bst,
                     "avg_gap": float(np.mean([t["gap"] for t in blocked])) if blocked else 0.0,
                     "curve": eq[si:]})
        print(f"gap_min={th:2d}%  保留{len(kept):6d} 拦掉{len(blocked):5d} "
              f"| 6仓 {m['total']*100:+7.2f}% 回撤{m['mdd']*100:6.2f}% Sharpe{m['sharpe']:5.2f} "
              f"| 单笔{st['avg']*100:+6.2f}% 胜率{st['win']*100:4.1f}% "
              f"| 被拦{bst['n']:5d} 单笔{bst['avg']*100:+6.2f}% 胜率{bst['win']*100:4.1f}%", flush=True)

    def pct(v): return f"{v*100:+.2f}%"
    def cls(v): return "pos" if v > 0 else "neg"

    body = ""
    for r in rows:
        m, st, mw, bst = r["port"], r["stats"], r["eqw"], r["bstats"]
        lab = "基准（不拦）" if r["th"] == 0 else "向下跳空 > %d%% 放弃" % r["th"]
        body += (f"<tr><td>{lab}</td>"
                 f"<td class='num'>{r['kept']}</td><td class='num'>{r['blocked']}</td>"
                 f"<td class='num'>{st['win']*100:.1f}%</td>"
                 f"<td class='num {cls(st['avg'])}'>{pct(st['avg'])}</td>"
                 f"<td class='num {cls(m['total'])}'>{pct(m['total'])}</td>"
                 f"<td class='num {cls(m['mdd'])}'>{pct(m['mdd'])}</td>"
                 f"<td class='num'>{m['sharpe']:.2f}</td>"
                 f"<td class='num {cls(mw['total'])}'>{pct(mw['total'])}</td>"
                 f"<td class='num {cls(bst['avg'])}'>{pct(bst['avg'])}</td></tr>")

    si = next(i for i, d in enumerate(dates) if d >= EVAL_START)
    bench_curve = bench["close"][si:].astype(float)
    bench_curve = bench_curve / bench_curve[0]
    curves = ""
    for r in rows:
        n = min(len(r["curve"]), len(bench_curve))
        curves += (f"<div class='box'><div class='cap'>gap_min {r['th']}% · 6 仓 "
                   f"{pct(r['port']['total'])} / 回撤 {pct(r['port']['mdd'])}</div>"
                   f"{svg_curve(r['curve'][:n].tolist(), bench_curve[:n].tolist(), dates[si:si+n], '6 仓实盘（红）')}</div>")

    best = max(rows, key=lambda r: r["port"]["sharpe"])
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>向下跳空过滤扫描</title><style>
:root{{--ink:#2C2C2A;--ink2:#5F5E5A;--ink3:#888780;--line:#e3e5e8;--up:#C43A2E;--down:#1D7A4C}}
*{{box-sizing:border-box;margin:0;padding:0}}body{{background:#f5f6f8;color:var(--ink);font:14px/1.65 -apple-system,"PingFang SC",sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:30px 24px 70px}}h1{{font-size:24px;font-weight:650}}
h2{{font-size:15px;font-weight:600;margin:26px 0 10px}}.sub{{color:var(--ink3);font-size:12px;margin-top:6px}}
.box{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{text-align:left;font-size:11px;color:var(--ink3);font-weight:500;padding:8px 8px;border-bottom:1px solid var(--line);background:#fafbfc}}
td{{padding:8px 8px;border-bottom:1px solid #f2f3f5}}td.num,th.num{{text-align:right}}
.pos{{color:var(--up)}}.neg{{color:var(--down)}}.cap{{font-size:12px;color:var(--ink3);margin-bottom:6px}}
.note{{font-size:12.5px;color:var(--ink2);line-height:1.9}}.note li{{margin-left:18px}}
</style></head><body><div class="wrap">
<h1>向下跳空过滤扫描</h1>
<div class="sub">2020 起 · {len(data)} 只主板 · 突破次日开盘买入 · 6 仓 / 等权 · 成本 0.4% · 生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<h2>结果</h2>
<div class="box"><table><thead><tr><th>入场规则</th><th class="num">保留笔数</th><th class="num">拦掉笔数</th>
<th class="num">胜率</th><th class="num">单笔净期望</th><th class="num">6仓累计</th><th class="num">6仓回撤</th>
<th class="num">Sharpe</th><th class="num">等权累计</th><th class="num">被拦单笔</th></tr></thead><tbody>{body}</tbody></table></div>
<div class="sub">「被拦单笔」= 那些向下跳空超阈值的单子如果不拦、照基准持有，本来能拿到多少。这项为负说明拦得对。</div>
<h2>净值曲线</h2>{curves}
<h2>怎么读</h2>
<div class="box note"><ul>
<li>向下跳空和向上跳空不同：大幅低开往往是隔夜利空/涨停炸板，是「信号失效」的表现，而不只是买贵了。</li>
<li>判断标准看两列：拦掉之后组合变好（累计升、回撤降），且「被拦单笔」明显为负，才算拦得对。</li>
<li>阈值越高拦得越少。Shapre 最高的档位未必最稳，还要看保留笔数是否被砍得太狠。</li>
</ul></div>
<div class="sub">脚本 <code>backtest/gap_sweep.py</code>　·　历史统计不代表未来。</div>
</div></body></html>"""
    (ROOT / "gap_sweep.html").write_text(html, encoding="utf-8")
    (ROOT / "gap_sweep.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "curve"} for r in rows],
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"→ gap_sweep.html / gap_sweep.json  |  Sharpe 最优档 gap_min={best['th']}%")


if __name__ == "__main__":
    main()
