#!/usr/bin/env python3
"""v3 冻结口径 · 唯一对外数字来源

所有文档（README / backtest/README / rules.html）、面板（build_panel_from_db.py）
引用的绩效数字都必须来自本脚本写出的 spec_v3.json，禁止再手写。

    .venv-bt/bin/python run_v3.py            # 约 3-5 分钟
    .venv-bt/bin/python run_v3.py --quick    # 只算主口径，跳过稳健性
输出：spec_v3.json / spec_v3.html
"""
import argparse
import json
import math
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

import engine_v2 as E

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "spec_v3.json"
OUT_HTML = ROOT / "spec_v3.html"
TRAIN_START, TRAIN_END = E.V3_TRAIN


def period_stats(rows):
    out = []
    for tag, lo, hi in (("训练 2020-2021", TRAIN_START, TRAIN_END),
                        ("验证 2022-2026", TRAIN_END, "9999")):
        rs = [r for r in rows if lo <= r["entry_date"] < hi
              and r["fwd20"] is not None and r["bench20"] is not None]
        f = np.array([r["fwd20"] for r in rs])
        b = np.array([r["bench20"] for r in rs])
        d = f - b
        out.append({
            "period": tag, "n": len(rs),
            "bench20": float(b.mean()), "fwd20": float(f.mean()),
            "excess20": float(d.mean()),
            "t_excess": float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))),
            "excess_win": float((d > 0).mean()),
        })
    return out


def quintiles(rows):
    rs = [r for r in rows if r["entry_date"] >= TRAIN_END
          and r["fwd20"] is not None and r["bench20"] is not None]
    s = np.array([r["score3"] for r in rs])
    y = np.array([r["fwd20"] - r["bench20"] for r in rs])
    order = np.argsort(s, kind="stable")
    buckets = np.array_split(order, 5)
    return [{"q": i + 1, "n": int(len(b)),
             "excess20": float(y[b].mean()),
             "win": float((y[b] > 0).mean()),
             "score_lo": float(s[b].min()), "score_hi": float(s[b].max())}
            for i, b in enumerate(buckets)]


def year_rows(rows, dates, closes, lot_price, sc):
    """冻结分数，逐年独立起跑（每年重置资金），暴露「只靠某一年」。"""
    out = []
    for y in range(2020, 2027):
        lo, hi = f"{y}-01-01", f"{y + 1}-01-01"
        sub = [r for r in rows if lo <= r["entry_date"] < hi]
        if not sub:
            continue
        i0 = next((i for i, d in enumerate(dates) if d >= lo), 0)
        i1 = next((i for i, d in enumerate(dates) if d >= hi), len(dates))
        eq, m = E.run_cfg(sub, dates, closes, lot_price, E.V3_SPEC, sc, start=lo)
        out.append({"year": y, "total": m.get("total", 0), "mdd": m.get("mdd", 0),
                    "fills": m.get("fills", 0), "avg_trade": m.get("avg_trade"),
                    "win": m.get("win")})
    return out


def concentration(filled):
    if not filled:
        return {}
    r = np.array([t["ret"] for t in filled])
    s = np.sort(r)[::-1]
    return {"n": int(len(r)), "avg": float(r.mean()), "med": float(np.median(r)),
            "win": float((r > 0).mean()), "sd": float(r.std(ddof=1)),
            "top1": float(s[0]), "top3_share": float(s[:3].sum() / s.sum()) if s.sum() else None,
            "worst": float(s[-1]),
            "avg_win": float(r[r > 0].mean()) if (r > 0).any() else None,
            "avg_loss": float(r[r <= 0].mean()) if (r <= 0).any() else None}


def capacity(rows):
    n_sig = len([r for r in rows if r["entry_date"] >= TRAIN_END]) / 5.0
    avg_hold = float(np.mean([r["hold"] for r in rows]))
    cap = E.V3_SPEC["slots"] * TRAIN_DAYS / avg_hold
    return {"signals_per_year": n_sig, "avg_hold": avg_hold,
            "capacity_per_year": cap, "fill_ratio": cap / n_sig}


TRAIN_DAYS = E.TRADING_DAYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    rows = E.load_features()
    for r in rows:
        r["score3"] = E.score_v3(r)
    dates, bench = E.load_calendar()
    closes, lot_price = E.prep(dates, rows)
    sc = {(r["code"], r["entry_date"]): r["score3"] for r in rows}

    # 1) 信号层：全样本是否跑赢同池等权基准
    per = period_stats(rows)
    print("[信号层]")
    for p in per:
        print(f"  {p['period']}: n={p['n']} 基准{p['bench20']:+.2%} "
              f"信号{p['fwd20']:+.2%} 超额{p['excess20']:+.2%} (t={p['t_excess']:+.2f})")

    # 2) 排序分样本外五分位
    q = quintiles(rows)
    print("[排序分 样本外五分位]", " → ".join(f"Q{x['q']}:{x['excess20']:+.2%}" for x in q))

    # 3) 组合：全样本 / 样本外
    eq_all, m_all = E.run_cfg(rows, dates, closes, lot_price, E.V3_SPEC, sc)
    oos = [r for r in rows if r["entry_date"] >= TRAIN_END]
    eq_oos, m_oos = E.run_cfg(oos, dates, closes, lot_price, E.V3_SPEC, sc, start=TRAIN_END)
    print(f"[组合 全样本] 累计{m_all['total']:+.1%} 年化{m_all['cagr']:+.2%} "
          f"回撤{m_all['mdd']:.1%} Sharpe{m_all['sharpe']:.2f} 成交{m_all['fills']}")
    print(f"[组合 样本外] 累计{m_oos['total']:+.1%} 年化{m_oos['cagr']:+.2%} "
          f"回撤{m_oos['mdd']:.1%} Sharpe{m_oos['sharpe']:.2f} 成交{m_oos['fills']}")

    fills = [t for t in E.simulate(oos, dates, dict(E.V3_SPEC, closes=closes,
                                                   lot_price=lot_price), sc)["filled"]]
    per_year = year_rows(rows, dates, closes, lot_price, sc)
    cap = capacity(rows)
    idx_stats = {}
    if bench is not None:
        i0 = next((i for i, d in enumerate(dates) if d >= TRAIN_END), 0)
        h = np.array(bench[i0:], float)
        idx_stats = {"total": float(h[-1] / h[0] - 1),
                     "mdd": float((h / np.maximum.accumulate(h) - 1).min())}

    # 4) 稳健性：随机填仓 / 年度 bootstrap
    if a.quick:
        sh, bs = None, None
    else:
        sh = E.shuffle_paths(rows, dates, closes, lot_price, E.V3_SPEC, n=30)
        bs = E.bootstrap_years(rows, dates, closes, lot_price, E.V3_SPEC, sc, n=200)
        print("[随机填仓]", {k: round(v, 4) for k, v in sh.items()})
        print("[年度 bootstrap]", {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in bs.items()})

    doc = {
        "version": "v3-frozen",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "env": f"python {platform.python_version()}",
        "spec": E.V3_SPEC,
        "score": {"train": list(E.V3_TRAIN), "mu": list(E.V3_MU), "sd": list(E.V3_SD),
                  "w": list(E.V3_W),
                  "features": ["regime(强+1/弱-1)", "gain(距突破位)", "days_above", "vol_ratio"]},
        "periods": per,
        "quintiles_oos": q,
        "portfolio_all": {k: v for k, v in m_all.items() if k != "signals"},
        "portfolio_oos": {k: v for k, v in m_oos.items() if k != "signals"},
        "trade_stats_oos": concentration(fills),
        "per_year": per_year,
        "capacity": cap,
        "bench_oos": idx_stats,
        "shuffle": sh,
        "bootstrap": bs,
        "signals_total": len(rows),
        "period": f"{min(r['entry_date'] for r in rows)} → {max(r['exit_date'] for r in rows)}",
    }
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float),
                        encoding="utf-8")
    OUT_HTML.write_text(render_html(doc), encoding="utf-8")
    print(f"→ {OUT_JSON.name} / {OUT_HTML.name}")


def pct(v, d=1):
    return "—" if v is None else f"{v * 100:+.{d}f}%"


def render_html(d):
    per = "".join(
        f"<tr><td>{p['period']}</td><td class=num>{p['n']}</td><td class=num>{pct(p['bench20'],2)}</td>"
        f"<td class=num>{pct(p['fwd20'],2)}</td><td class=num>{pct(p['excess20'],2)}</td>"
        f"<td class=num>{p['t_excess']:+.2f}</td></tr>" for p in d["periods"])
    qs = "".join(f"<tr><td>Q{x['q']}</td><td class=num>{x['n']}</td>"
                 f"<td class=num>{pct(x['excess20'],2)}</td><td class=num>{x['win']*100:.0f}%</td>"
                 f"<td class=num>{x['score_lo']:.3f} ~ {x['score_hi']:.3f}</td></tr>"
                 for x in d["quintiles_oos"])
    yrs = "".join(f"<tr><td>{y['year']}</td><td class=num>{pct(y['total'])}</td>"
                  f"<td class=num>{pct(y['mdd'])}</td><td class=num>{y['fills']}</td>"
                  f"<td class=num>{pct(y['avg_trade'],2)}</td></tr>" for y in d["per_year"])
    a, o, t, c = d["portfolio_all"], d["portfolio_oos"], d.get("trade_stats_oos", {}), d["capacity"]
    sh, bs = d.get("shuffle"), d.get("bootstrap")
    robust = ""
    if sh:
        robust += (f"<li>随机填仓 30 次：年化中位 {pct(sh['cagr_med'],2)}，"
                   f"5–95% {pct(sh['cagr_p5'],2)} ~ {pct(sh['cagr_p95'],2)}，"
                   f"为正比例 {sh['positive_frac']*100:.0f}%。</li>")
    if bs:
        robust += (f"<li>年度 bootstrap 200 次：累计中位 {pct(bs['total_med'])}，"
                   f"5–95% {pct(bs['total_p5'])} ~ {pct(bs['total_p95'])}。</li>")
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>突破趋势 v3 冻结口径</title><style>
body{{font:15px/1.7 -apple-system,"PingFang SC",sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#222}}
h1{{font-size:24px}} h2{{font-size:18px;margin-top:32px;border-left:3px solid #4a6cf7;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border-bottom:1px solid #eee;padding:8px 10px;text-align:left;font-size:14px}}
.num{{text-align:right;font-variant-numeric:tabular-nums}} .sub{{color:#888;font-size:13px}}
</style></head><body>
<h1>突破趋势跟踪 · v3 冻结口径</h1>
<div class="sub">{d['period']} · 信号 {d['signals_total']} 笔 · 生成 {d['generated']} · {d['env']}</div>
<h2>一、信号层是否跑赢同池基准</h2>
<table><thead><tr><th>区间</th><th class=num>信号数</th><th class=num>基准20日</th>
<th class=num>信号20日</th><th class=num>超额</th><th class=num>t</th></tr></thead><tbody>{per}</tbody></table>
<div class="sub">基准＝同池等权持有；训练期超额显著、验证期接近 0，是本策略最关键的失效证据。</div>
<h2>二、排序分样本外五分位（2022-2026）</h2>
<table><thead><tr><th>分位</th><th class=num>样本</th><th class=num>20日超额</th>
<th class=num>胜率</th><th class=num>分数区间</th></tr></thead><tbody>{qs}</tbody></table>
<h2>三、组合层（10 仓等额 · 强市开仓 · 现金 1.5% · 回撤 10/15% 降档）</h2>
<table><thead><tr><th>口径</th><th class=num>成交</th><th class=num>累计</th><th class=num>年化</th>
<th class=num>回撤</th><th class=num>Sharpe</th><th class=num>仓位</th></tr></thead><tbody>
<tr><td>全样本 2020-2026</td><td class=num>{a['fills']}</td><td class=num>{pct(a['total'])}</td>
<td class=num>{pct(a['cagr'],2)}</td><td class=num>{pct(a['mdd'])}</td>
<td class=num>{a['sharpe']:.2f}</td><td class=num>{a['avg_exposure']*100:.0f}%</td></tr>
<tr><td>样本外 2022-2026</td><td class=num>{o['fills']}</td><td class=num>{pct(o['total'])}</td>
<td class=num>{pct(o['cagr'],2)}</td><td class=num>{pct(o['mdd'])}</td>
<td class=num>{o['sharpe']:.2f}</td><td class=num>{o['avg_exposure']*100:.0f}%</td></tr>
</tbody></table>
<div class="sub">样本外单笔：均值 {pct(t.get('avg'),2)} · 中位 {pct(t.get('med'),2)} · 胜率 {t.get('win',0)*100:.0f}% ·
最差 {pct(t.get('worst'))} · 前 3 笔占全部盈利 {t.get('top3_share',0)*100:.0f}%。</div>
<h2>四、逐年（冻结分数、每年独立起跑）</h2>
<table><thead><tr><th>年份</th><th class=num>收益</th><th class=num>回撤</th>
<th class=num>成交</th><th class=num>单笔均值</th></tr></thead><tbody>{yrs}</tbody></table>
<h2>五、容量算术</h2>
<div class="sub">平均持仓 {c['avg_hold']:.0f} 天，{E.V3_SPEC['slots']} 仓 → 每年最多
{c['capacity_per_year']:.0f} 笔；同期信号 {c['signals_per_year']:.0f} 笔/年 →
只能成交 {c['fill_ratio']*100:.0f}%。这就是「单笔有梯度、组合没收益」的结构原因。</div>
<h2>六、稳健性</h2>
<ul>{robust}</ul>
</body></html>"""


if __name__ == "__main__":
    main()
