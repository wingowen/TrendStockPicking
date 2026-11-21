"""把 alt_scans.json + cross_check.json + cross_check_ctrl.json 渲染成一份可读报告。"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
J = json.loads((ROOT / "alt_scans.json").read_text(encoding="utf-8"))
C = json.loads((ROOT / "cross_check.json").read_text(encoding="utf-8"))
CT = json.loads((ROOT / "cross_check_ctrl.json").read_text(encoding="utf-8"))


def p(x, d=2):
    return "—" if x is None else f"{x*100:+.{d}f}%"


def ann(tot, months, hold=1):
    """months 为窗口数，hold 为每个窗口几个月。"""
    span = months * hold
    return (1 + tot) ** (12 / span) - 1 if span else float("nan")


V = J["variants"]
HOLD = {"10仓·月调": 1, "10仓·月调·零成本": 1, "30仓·季调": 3, "50仓·半年调": 6,
        "30仓·季调·零成本": 3, "30仓·季调·起点+1月": 3, "30仓·季调·起点+2月": 3}
rows_var = []
for name in ("10仓·月调", "10仓·月调·零成本", "30仓·季调", "50仓·半年调",
             "30仓·季调·零成本", "30仓·季调·起点+1月", "30仓·季调·起点+2月"):
    v = V[name]
    rows_var.append(
        f"<tr><td>{name}</td><td class=num>{p(v['all_total'],1)}</td>"
        f"<td class=num>{p(v['oos_total'],1)}</td>"
        f"<td class=num>{p(ann(v['oos_total'], v['months_oos'], HOLD[name]),1)}</td>"
        f"<td class=num>{v['months_oos']}</td>"
        f"<td class=num>{v['t_oos']:+.2f}</td>"
        f"<td class=num>{v['avg_turnover']*100:.0f}%</td>"
        f"<td class=num>{p(v['win_oos'],0)}</td></tr>")

rows_fac = []
for n, r in J["factors"].items():
    rows_fac.append(
        f"<tr><td>{n}</td><td class=num>{r['ic_train']:+.3f} ({r['t_train']:+.2f})</td>"
        f"<td class=num>{r['ic_oos20']:+.3f} ({r['t_oos20']:+.2f})</td>"
        f"<td class=num>{r['ic_oos60']:+.3f} ({r['t_oos60']:+.2f})</td>"
        f"<td class=num>{'是' if r['same_sign'] else '否'}</td>"
        f"<td class=num>{r['folds']['2022-2023 验证A']:+.3f} / {r['folds']['2024-2026 验证B']:+.3f}</td>"
        f"<td class=num>{' → '.join(f'{x*100:+.2f}%' for x in r['q_oos'])}</td></tr>")

rows_cross = []
for q in ("1", "2", "3", "4", "5"):
    c = J["cross_breakout"].get(q)
    if c:
        rows_cross.append(f"<tr><td>Q{q}（组合分 {q}/5 档）</td><td class=num>{c['n']}</td>"
                          f"<td class=num>{p(c['exc20'])}</td><td class=num>{c['t']:+.2f}</td></tr>")
hv = J["cross_breakout"].get("hi_vs_lo", {})
rows_cross.append(f"<tr><td><b>Q4+Q5 减 Q1+Q2</b></td><td class=num>—</td>"
                  f"<td class=num><b>{p(hv.get('exc20'))}</b></td>"
                  f"<td class=num><b>{hv.get('t', 0):+.2f}</b></td></tr>")

rows_eng = []
for e in C:
    ctrl = CT.get(e["name"].split()[0])
    extra = ""
    if ctrl:
        extra = (f"<td class=num>{p(ctrl['random_median'],1)}</td>"
                 f"<td class=num>{p(ctrl['p5'],1)} ~ {p(ctrl['p95'],1)}</td>")
    else:
        extra = "<td class=num>—</td><td class=num>—</td>"
    rows_eng.append(f"<tr><td>{e['name']}</td><td class=num>{int(e['n_all'])}</td>"
                    f"<td class=num>{p(e['total'],1)}</td><td class=num>{p(e['cagr'])}</td>"
                    f"<td class=num>{p(e['mdd'],1)}</td><td class=num>{e['sharpe']:.2f}</td>"
                    f"<td class=num>{e['fills']}</td>{extra}</tr>")

combo = J["combo"]
folds = combo["ic_folds"]
w = combo["weights"]
html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>横截面方向与交叉验证</title><style>
body{{font:15px/1.7 -apple-system,"PingFang SC",sans-serif;max-width:940px;margin:40px auto;padding:0 20px;color:#222}}
h1{{font-size:24px}} h2{{font-size:18px;margin-top:34px;border-left:3px solid #4a6cf7;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border-bottom:1px solid #eee;padding:7px 9px;text-align:left;font-size:13.5px}}
.num{{text-align:right;font-variant-numeric:tabular-nums}} .sub{{color:#888;font-size:13px}}
.box{{background:#f7f8fc;border:1px solid #e3e7f5;padding:12px 14px;margin:14px 0;font-size:14px}}
.warn{{background:#fff8f2;border:1px solid #f3e0cf}}
b{{color:#1a1a1a}}
</style></head><body>
<h1>换一条路：横截面反转/低波 + 四道交叉验证</h1>
<div class="sub">数据 {J['universe']['dates']} 交易日 × {J['universe']['codes']} 只 ·
{J['universe']['monthly']} 个月末截面 · 训练 {J['universe']['train'][0]}→{J['universe']['train'][1]} ·
冻结验证 {J['universe']['oos'][0]}→{J['universe']['oos'][1]} · 生成 {datetime.now():%Y-%m-%d %H:%M} ·
后复权、人民币、成本 0.4%/新进仓位</div>

<div class="box"><b>一句话结论</b>：因子层是站得住的（8 个因子在 20 日与 60 日两个标签上符号全一致，
三折 IC {folds["2020-2021 训练"]:.3f} / {folds['2022-2023 验证A']:.3f} / {folds['2024-2026 验证B']:.3f}，验证期反而更稳），
但<b>做成组合只有「宽仓位 + 低频」能留下钱</b>：30 仓季调样本外 {p(V['30仓·季调']['oos_total'],1)}，
而 10 仓月调只有 {p(V['10仓·月调']['oos_total'],1)}。所有变体的样本外 t 值都 &lt; 1，
所以这只能算「方向可试」，不能算「已验证的 alpha」。真正能拿去用的，是把这套分当
<b>排除器</b>（剔除最低 20% 分位的突破信号，t = {J['cross_breakout']['1']['t']:+.2f}），
不是当筛选器。</div>

<h2>一、因子层：先证明这不是标签挑出来的</h2>
<table><thead><tr><th>因子</th><th class=num>训练 IC (t)</th><th class=num>验证 IC 20日 (t)</th>
<th class=num>验证 IC 60日 (t)</th><th class=num>符号一致</th>
<th class=num>验证A / 验证B IC</th><th class=num>验证五分位 Q1→Q5</th></tr></thead>
<tbody>{''.join(rows_fac)}</tbody></table>
<div class="sub">IC 为月末截面 Spearman 相关；五分位是各档下期 20 日收益（未扣成本）。
mom* 为动量（取负权重＝反向用，即短期反转更有效）；vol20 取负＝低波动；amt_trend 取负＝量能未放大；
dist_high250 是唯一 t &lt; 1 的因子（接近历史高点这件事本身没有信息）。</div>

<h2>二、四道交叉验证</h2>
<div class="box"><b>① 标签交叉</b>：同一批截面同时用 20 日与 60 日两个前瞻窗口，8/8 因子符号一致，
60 日 IC 普遍更高（vol20 {J['factors']['vol20']['ic_oos20']:+.3f} → {J['factors']['vol20']['ic_oos60']:+.3f}），
说明不是标签窗口的偶然。<br>
<b>② 分折交叉</b>：训练 {folds['2020-2021 训练']:.3f}、验证A {folds['2022-2023 验证A']:.3f}、
验证B {folds['2024-2026 验证B']:.3f} —— 三段几乎一条线，没有「训练好、验证崩」。<br>
<b>③ 时点交叉</b>：把季调起点整体平移 0/1/2 个月，样本外 {p(V['30仓·季调']['oos_total'],1)} /
{p(V['30仓·季调·起点+1月']['oos_total'],1)} / {p(V['30仓·季调·起点+2月']['oos_total'],1)} 全为正，
但幅度差 4 倍 —— 方向不是单一时点幸运，幅度不可当点估计信。<br>
<b>④ 范式交叉</b>：拿 20430 条旧突破信号，按它所在月末的 combo 分位分档，看信号自身 20 日超额（见下表）。</div>

<h2>三、组合层：换手成本把 10 仓吃干</h2>
<table><thead><tr><th>组合变体</th><th class=num>全样本</th><th class=num>样本外</th>
<th class=num>样本外年化</th><th class=num>窗口数</th><th class=num>样本外 t</th>
<th class=num>月均换手</th><th class=num>胜率</th></tr></thead>
<tbody>{''.join(rows_var)}</tbody></table>
<div class="sub">月调 10 仓：毛收益 {p(V['10仓·月调·零成本']['oos_total'],1)} → 扣成本
{p(V['10仓·月调']['oos_total'],1)}，成本吃掉约 {abs(V['10仓·月调·零成本']['oos_total']-V['10仓·月调']['oos_total'])*100:.0f}pp；
换手 {V['10仓·月调']['avg_turnover']*100:.0f}%/月 × 0.4% ≈ 3.7%/年，而 IC 0.10 的毛边只有 3% 上下 —— 结构上就是负的。
30 仓把单只权重从 10% 压到 3.3%，月度收益不再由 2–3 只票决定，同样因子才留得下钱。
同期沪深300 {p(J['portfolio']['bench']['沪深300']['total'],1)}。</div>

<h2>四、与旧突破信号交叉（最硬的一条）</h2>
<table><thead><tr><th>突破信号所在 combo 档</th><th class=num>信号数</th>
<th class=num>其后 20 日超额</th><th class=num>聚类 t</th></tr></thead>
<tbody>{''.join(rows_cross)}</tbody></table>
<div class="sub">按 signal_date 聚类（同一天多条信号算一组）。Q1 显著为负（t = {J['cross_breakout']['1']['t']:+.2f}），
Q2–Q5 全为正、Q5 最高但不显著（t = {J['cross_breakout']['5']['t']:+.2f}）。
结论：两个正交范式同向——横截面分低的突破信号确实是负 alpha；但「高分更值钱」只到边缘证据，
而且组间差异只有 1.37 个百分点，不足以支撑「只做 Q4/Q5」。</div>

<h2>五、把过滤网接进引擎：能不能多赚钱</h2>
<table><thead><tr><th>信号池</th><th class=num>信号</th><th class=num>样本外</th>
<th class=num>年化</th><th class=num>回撤</th><th class=num>Sharpe</th><th class=num>成交</th>
<th class=num>随机对照中位</th><th class=num>随机 5–95%</th></tr></thead>
<tbody>{''.join(rows_eng)}</tbody></table>
<div class="sub">同一 v3 口径（10 仓等额、仅强市、25%/120 日出场），只改「让哪些信号进池」。
「随机对照」＝从同一信号池随机删/留同样条数跑 30 次，用来判断过滤到底是真信号还是运气。
剔除 Q1 的 {p([e for e in C if e['name'].startswith('drop_q1')][0]['total'],1)}
刚过随机 95 分位（{p(CT['drop_q1']['p95'],1)}），属于「边缘证据」；
「只留 Q4+Q5」的 {p([e for e in C if e['name'].startswith('only_q45')][0]['total'],1)}
落在随机区间（{p(CT['only_q45']['p5'],1)} ~ {p(CT['only_q45']['p95'],1)}）内，等于没证据。</div>

<div class="box warn"><b>顺带发现的容量问题（比新方向更要紧）</b>：
把「月末截面有 combo 分」当流动性/上市时长门槛（≥250 根 K 线、20 日均额 ≥ 5000 万）后，
v3 本体在同池重跑的样本外只有 {p([e for e in C if e['name'].startswith('baseline')][0]['total'],1)}
（官方口径 {p(0.193,1)}，含全部 20430 条信号）；被门槛挡掉的 5850 条（次新 + 当日流动性不足）
单独跑是 {p([e for e in C if e['name'].startswith('excluded')][0]['total'],1)}。
也就是说：v3 的 +19.3% 在「更严格的可成交池」里站不住（同池只剩 +1.3%），
二者差多少由路径决定、但方向明确——样本外收益对这批票的参与敏感，实盘容量结论要下调，
需要按 20 日均额 5000 万 / 1 亿 / 2 亿三档分层再核。</div>

<h2>六、所以下一步做什么</h2>
<ul>
<li><b>做</b>：把「剔除 combo 最低 20% 分位」作为面板上的<b>影子标记</b>（只标注、不改主口径），
先纸面跟踪 3–6 个月，看它是否真的躲掉了 Q1 那批亏损单。</li>
<li><b>做</b>：v3 容量分层核查（按 20 日均额 5000 万 / 1 亿 / 2 亿三档各跑一遍样本外），
这决定 5 万资金能不能真按 10 仓做。</li>
<li><b>暂缓</b>：把横截面当独立策略上钱。30 仓季调的点估计好看但 t &lt; 1，
且时点平移能差 4 倍，必须先有纸面 track record。</li>
<li><b>不做</b>：「只做 Q4/Q5 突破信号」——随机对照证明不可区分；
「10 仓月频横截面」——成本结构性吃光毛边。</li>
</ul>
<div class="sub">重现：<code>python backtest/alt_scans.py</code>（约 90 秒）→ <code>python backtest/cross_check.py</code>（约 13 秒）。
数字源：alt_scans.json / cross_check.json / cross_check_ctrl.json。</div>
</body></html>"""

(ROOT / "alt_report.html").write_text(html, encoding="utf-8")
print("→ alt_report.html")
