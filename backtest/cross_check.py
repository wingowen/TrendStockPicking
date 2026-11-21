"""引擎层交叉验证：把横截面 combo 分位当作突破信号的过滤条件，看 v3 组合净值怎么变。

口径全部继承 v3（V3_SPEC、score_v3、成本 0.4%、10 仓等额、仅强市），
只改「让哪些信号进池」：
  baseline  全部突破信号（v3 官方口径）
  drop_q1   剔除 combo 最低 20% 分位的信号
  only_q45  只保留 combo 最高 40% 分位的信号
  only_q5   只保留最高 20% 分位
样本外从 2022-01-01 起。
"""
import csv
import numpy as np

import engine_v2 as E

OOS_START = "2022-01-01"
ROOT = E.Path(__file__).resolve().parent


def load_qmap():
    p = ROOT / "combo_map.csv"
    if not p.exists():
        raise SystemExit("缺少 combo_map.csv，先跑 alt_scans.py")
    with open(p, encoding="utf-8") as f:
        return {(r["code"], r["signal_date"]): int(r["q"]) for r in csv.DictReader(f)}


def main():
    rows = E.load_features()
    dates, _bench = E.load_calendar()
    qmap = load_qmap()
    for r in rows:
        r["cq"] = qmap.get((r["code"], r["signal_date"]))

    groups = {
        "baseline 全部信号": [r for r in rows if r["cq"] is not None],
        "drop_q1 剔除最低20%": [r for r in rows if r["cq"] is not None and r["cq"] >= 2],
        "only_q45 只留最高40%": [r for r in rows if r["cq"] is not None and r["cq"] >= 4],
        "only_q5 只留最高20%": [r for r in rows if r["cq"] is not None and r["cq"] == 5],
        "excluded 无分位(次新/低流动性)": [r for r in rows if r["cq"] is None],
    }
    allrows = rows                             # 用全量建价量表（含无分位的次新票）
    closes, lot = E.prep(dates, allrows)
    out = []
    for name, rs in groups.items():
        if not rs:
            continue
        rank = {(r["code"], r["entry_date"]): E.score_v3(r) for r in rs}
        eq, m = E.run_cfg(rs, dates, closes, lot, E.V3_SPEC, rank=rank, start=OOS_START)
        out.append((name, len(rs), m))
        print(f"{name:22s} 信号 {len(rs):5d}  OOS {m['total']*100:+6.1f}%  "
              f"年化 {m['cagr']*100:+5.2f}%  回撤 {m['mdd']*100:+6.1f}%  "
              f"Sharpe {m['sharpe']:.2f}  成交 {m['fills']}")

    # 随机对照：drop_q1 等价于随机删 3935 条；only_q45 等价于随机留 3932 条
    def rand_run(rs, n=30, seed=3):
        rng = np.random.default_rng(seed)
        tot = []
        for _ in range(n):
            sub = list(rng.choice(len(rs), size=len(rs), replace=False))
            sel = [rs[i] for i in sub]
            rank = {(r["code"], r["entry_date"]): E.score_v3(r) for r in sel}
            _eq, m = E.run_cfg(sel, dates, closes, lot, E.V3_SPEC, rank=rank, start=OOS_START)
            tot.append(m["total"])
        return np.array(tot)

    ctrl = {}
    base = groups["baseline 全部信号"]
    keep_n = len(groups["drop_q1 剔除最低20%"])
    rng = np.random.default_rng(11)
    drop_ctrl = []
    for _ in range(30):
        sel = [base[i] for i in rng.choice(len(base), size=keep_n, replace=False)]
        rank = {(r["code"], r["entry_date"]): E.score_v3(r) for r in sel}
        _eq, m = E.run_cfg(sel, dates, closes, lot, E.V3_SPEC, rank=rank, start=OOS_START)
        drop_ctrl.append(m["total"])
    ctrl["drop_q1"] = {"random_median": float(np.median(drop_ctrl)),
                       "p95": float(np.percentile(drop_ctrl, 95)),
                       "p5": float(np.percentile(drop_ctrl, 5))}
    keep4 = len(groups["only_q45 只留最高40%"])
    q45_ctrl = []
    for _ in range(30):
        sel = [base[i] for i in rng.choice(len(base), size=keep4, replace=False)]
        rank = {(r["code"], r["entry_date"]): E.score_v3(r) for r in sel}
        _eq, m = E.run_cfg(sel, dates, closes, lot, E.V3_SPEC, rank=rank, start=OOS_START)
        q45_ctrl.append(m["total"])
    ctrl["only_q45"] = {"random_median": float(np.median(q45_ctrl)),
                        "p95": float(np.percentile(q45_ctrl, 95)),
                        "p5": float(np.percentile(q45_ctrl, 5))}
    print("随机对照 drop_q1：中位 {:+.1f}%（5–95 {:+.1f}% ~ {:+.1f}%）".format(
        ctrl["drop_q1"]["random_median"]*100, ctrl["drop_q1"]["p5"]*100, ctrl["drop_q1"]["p95"]*100))
    print("随机对照 only_q45：中位 {:+.1f}%（5–95 {:+.1f}% ~ {:+.1f}%）".format(
        ctrl["only_q45"]["random_median"]*100, ctrl["only_q45"]["p5"]*100, ctrl["only_q45"]["p95"]*100))

    import json
    with open(ROOT / "cross_check.json", "w", encoding="utf-8") as f:
        json.dump([{"name": n, "n_all": s,
                    **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in m.items()}} for n, s, m in out],
                  f, ensure_ascii=False, indent=1)
    with open(ROOT / "cross_check_ctrl.json", "w", encoding="utf-8") as f:
        json.dump(ctrl, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
