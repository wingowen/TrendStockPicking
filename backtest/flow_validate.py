#!/usr/bin/env python3
"""验证「资金流代理口径」是否可信

东财 fflow 历史只保留 120 个交易日（个股、板块都一样，lmt 参数无效），
所以 2020 年起的回测只能用代理口径（上涨日成交额记正、下跌日记负）。
这个脚本把代理口径和东财真实「主力净流入净占比」在最近 120 个交易日上做对照：
  - 符号一致率（正负方向是否一致）
  - 数值相关系数（逐日横截面 corr 的均值）
输出 flow_check.json / flow_check.csv
"""
import csv
import json
import sqlite3
import urllib.request
from pathlib import Path

import numpy as np

from run_backtest import DB, load_data
from sector_trend import build_industry_indices

OUT_JSON = Path(__file__).resolve().parent / "flow_check.json"
OUT_CSV = Path(__file__).resolve().parent / "flow_check.csv"
HDR = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}


def get(u):
    req = urllib.request.Request(u, headers=HDR)
    return json.loads(urllib.request.urlopen(req, timeout=20).read().decode())


def norm(s):
    return (s or "").replace("Ⅱ", "").replace("Ⅲ", "").replace("Ｉ", "I").strip()


def board_list():
    out, pn = {}, 1
    while True:
        d = get("https://push2.eastmoney.com/api/qt/clist/get?pn=%d&pz=100&po=1&np=1"
                "&fltt=2&invt=2&fid=f12&po=0&fs=m:90+t:2&fields=f12,f14" % pn)
        dd = d.get("data")
        if not dd or not dd.get("diff"):
            break
        diff = dd["diff"]
        for x in diff:
            out[norm(x["f14"])] = x["f12"]
        pn += 1
        if pn > 10:
            break
    return out


def board_flow(bk):
    u = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?lmt=0&klt=101"
         f"&secid=90.{bk}&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57")
    k = get(u)["data"]["klines"]
    out = {}
    for row in k:
        p = row.split(",")
        # date, 主力净额, 小单, 中单, 大单, 超大单, 主力净流入净占比(%)
        if len(p) >= 7 and p[6] not in ("-", ""):
            out[p[0]] = float(p[6]) / 100.0
    return out


def main():
    stocks, bars, idx = load_data()
    con = sqlite3.connect(DB)
    industry = {c: i for c, i in con.execute("select code, industry from stocks")}
    con.close()

    calendar = idx["上证指数"]["dates"]
    ind_series = build_industry_indices(bars, industry, calendar)

    boards = board_list()
    mapped = {k: v for k, v in boards.items() if k in ind_series}
    print(f"[map] 我们的行业 {len(ind_series)} 个，东财板块 {len(boards)} 个，按名匹配 {len(mapped)} 个")

    real = {}
    for name, bk in mapped.items():
        try:
            real[name] = board_flow(bk)
        except Exception as e:
            print("  fail", name, bk, e)
    dates = sorted({d for f in real.values() for d in f})
    print(f"[real] 真实资金流覆盖 {len(real)} 个板块 / {len(dates)} 个交易日 "
          f"{dates[0]}→{dates[-1]}")

    rows = []
    for name, o in ind_series.items():
        if name not in real:
            continue
        ds, flow = o["dates"], o["flow"]
        pos = {d: i for i, d in enumerate(ds)}
        for p in range(5, len(ds)):
            d = ds[p]
            if d not in real[name]:
                continue
            win = ds[p - 4:p + 1]
            if any(x not in real[name] for x in win):
                continue
            proxy = np.nanmean(flow[p - 4:p + 1])
            if np.isnan(proxy):
                continue
            r = float(np.mean([real[name][x] for x in win]))
            rows.append({"date": d, "industry": name, "proxy": float(proxy), "real": r})

    if not rows:
        print("没有可比样本")
        return
    p = np.array([r["proxy"] for r in rows])
    q = np.array([r["real"] for r in rows])
    agree = float(np.mean(np.sign(p) == np.sign(q)))
    corr = float(np.corrcoef(p, q)[0, 1])

    # 逐日横截面相关
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append((r["proxy"], r["real"]))
    xcorr = []
    for d, v in by_date.items():
        if len(v) < 20:
            continue
        a = np.array([x[0] for x in v])
        b = np.array([x[1] for x in v])
        if a.std() > 0 and b.std() > 0:
            xcorr.append(float(np.corrcoef(a, b)[0, 1]))

    last = max(by_date)
    tail = by_date[last]
    tail_agree = float(np.mean([np.sign(a) == np.sign(b) for a, b in tail]))

    out = {"n_pairs": len(rows), "sectors": len(real), "dates": len(by_date),
           "date_from": dates[0], "date_to": dates[-1],
           "sign_agreement": round(agree, 4), "pearson": round(corr, 4),
           "xcorr_mean": round(float(np.mean(xcorr)), 4) if xcorr else None,
           "xcorr_min": round(float(np.min(xcorr)), 4) if xcorr else None,
           "last_date": last, "last_agree": round(tail_agree, 4)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
