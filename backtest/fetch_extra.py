#!/usr/bin/env python3
"""补齐面板需要的辅助数据（东方财富公开接口，无需鉴权）：

1. 行业：clist 的 f100 字段，写入 stocks.industry
2. 不复权收盘：最近约 120 个交易日的 fqt=0 收盘，写入 bars_raw，用于面板展示真实价格
3. 指数：深证成指（399001）、创业板指（399006）写入 bars（IDX_ 前缀）

用法：
    ../.venv-bt/bin/python fetch_extra.py            # 全部刷新
    ../.venv-bt/bin/python fetch_extra.py --industry # 只刷行业
"""
import json
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "market.db"

LIST_URL = ("https://push2.eastmoney.com/api/qt/clist/get?pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2"
            "&fid=f12&po=0&fs=m:1+t:2,m:0+t:6&fields=f12,f13,f14,f26,f20,f100")
KLINE_URL = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&klt=101&fqt={fqt}"
             "&beg={beg}&end={end}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
RAW_BEG = (datetime.now() - timedelta(days=240)).strftime("%Y%m%d")
END = datetime.now().strftime("%Y%m%d")


def get(url, timeout=20, retries=3):
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception:  # noqa: BLE001
            if i == retries:
                raise
            time.sleep(0.4 * (i + 1))


def ensure_schema(con):
    cols = {r[1] for r in con.execute("pragma table_info(stocks)")}
    if "industry" not in cols:
        con.execute("alter table stocks add column industry text")
    con.executescript("""
        create table if not exists bars_raw(code text, date text, close real,
                                           primary key(code, date));
        create index if not exists idx_raw_date on bars_raw(date);
    """)
    con.commit()


def fetch_industries(con):
    rows, total, pn = [], None, 1
    while total is None or len(rows) < total:
        d = get(LIST_URL.format(pn=pn))
        p = d["data"]
        total = p["total"]
        rows.extend(p["diff"])
        pn += 1
    con.executemany("update stocks set industry=? where code=?",
                    [(r.get("f100") or "", r["f12"]) for r in rows])
    con.commit()
    print(f"行业已更新：{len(rows)} 条", flush=True)


def fetch_raw(code, market):
    url = KLINE_URL.format(secid=f"{market}.{code}", fqt=0, beg=RAW_BEG, end=END)
    kl = (get(url, timeout=25).get("data") or {}).get("klines") or []
    return [(line.split(",")[0], float(line.split(",")[2])) for line in kl]


def fetch_raw_all(con):
    stocks = list(con.execute("select code, market from stocks"))
    done, failed = 0, []
    t0 = time.time()

    def work(s):
        try:
            return s[0], fetch_raw(s[0], s[1]), None
        except Exception as e:  # noqa: BLE001
            return s[0], [], str(e)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(work, s) for s in stocks]
        for f in as_completed(futs):
            code, rows, err = f.result()
            done += 1
            if err or not rows:
                failed.append(code)
            else:
                con.executemany("insert or replace into bars_raw values (?,?,?)",
                                [(code, d, c) for d, c in rows])
            if done % 500 == 0:
                con.commit()
                print(f"  不复权收盘 {done}/{len(stocks)}，耗时 {time.time()-t0:.0f}s，失败 {len(failed)}", flush=True)
    con.commit()
    print(f"不复权收盘完成：失败 {len(failed)} 只", flush=True)


def fetch_indices(con):
    for secid, name in (("0.399001", "深证成指"), ("0.399006", "创业板指")):
        url = KLINE_URL.format(secid=secid, fqt=1, beg="20180101", end=END)
        kl = (get(url).get("data") or {}).get("klines") or []
        rows = []
        for line in kl:
            p = line.split(",")
            rows.append(("IDX_" + name, p[0], float(p[1]), float(p[2]), float(p[3]),
                         float(p[4]), float(p[5]), float(p[6])))
        con.executemany("insert or replace into bars values (?,?,?,?,?,?,?,?)", rows)
        print(f"{name}：{len(rows)} 行", flush=True)
    con.commit()


def main():
    con = sqlite3.connect(DB)
    ensure_schema(con)
    if "--industry" in sys.argv:
        fetch_industries(con)
        return
    fetch_industries(con)
    fetch_raw_all(con)
    fetch_indices(con)
    con.close()
    print("完成", flush=True)


if __name__ == "__main__":
    main()
