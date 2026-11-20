#!/usr/bin/env python3
"""拉取 A 股主板全市场日线（东方财富公开行情接口），存入 SQLite，供回测使用。

- 股票池：沪深主板（600/601/603/605/000/001/002/003），剔除当前 ST / 退市名
- 行情：日线，后复权（fqt=2），2018-01-01 起，用于 20 日均线等暖机
- 基准：沪深300（000300）与上证指数（000001）

不依赖任何付费/鉴权接口。
"""
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "market.db"
START = "20180101"
END = datetime.now().strftime("%Y%m%d")

LIST_URL = ("https://push2.eastmoney.com/api/qt/clist/get?pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2"
            "&fid=f12&po=0&fs=m:1+t:2,m:0+t:6&fields=f12,f13,f14,f26,f20")
KLINE_URL = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&klt=101&fqt={fqt}"
             "&beg={beg}&end={end}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57")

MAIN_PREFIX = ("600", "601", "603", "605", "000", "001", "002", "003")
EXCLUDE_PREFIX = ("688", "689", "300", "301", "8", "4", "920")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"


def get(url, timeout=20, retries=3):
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            if i == retries:
                raise
            time.sleep(0.4 * (i + 1))


def fetch_universe():
    rows, total, pn = [], None, 1
    while total is None or len(rows) < total:
        d = get(LIST_URL.format(pn=pn))
        payload = d["data"]
        total = payload["total"]
        rows.extend(payload["diff"])
        pn += 1
    print(f"接口返回股票数：{len(rows)} / {total}")
    out = []
    for r in rows:
        code, name = r["f12"], r["f14"]
        if not code.startswith(MAIN_PREFIX) or code.startswith(EXCLUDE_PREFIX):
            continue
        if "ST" in name.upper() or "退" in name:
            continue
        out.append({"code": code, "name": name, "market": int(r["f13"]),
                    "list_date": str(r.get("f26") or ""), "mktcap": r.get("f20")})
    print(f"主板可交易（剔除 ST/退市）：{len(out)}")
    return out


def fetch_kline(code, market, fqt=2):
    secid = f"{market}.{code}"
    url = KLINE_URL.format(secid=secid, fqt=fqt, beg=START, end=END)
    d = get(url, timeout=25)
    data = d.get("data") or {}
    klines = data.get("klines") or []
    out = []
    for line in klines:
        p = line.split(",")
        # 日期, 开, 收, 高, 低, 成交量(手), 成交额
        out.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                    float(p[5]), float(p[6])))
    return out


def init_db():
    con = sqlite3.connect(DB)
    con.executescript("""
        drop table if exists bars;
        drop table if exists stocks;
        create table stocks(code text primary key, name text, market int,
                            list_date text, mktcap real);
        create table bars(code text, date text, open real, close real, high real,
                          low real, volume real, amount real,
                          primary key(code, date));
        create index idx_bars_date on bars(date);
    """)
    return con


def main():
    force = "--force" in sys.argv
    if DB.exists() and not force:
        con = sqlite3.connect(DB)
        n = con.execute("select count(*) from bars").fetchone()[0]
        print(f"已存在 {DB.name}（{n} 行），如需重抓请加 --force")
        return con
    con = init_db()
    uni = fetch_universe()
    seen = {}
    for s in uni:
        seen[s["code"]] = s
    uni = list(seen.values())
    print(f"去重后：{len(uni)}")
    con.executemany("insert or replace into stocks values (?,?,?,?,?)",
                    [(s["code"], s["name"], s["market"], s["list_date"], s["mktcap"]) for s in uni])
    con.commit()

    total = len(uni)
    done = 0
    failed = []

    def work(s):
        try:
            bars = fetch_kline(s["code"], s["market"])
            return s["code"], bars, None
        except Exception as e:  # noqa: BLE001
            return s["code"], [], str(e)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(work, s) for s in uni]
        for f in as_completed(futs):
            code, bars, err = f.result()
            done += 1
            if err or not bars:
                failed.append((code, err or "空数据"))
            else:
                con.executemany(
                    "insert or replace into bars values (?,?,?,?,?,?,?,?)",
                    [(code, b[0], b[1], b[2], b[3], b[4], b[5], b[6]) for b in bars])
            if done % 300 == 0:
                con.commit()
                print(f"  {done}/{total} 只，耗时 {time.time()-t0:.0f}s，失败 {len(failed)}", flush=True)
    con.commit()

    # 指数：沪深300、上证指数（不复权）
    for secid, name in (("1.000300", "沪深300"), ("1.000001", "上证指数")):
        url = KLINE_URL.format(secid=secid, fqt=1, beg=START, end=END)
        d = get(url)
        kl = d["data"]["klines"]
        rows = []
        for line in kl:
            p = line.split(",")
            rows.append(("IDX_" + name, p[0], float(p[1]), float(p[2]), float(p[3]),
                         float(p[4]), float(p[5]), float(p[6])))
        con.executemany("insert or replace into bars values (?,?,?,?,?,?,?,?)", rows)
        print(f"{name}：{len(rows)} 行")

    con.commit()
    n_bars = con.execute("select count(*) from bars").fetchone()[0]
    n_stock = con.execute("select count(distinct code) from stocks").fetchone()[0]
    print(f"完成：{n_stock} 只股票，{n_bars} 行日线，失败 {len(failed)} 只")
    if failed:
        print("失败样例：", failed[:5])
    con.close()


if __name__ == "__main__":
    main()
