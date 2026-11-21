#!/usr/bin/env python3
"""板块真实主力净流入（东财 fflow，只有近 120 个交易日）

面板只用最近几天，所以这里用真实数据；回测（2020 起）仍用代理口径，
两者的相关性已在 flow_validate.py 里验证过（横截面 ρ≈0.47）。
"""
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "fflow_cache.json"
# 缓存最大有效天数：board_flow 数据按交易日变化，隔日必须重抓。
# 每日脚本 server_daily.sh 会直接删缓存；这里再加 TTL 兜底，防止漏清理。
CACHE_MAX_AGE_DAYS = 1
HDR = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}


def _cache_fresh(cache, max_age_days):
    """判断缓存是否仍在有效期内（generated_at 为 UTC ISO-8601 字符串）。"""
    ts = cache.get("generated_at")
    if not ts:
        return False
    try:
        t = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return False
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - t <= timedelta(days=max_age_days)


def _get(u):
    req = urllib.request.Request(u, headers=HDR)
    return json.loads(urllib.request.urlopen(req, timeout=20).read().decode())


def norm(s):
    return (s or "").replace("Ⅱ", "").replace("Ⅲ", "").strip()


def board_map():
    """{板块名(去 Ⅱ/Ⅲ): BK 代码}"""
    out, pn = {}, 1
    while pn <= 8:
        d = _get("https://push2.eastmoney.com/api/qt/clist/get?pn=%d&pz=100&po=0&np=1"
                 "&fltt=2&invt=2&fid=f12&fs=m:90+t:2&fields=f12,f14" % pn)
        dd = d.get("data")
        if not dd or not dd.get("diff"):
            break
        for x in dd["diff"]:
            out[norm(x["f14"])] = x["f12"]
        pn += 1
    return out


def board_flow(bk):
    """{日期: 主力净流入净占比(小数)，如 0.036 = 3.6%}"""
    u = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?lmt=0&klt=101"
         f"&secid=90.{bk}&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57")
    k = _get(u)["data"]["klines"]
    out = {}
    for row in k:
        p = row.split(",")
        if len(p) >= 7 and p[6] not in ("-", ""):
            try:
                out[p[0]] = float(p[6]) / 100.0
            except ValueError:
                pass
    return out


def load_real_flow(industries, use_cache=True, max_age_days=CACHE_MAX_AGE_DAYS):
    """返回 {行业名: {日期: 净占比}}，只含能按名匹配到东财板块的行业。

    max_age_days: 缓存有效期（天），0 表示强制忽略缓存重抓。
    """
    industries = {norm(i) for i in industries if i}
    if use_cache and CACHE.exists():
        try:
            c = json.loads(CACHE.read_text(encoding="utf-8"))
            if (set(c.get("want", [])) >= industries
                    and _cache_fresh(c, max_age_days)):
                return c["flow"]
        except Exception:
            pass
    bm = board_map()
    flow = {}
    for name in sorted(industries):
        bk = bm.get(name)
        if not bk:
            continue
        try:
            flow[name] = board_flow(bk)
        except Exception:
            continue
    CACHE.write_text(json.dumps({
        "want": sorted(industries),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flow": flow,
    }, ensure_ascii=False), encoding="utf-8")
    return flow
