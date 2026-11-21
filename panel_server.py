#!/usr/bin/env python3
"""面板本地刷新服务：只在收盘(15:05)后允许重建今日清单。

面板页面(file://)里的「刷新今日清单」按钮会请求本服务；
本服务负责做时间守卫，并执行 ./run_daily.sh（抓数据→补板块资金→重生成面板）。
启动一次即可常驻：双击 start_panel_server.command。
"""
import datetime
import http.server
import json
import os
import sqlite3
import subprocess
import threading
import time

# 收盘守卫必须用北京时间；不依赖启动者的 shell 时区
os.environ.setdefault("TZ", "Asia/Shanghai")
if hasattr(time, "tzset"):
    time.tzset()

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8765
_lock = threading.Lock()


def now():
    return datetime.datetime.now()


def is_after_close():
    return (now().hour, now().minute) >= (15, 5)


def db_latest():
    try:
        con = sqlite3.connect(os.path.join(ROOT, "backtest", "market.db"))
        v = con.execute("select max(date) from bars").fetchone()[0]
        con.close()
        return v
    except Exception:
        return None


def _cors(handler):
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class Handler(http.server.BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        _cors(self)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/ping"):
            self._json({
                "ok": True,
                "now": now().strftime("%Y-%m-%d %H:%M:%S"),
                "after_close": is_after_close(),
                "db_latest": db_latest(),
            })
        else:
            self._json({"ok": False, "msg": "not found"}, 404)

    def do_POST(self):
        if self.path.startswith("/rebuild"):
            if not is_after_close():
                self._json({
                    "ok": False,
                    "msg": f"未到收盘（当前 {now().strftime('%H:%M')}），15:05 后才能刷新今日清单。",
                })
                return
            got = _lock.acquire(blocking=False)
            if not got:
                self._json({"ok": False, "msg": "正在刷新中，请稍候（约 4–5 分钟）…"})
                return
            try:
                proc = subprocess.run(
                    ["bash", "run_daily.sh"], cwd=ROOT,
                    capture_output=True, text=True, timeout=900,
                )
                tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-10:])
                ok = proc.returncode == 0
                latest = db_latest()
                extra = ""
                if ok and latest and latest != now().strftime("%Y-%m-%d"):
                    extra = "（今日无新交易日数据，已按最新 %s 生成）" % latest
                self._json({
                    "ok": ok,
                    "msg": (tail or ("刷新完成" if ok else "刷新失败")) + extra,
                    "latest": latest,
                })
            except subprocess.TimeoutExpired:
                self._json({"ok": False, "msg": "刷新超时（超过 15 分钟），请查看面板日志或手动运行 run_daily.sh"})
            finally:
                _lock.release()
        else:
            self._json({"ok": False, "msg": "not found"}, 404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"面板刷新服务已启动：http://127.0.0.1:{PORT}  （Ctrl+C 退出）")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
