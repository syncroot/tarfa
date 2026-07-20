#!/usr/bin/env python3
"""Tarfa cockpit - chat + memory-movement telemetry for the fused-kernel engine (port 8089)."""
import json, time, subprocess, os
import requests
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Static
from textual import work, on

URL = "http://127.0.0.1:8089"
MODEL = "qwen3.5-122b-tarfa"
NVME_CEIL, PCIE_CEIL, FLOOR_MS = 3.4, 15.7, 210

TELEM = {"util": 0, "used": 0, "total": 16380, "temp": 0, "pw": 0, "pwmax": 165,
         "toks": 0.0, "state": "idle", "big": "…", "gbps": 0.0, "mbs": 0.0, "eps": 0, "hit": 0.0, "resident": 0}

def nvsmi():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0]
        p = [x.strip() for x in o.split(",")]
        return dict(util=int(float(p[0])), used=int(float(p[1])), total=int(float(p[2])), temp=int(float(p[3])), pw=float(p[4]), pwmax=float(p[5]))
    except Exception:
        return {}

def get(path):
    try: return requests.get(URL + path, timeout=3).json()
    except Exception: return {}

def bar(frac, w=18, color="#6FA3A0"):
    frac = max(0.0, min(1.0, frac)); n = int(round(frac * w))
    return f"[{color}]" + "█" * n + "[/][#26302f]" + "·" * (w - n) + "[/]"

class Tarfa(App):
    TITLE = "TARFA"
    CSS = """
    Screen { background: #0f1413; }
    #chat { width: 1fr; }
    #history { height: 1fr; border: round #2C8C7A; background: #121a18; padding: 0 1; }
    #inp { border: round #26302f; background: #161e1c; color: #EAF4F1; margin-top: 1; }
    #tele { width: 48; border: round #2C8C7A; background: #121a18; padding: 1 2; }
    .you { color: #EAF4F1; margin-top: 1; }
    .bot { color: #bfe0d8; margin-bottom: 1; }
    """
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="chat"):
                yield VerticalScroll(id="history")
                yield Input(placeholder="Ask Tarfa…  ·  /toggle  ·  /quit", id="inp")
            yield Static("", id="tele")

    def on_mount(self):
        self.cur = None
        self.query_one("#history").mount(Static("[#2C8C7A]░▒▓ TARFA ▓▒░[/]  [dim]exact-first 122B runtime · ask away[/]"))
        self.query_one("#inp").focus(); self.poll()

    @work(thread=True)
    def poll(self):
        while True:
            g = nvsmi(); s = get("/status"); io = get("/iostat")
            if g: TELEM.update(g)
            if s:
                TELEM["big"] = s.get("big_model") or "-"
                TELEM["hit"] = s.get("cache_hit_rate", 0.0); TELEM["resident"] = s.get("cached_experts", 0)
                TELEM["state"] = "streaming" if s.get("serving") else ("idle" if not self.cur else TELEM["state"])
                if not s.get("serving") and not self.cur: TELEM["toks"] = 0.0
            else:
                TELEM["state"] = "asleep"; TELEM["big"] = "-"; TELEM["toks"] = 0.0
            if io:
                TELEM["gbps"] = io.get("read_GBps", 0.0); TELEM["mbs"] = io.get("read_b", 0) / 1e6; TELEM["eps"] = io.get("experts", 0)
            else:
                TELEM["gbps"] = TELEM["mbs"] = 0.0; TELEM["eps"] = 0
            self.call_from_thread(self.draw_tele); time.sleep(1)

    def draw_tele(self):
        t = TELEM
        st = {"streaming": "[#7ec27e]● streaming[/]", "asleep": "[#B5432B]● asleep[/]"}.get(t["state"], "[#c9a06f]○ idle[/]")
        toks = t["toks"]; ms = (1000.0 / toks) if toks > 0 else 0.0
        ratio = (ms / FLOOR_MS) if ms > 0 else 0.0
        moving = t["mbs"] > 1.0
        busline = (f"[b #EAF4F1]{t['gbps']:.2f}[/] GB/s   [dim]{t['eps']} exp/s[/]" if moving else "[#5f6b68]quiet[/]")
        util_note = "[#B5432B]idle · read-bound[/]" if t["util"] < 15 else "[dim]busy[/]"
        txt = (
            f"[b #2C8C7A]TARFA[/] [dim]· fused-kernel engine[/]   {st}\n"
            f"[#9fb8b2]{t['big']}[/]\n[#6FA3A0]int4 dequant in SRAM · GEMV[/]\n\n"

            f"[#2C8C7A]━━ THE FUSED KERNEL ━━━━━━━[/]\n"
            f"  MoE compute  [b #7ec27e]~10 ms[/]/token\n"
            f"  [dim]was ~450ms (per-expert loop)[/]\n"
            f"  [#6FA3A0]✓ correct · dequant never hits VRAM[/]\n\n"

            f"[#2C8C7A]━━ THE BUS  (now the wall) ━━[/]\n"
            f"  {bar(t['gbps']/PCIE_CEIL, 18, '#5DCAA5')}\n"
            f"  {busline}\n"
            f"  [dim]ceiling {NVME_CEIL} drive · {PCIE_CEIL} PCIe x8[/]\n\n"

            f"[#2C8C7A]━━ SPEED ━━━━━━━━━━━━━━━━━━[/]\n"
            f"  [b #EAF4F1]{toks:.2f}[/] tok/s   [dim]{ms:.0f} ms/tok[/]\n"
            f"  [dim]floor {FLOOR_MS}ms = 4.7 tok/s[/]\n"
            f"  {bar(1/ratio if ratio>0 else 0, 18, '#5DCAA5')} [dim]{ratio:.1f}× over floor[/]\n\n"

            f"[#2C8C7A]━━ CACHE ━━━━━━━━━━━━━━━━━━[/]\n"
            f"  hit [b #EAF4F1]{t['hit']*100:.0f}%[/] · {t['resident']} experts resident\n\n"

            f"[#9fb8b2]GPU[/]  {bar(t['util']/100,18,'#5DCAA5')} [b]{t['util']}%[/]  {util_note}\n"
            f"[#9fb8b2]VRAM[/] {bar(t['used']/max(t['total'],1),18,'#5DCAA5')} {t['used']/1024:.1f}/{t['total']/1024:.0f}G\n"
            f"[#9fb8b2]pwr[/]  {t['pw']:.0f}/{t['pwmax']:.0f}W · {t['temp']}°C\n"
        )
        self.query_one("#tele", Static).update(txt)

    def note(self, text):
        h = self.query_one("#history"); h.mount(Static(f"[#c9a06f]{text}[/]")); h.scroll_end()

    @work(thread=True)
    def do_toggle(self):
        tarfa = os.path.expanduser("~/.local/bin/tarfa")
        if get("/status"):
            self.call_from_thread(self.note, "→ stopping Tarfa…")
            subprocess.run([tarfa, "stop"], capture_output=True, timeout=30)
            self.call_from_thread(self.note, "○ Tarfa stopped - GPU freed")
        else:
            self.call_from_thread(self.note, "→ starting Tarfa… (~90s, streaming residency)")
            subprocess.run([tarfa, "start"], capture_output=True, timeout=220)
            self.call_from_thread(self.note, "● Tarfa up" if get("/status") else "✗ start failed - see `tarfa logs`")

    @on(Input.Submitted, "#inp")
    async def submit(self, e: Input.Submitted):
        msg = e.value.strip(); e.input.value = ""
        if not msg: return
        if msg in ("/quit", "/q", "exit"): self.exit(); return
        h = self.query_one("#history")
        if msg in ("/toggle", "/sleep", "/wake"): self.do_toggle(); return
        if TELEM.get("state") == "asleep":
            await h.mount(Static("[#c9a06f]Tarfa is down - type [b]/toggle[/] to start it (~90s).[/]")); h.scroll_end(); return
        await h.mount(Static(f"[b]you[/]  {msg}", classes="you"))
        self.cur = Static("[#2C8C7A]tarfa[/]  [dim]…[/]", classes="bot")
        await h.mount(self.cur); h.scroll_end(); self.stream(msg)

    @work(thread=True)
    def stream(self, msg):
        cur = self.cur; t0 = time.time(); n = 0; acc = ""
        try:
            r = requests.post(URL + "/v1/chat/completions", json={"model": MODEL, "messages": [{"role": "user", "content": msg}],
                              "max_tokens": 8192, "stream": True}, stream=True, timeout=3600)
            for line in r.iter_lines():
                if not line: continue
                s = line.decode().removeprefix("data: ")
                if s == "[DONE]": break
                try: c = json.loads(s)["choices"][0]["delta"].get("content", "")
                except Exception: continue
                if c:
                    n += 1; acc += c; el = time.time() - t0
                    TELEM["toks"] = n / el if el > 0 else 0.0
                    self.call_from_thread(cur.update, f"[#2C8C7A]tarfa[/]  {acc}")
            el = time.time() - t0
            self.call_from_thread(cur.update, f"[#2C8C7A]tarfa[/]  {acc}\n[#5f6b68]· {n} tok · {n/max(el,1e-9):.2f} tok/s · {el:.0f}s · fused MoE[/]")
        except Exception as ex:
            asleep = "Connection refused" in str(ex) or "Max retries" in str(ex)
            txt = "Tarfa is down - type /toggle to start it" if asleep else f"error: {ex}"
            self.call_from_thread(cur.update, f"[#B5432B]{txt}[/]")
        self.cur = None; TELEM["toks"] = 0.0
        self.call_from_thread(lambda: self.query_one("#history").scroll_end())

if __name__ == "__main__":
    Tarfa().run()
