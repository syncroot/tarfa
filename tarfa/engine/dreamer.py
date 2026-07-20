# -*- coding: utf-8 -*-
"""Dreamer — idle-time expert mapper. The 0.8B drafter invents probes, runs them
through the resident 122B, and accumulates a persistent atlas of which experts fire
for which kind of input. Runs ONLY when the engine is idle; yields instantly to any
real request by sharing the serving LOCK. Never raises into the serving path."""
import os, json, time, threading, collections, math, torch
from transformers.cache_utils import DynamicCache

ATLAS = os.path.expanduser("~/airllm-test/runner/atlas_live.json")
GRACE = 20.0       # seconds of idle before probing starts
POLL = 3.0         # how often the loop looks for an idle window
SAVE_EVERY = 8     # recompute + persist the derived atlas every N probes
GEN_MAX = 48       # cap on 0.8B probe length

# name -> (generation prompt for the 0.8B, [seed probes used as fallback / for cold start])
CATS = {
 "english":  ("Write one short, vivid English sentence.", ["The harbor was quiet at dusk.", "Rain fell on the roof all night."]),
 "french":   ("Ecris une phrase courte en francais.", ["Le chat dort sur le canape.", "Il fait tres beau a Paris."]),
 "spanish":  ("Escribe una frase corta en espanol.", ["El gato duerme en el sofa.", "Hace muy buen tiempo hoy."]),
 "german":   ("Schreibe einen kurzen deutschen Satz.", ["Die Katze schlaft auf dem Sofa.", "Heute ist das Wetter schon."]),
 "italian":  ("Scrivi una breve frase in italiano.", ["Il gatto dorme sul divano.", "Oggi il tempo e bello."]),
 "chinese":  ("写一个简短的中文句子。", ["今天天气很好。", "孩子们在花园里玩耍。"]),
 "japanese": ("短い日本語の文を書いてください。", ["今日はいい天気です。", "猫がソファで寝ています。"]),
 "arabic":   ("اكتب جملة قصيرة بالعربية.", ["القطة تنام على الأريكة.", "الطقس جميل اليوم."]),
 "russian":  ("Напишите короткое предложение на русском.", ["Кошка спит на диване.", "Сегодня хорошая погода."]),
 "code":     ("Write one short Python statement.", ["for i in range(10):\n    print(i)", "x = [n*n for n in range(5)]"]),
 "sql":      ("Write one short SQL query.", ["SELECT name FROM users WHERE age > 30;", "UPDATE t SET x = 1 WHERE id = 5;"]),
 "math":     ("Write one short math statement.", ["Solve for x: 3x + 7 = 22.", "The integral of x^2 is x^3/3."]),
 "json":     ("Write one small JSON object.", ['{"name": "Alice", "age": 30}', '{"items": [1, 2, 3]}']),
 "poetry":   ("Write one short line of poetry.", ["The moon spills silver on the sea,", "and shadows dance where willows weep,"]),
 "biology":  ("Write one short biology fact.", ["Mitochondria produce ATP in cells.", "DNA encodes genetic information."]),
 "law":      ("Write one short legal sentence.", ["The party shall indemnify the licensor.", "This agreement is governed by state law."]),
}
_names = list(CATS.keys())

_S = {"M": None, "DRAFT": None, "big_fwd": None, "LOCK": None, "BACKEND": None, "LAST_REQ": None, "EOS": None}
COUNTS = collections.defaultdict(lambda: collections.defaultdict(int))   # (cat, layer) -> {expert: count}
TOKS = collections.defaultdict(int)                                       # cat -> token count
STATE = {"state": "starting", "probes": 0, "last_cat": None, "last_probe": None,
         "cats": 0, "experts": 0, "selective": 0, "started": time.time()}
DREAM = {"logging": False, "cat": None}

def _load():
    try:
        d = json.load(open(ATLAS))
        for k, ev in d.get("counts", {}).items():
            cat, layer = k.rsplit("|", 1)
            for e, c in ev.items(): COUNTS[(cat, int(layer))][int(e)] = c
        for cat, n in d.get("toks", {}).items(): TOKS[cat] = n
        STATE["probes"] = d.get("probes", 0)
    except Exception: pass

def _wrap_one(layer, li):
    exp = getattr(getattr(layer, "mlp", None), "experts", None)
    if exp is None or getattr(exp, "_dream_wrapped", False): return
    orig = exp.forward; nexp = exp.num_experts
    def logged(h, tki, tkw):
        if DREAM["logging"] and DREAM["cat"]:
            cc = COUNTS[(DREAM["cat"], li)]
            try:
                for t in range(tki.shape[0]):
                    for e in tki[t].tolist():
                        if int(e) != nexp: cc[int(e)] += 1
            except Exception: pass
        return orig(h, tki, tkw)
    exp.forward = logged; exp._dream_wrapped = True

def _ensure_wrapped():                       # re-wrap after any free/reload rebuilt the layers
    M = _S["M"]
    for i in range(M.nL):
        try: _wrap_one(M._make_layer(i), i)
        except Exception: pass

def _next_probe():
    cat = min(_names, key=lambda c: TOKS[c])  # keep coverage balanced — feed the least-sampled category
    prompt, seeds = CATS[cat]
    M, DRAFT, EOS = _S["M"], _S["DRAFT"], _S["EOS"]
    probe = None
    try:
        pin = M.tok(M.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True),
                    return_tensors="pt").input_ids.to(M.device)
        out = DRAFT.generate(pin, max_new_tokens=GEN_MAX, do_sample=False, pad_token_id=EOS)
        txt = M.tok.decode(out[0, pin.shape[1]:], skip_special_tokens=True)
        cand = max((l.strip() for l in txt.split("\n")), key=len, default="")   # longest line = the content, not the preamble
        if len(cand) >= 12: probe = cand[:200]
    except Exception: pass
    if not probe: probe = seeds[STATE["probes"] % len(seeds)]   # fall back to a full seed sentence
    return cat, probe

@torch.no_grad()
def _probe(cat, text):
    M = _S["M"]
    ids = M.tok(text, return_tensors="pt").input_ids[0].tolist()
    if not ids: return 0
    cache = DynamicCache(config=M.cfg)
    DREAM["cat"] = cat; DREAM["logging"] = True
    try: _S["big_fwd"](ids, cache)
    finally: DREAM["logging"] = False
    TOKS[cat] += len(ids)
    return len(ids)

def _cos(a, b):
    d = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d / (na*nb + 1e-9)

def _derive():
    cats = [c for c in _names if TOKS[c] > 0]
    keys = sorted({(l, e) for c in cats for l in range(_S["M"].nL) for e in COUNTS[(c, l)]})
    rate = {(c, l, e): COUNTS[(c, l)].get(e, 0) / max(TOKS[c], 1) for (l, e) in keys for c in cats}
    sel = {}
    for c in cats:
        cand = []
        for (l, e) in keys:
            mx = rate[(c, l, e)]
            if mx < 0.08: continue
            others = [rate[(o, l, e)] for o in cats if o != c]
            mo = sum(others) / len(others) if others else 0
            if mx >= mo * 2.5: cand.append([l, e, round(mx, 3)])
        sel[c] = sorted(cand, key=lambda x: -x[2])[:8]
    catvec = {c: [rate[(c, l, e)] for (l, e) in keys] for c in cats}
    sim = [[round(_cos(catvec[a], catvec[b]), 3) for b in cats] for a in cats]
    depth = {"early": 0, "mid": 0, "late": 0}
    hmrows = []
    for c in cats:
        for (l, e, r) in sel[c]:
            depth["early" if l < 16 else "mid" if l < 32 else "late"] += 1
        for (l, e, r) in sel[c][:3]:
            hmrows.append({"label": f"L{l}#{e}", "owner": c, "rates": [round(rate[(o, l, e)], 3) for o in cats]})
    nsel = sum(len(v) for v in sel.values())
    return {"cats": cats, "keys": keys, "sel": sel, "sim": sim, "depth": depth,
            "hmrows": hmrows, "n_fired": len(keys), "n_selective": nsel}

def _save():
    try:
        d = _derive()
        STATE["cats"] = len(d["cats"]); STATE["experts"] = d["n_fired"]; STATE["selective"] = d["n_selective"]
        counts = {f"{c}|{l}": {str(e): n for e, n in COUNTS[(c, l)].items()} for (c, l) in list(COUNTS) if COUNTS[(c, l)]}
        out = {"counts": counts, "toks": dict(TOKS), "probes": STATE["probes"], "updated": time.time(),
               "cats": d["cats"], "sim": d["sim"], "sel": d["sel"], "depth": d["depth"],
               "hmrows": d["hmrows"], "n_fired": d["n_fired"], "n_selective": d["n_selective"]}
        json.dump(out, open(ATLAS + ".tmp", "w"), ensure_ascii=False)
        os.replace(ATLAS + ".tmp", ATLAS)
    except Exception: pass

def snapshot():                              # compact view for /status (cheap — reads cached STATE)
    return {"state": STATE["state"], "probes": STATE["probes"], "cats": STATE["cats"],
            "experts": STATE["experts"], "selective": STATE["selective"], "last": STATE["last_cat"]}

def full_atlas():                            # the live derived atlas, for inspection / re-rendering
    try: return json.load(open(ATLAS))
    except Exception: return {"probes": STATE["probes"], "note": "no atlas yet"}

def _loop():
    _load()
    if STATE["probes"]: _save()              # refresh derived summary from loaded counts
    time.sleep(8)                            # let the server settle before first probe
    while True:
        try:
            time.sleep(POLL)
            S = _S
            if S["BACKEND"]["cur"] != "122b": STATE["state"] = "paused · LM Studio owns GPU"; continue
            if S["LOCK"].locked(): STATE["state"] = "yielding · engine serving"; continue
            lr = S["LAST_REQ"]["ts"]
            if lr and (time.time() - lr) < GRACE:
                STATE["state"] = f"resting · {GRACE - (time.time()-lr):.0f}s to wake"; continue
            if not S["LOCK"].acquire(blocking=False): continue
            try:
                STATE["state"] = "dreaming · probing the 122B"
                _ensure_wrapped()
                cat, probe = _next_probe()
                _probe(cat, probe)
                STATE["probes"] += 1; STATE["last_cat"] = cat; STATE["last_probe"] = probe[:60]
                STATE["cats"] = sum(1 for c in _names if TOKS[c] > 0)
            finally:
                DREAM["logging"] = False
                S["LOCK"].release()
            if STATE["probes"] % SAVE_EVERY == 0: _save()
        except Exception as ex:
            STATE["state"] = f"err: {str(ex)[:40]}"; time.sleep(5)

def start(M, DRAFT, big_fwd, LOCK, BACKEND, LAST_REQ, EOS):
    _S.update(M=M, DRAFT=DRAFT, big_fwd=big_fwd, LOCK=LOCK, BACKEND=BACKEND, LAST_REQ=LAST_REQ, EOS=EOS)
    threading.Thread(target=_loop, daemon=True, name="dreamer").start()
