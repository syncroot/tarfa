# quality.py — decode-time QUALITY strategies for the 122B (spend N x compute for better reasoning).
#   vote   : self-consistency  — N temp-samples, then synthesize the consensus answer
#   judge  : best-of-N + judge  — N diverse drafts, then score & merge the best
#   refine : self-refine loop   — draft, then K rounds of check-and-correct
# Each run() is a generator yielding ('status', msg) | ('token', text). The final answer streams as tokens.
import torch
from transformers.cache_utils import DynamicCache

def _gen(M, big_fwd, EOS, ids, max_new, temp, top_p, cancel):
    pids = ids[0].tolist(); cache = DynamicCache(config=M.cfg)
    if len(pids) > 1: big_fwd(pids[:-1], cache)
    pending = pids[-1]; produced = 0
    while produced < max_new:
        if cancel is not None and cancel.is_set(): return
        logits = big_fwd([pending], cache)[-1]
        if temp and temp > 0:                                  # sampled (diverse)
            probs = torch.softmax(logits.float() / temp, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True); cum = sp.cumsum(0)
                sp[(cum - sp) > top_p] = 0.0; sp = sp / sp.sum()
                nxt = int(si[torch.multinomial(sp, 1)])
            else:
                nxt = int(torch.multinomial(probs, 1))
        else:                                                  # greedy (deterministic, for synthesis)
            nxt = int(logits.argmax())
        if nxt == EOS: return
        yield M.tok.decode([nxt]); pending = nxt; produced += 1

def _collect(M, big_fwd, EOS, bp, msgs, mx, temp, top_p, cancel):
    ids = M.tok(bp(msgs), return_tensors="pt").input_ids
    return "".join(_gen(M, big_fwd, EOS, ids, mx, temp, top_p, cancel))

def _drafts_block(drafts):
    return "\n\n".join(f"--- Candidate {i+1} ---\n{d.strip()}" for i, d in enumerate(drafts))

def _vote_turn(drafts):
    return ("I independently produced the following answers to my previous request:\n\n" + _drafts_block(drafts) +
            "\n\nAct as a careful aggregator. For every part of the task, take the result the MAJORITY of the candidates "
            "agree on; where they disagree, decide strictly by correctness (re-derive if needed). Produce the single best, "
            "complete final answer. Do NOT mention these candidates or that any aggregation happened.")

def _judge_turn(drafts):
    return ("Below are candidate answers to my previous request:\n\n" + _drafts_block(drafts) +
            "\n\nEvaluate them for correctness and completeness, silently discard the flawed ones, then write the single "
            "best final answer, merging the strongest and most correct parts. Output ONLY that final answer — no scores, "
            "no commentary about the candidates.")

def _refine_turn(ans):
    return ("Here is a draft answer to my previous request:\n\n--- Draft ---\n" + ans.strip() +
            "\n\nCarefully check it for errors, omissions, and faulty reasoning. Re-derive anything uncertain. Then write a "
            "corrected, improved FINAL answer. Output ONLY the final answer — do not show your critique.")

def run(strategy, n, M, big_fwd, EOS, bp, msgs, mx, cancel, temp=0.7, top_p=0.95):
    draft_mx = min(mx, 1200)                                   # bound per-draft time; final gets full mx
    if strategy == "vote":
        drafts = []
        for i in range(n):
            yield ("status", f"sampling independent answer {i+1}/{n}")
            drafts.append(_collect(M, big_fwd, EOS, bp, msgs, draft_mx, temp, top_p, cancel))
            if cancel is not None and cancel.is_set(): return
        yield ("status", "synthesizing the consensus answer")
        am = msgs + [{"role": "user", "content": _vote_turn(drafts)}]
        for t in _gen(M, big_fwd, EOS, M.tok(bp(am), return_tensors="pt").input_ids, mx, 0.0, 1.0, cancel): yield ("token", t)
    elif strategy == "judge":
        drafts = []
        for i in range(n):
            yield ("status", f"drafting candidate {i+1}/{n}")
            drafts.append(_collect(M, big_fwd, EOS, bp, msgs, draft_mx, temp, top_p, cancel))
            if cancel is not None and cancel.is_set(): return
        yield ("status", "judging and merging the best answer")
        jm = msgs + [{"role": "user", "content": _judge_turn(drafts)}]
        for t in _gen(M, big_fwd, EOS, M.tok(bp(jm), return_tensors="pt").input_ids, mx, 0.0, 1.0, cancel): yield ("token", t)
    elif strategy == "refine":
        yield ("status", "initial draft")
        ans = _collect(M, big_fwd, EOS, bp, msgs, mx, 0.0, 1.0, cancel)
        for k in range(n):
            if cancel is not None and cancel.is_set(): break
            yield ("status", f"check & correct — round {k+1}/{n}")
            rm = msgs + [{"role": "user", "content": _refine_turn(ans)}]
            if k == n - 1:
                buf = ""
                for t in _gen(M, big_fwd, EOS, M.tok(bp(rm), return_tensors="pt").input_ids, mx, 0.0, 1.0, cancel):
                    buf += t; yield ("token", t)
                ans = buf
            else:
                ans = _collect(M, big_fwd, EOS, bp, rm, mx, 0.0, 1.0, cancel)
