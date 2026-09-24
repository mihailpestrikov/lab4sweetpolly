"""Семантические и статистические метрики текста."""
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9«"(])')


def free():
    import gc
    gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()


def sentences(t):
    return [s for s in SENT_RE.split(t) if s.strip()]


class BARTScorer:
    """Порт BARTScorer из neulab/BARTScore: средний log p(tgt | src) на токен."""

    def __init__(self, name, src_lang=None, max_length=512):
        self.tok = AutoTokenizer.from_pretrained(name)
        if src_lang:
            self.tok.src_lang = self.tok.tgt_lang = src_lang
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE).eval()
        self.max_length = max_length

    @torch.no_grad()
    def score(self, srcs, tgts, batch_size=4):
        # сортировка по длине убирает лишний паддинг, порядок потом восстанавливается
        order = np.argsort([len(s) + len(t) for s, t in zip(srcs, tgts)])
        out = np.zeros(len(srcs))
        for n, i in enumerate(range(0, len(srcs), batch_size)):
            ix = order[i:i + batch_size]
            s = self.tok([srcs[j] for j in ix], max_length=self.max_length, truncation=True,
                         padding=True, return_tensors='pt').to(DEVICE)
            t = self.tok(text_target=[tgts[j] for j in ix], max_length=self.max_length, truncation=True,
                         padding=True, return_tensors='pt').to(DEVICE)
            labels = t.input_ids.masked_fill(t.attention_mask == 0, -100)
            logits = self.model(**s, labels=labels).logits
            nll = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none', ignore_index=-100)
            out[ix] = (-nll.sum(1) / t.attention_mask.sum(1)).cpu().numpy()
            del logits, nll
            if n % 5 == 0:
                free()
        return out


class LMStats:
    """Перплексия, средняя энтропия предсказания (как в mgt-detection-benchmark) и burstiness,
    то есть разброс лог-перплексии между предложениями одного текста."""

    def __init__(self, name, max_length=512):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE).eval()
        self.max_length = max_length

    @torch.no_grad()
    def _token_stats(self, texts, batch_size=8):
        order = np.argsort([len(t) for t in texts])
        res = [None] * len(texts)
        for n, i in enumerate(range(0, len(texts), batch_size)):
            ix = order[i:i + batch_size]
            if n % 5 == 0:
                free()
            b = self.tok([texts[j] for j in ix], max_length=self.max_length, truncation=True,
                         padding=True, return_tensors='pt').to(DEVICE)
            logits = self.model(**b).logits[:, :-1].float()
            tgt, m = b.input_ids[:, 1:], b.attention_mask[:, 1:].bool()
            logp = logits.log_softmax(-1)
            nll = -logp.gather(-1, tgt[..., None])[..., 0]
            ent = -(logp.exp() * logp).sum(-1)
            rank = (logits > logits.gather(-1, tgt[..., None])).sum(-1) + 1
            for j, k in enumerate(ix):
                res[k] = (nll[j][m[j]].cpu().numpy(), ent[j][m[j]].cpu().numpy(), rank[j][m[j]].cpu().numpy())
        return res

    def __call__(self, texts):
        rows = []
        for nll, ent, rank in self._token_stats(texts):
            rows.append({'log_ppl': nll.mean(), 'entropy': ent.mean(),
                         'log_rank': np.log(rank).mean(), 'top10_share': (rank <= 10).mean()})
        # burstiness по предложениям, короткие отбрасываю
        sents = [[s for s in sentences(t) if len(s.split()) >= 4] for t in texts]
        flat = [s for ss in sents for s in ss]
        per_sent = iter([x[0].mean() for x in self._token_stats(flat, batch_size=16)])
        for r, ss in zip(rows, sents):
            v = [next(per_sent) for _ in ss]
            r['burstiness'] = np.std(v) if len(v) > 1 else np.nan
        return rows


def tokens(t):
    return re.findall(r'\w+', t.lower())


def distinct_n(toks, n):
    ng = list(zip(*[toks[i:] for i in range(n)]))
    return len(set(ng)) / max(len(ng), 1)


def mtld(toks, thr=0.72):
    def one_pass(ts):
        factors, types, cnt = 0, set(), 0
        for w in ts:
            types.add(w); cnt += 1
            if len(types) / cnt <= thr:
                factors, types, cnt = factors + 1, set(), 0
        if cnt:
            factors += (1 - len(types) / cnt) / (1 - thr)
        return len(ts) / factors if factors else np.nan
    return np.nanmean([one_pass(toks), one_pass(toks[::-1])])


def self_bleu(texts, n_hyp=200, n_ref=200, seed=0):
    """Self-BLEU (Zhu et al., Texygen): каждый текст против остальных текстов того же класса."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    rng = np.random.default_rng(seed)
    toks = [tokens(t) for t in texts]
    hyp = rng.choice(len(toks), min(n_hyp, len(toks)), replace=False)
    sm = SmoothingFunction().method1
    scores = []
    for i in hyp:
        refs = rng.choice([j for j in range(len(toks)) if j != i], n_ref, replace=False)
        scores.append(sentence_bleu([toks[j] for j in refs], toks[i], smoothing_function=sm))
    return float(np.mean(scores))


@torch.no_grad()
def embed(texts, name='intfloat/multilingual-e5-small', batch_size=64):
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).to(DEVICE).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        b = tok(['query: ' + t for t in texts[i:i + batch_size]], max_length=512, truncation=True,
                padding=True, return_tensors='pt').to(DEVICE)
        h = model(**b).last_hidden_state
        m = b.attention_mask[..., None]
        out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy())
    del model; free()
    return np.vstack(out)
