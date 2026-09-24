"""Лингвистический профиль текста по Profiling-UD (Brunato et al., LREC 2020).

Признаки считаются по UD-разбору. Названия и группы совпадают с тем, что отдаёт веб-версия
Profiling-UD, чтобы таблицы можно было сравнивать напрямую.
"""
from collections import Counter

import numpy as np

UPOS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON',
        'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
DEPS = ['acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'cc',
        'ccomp', 'compound', 'conj', 'cop', 'csubj', 'det', 'fixed', 'flat', 'iobj', 'mark',
        'nmod', 'nmod:poss', 'nsubj', 'nsubj:pass', 'nummod', 'obj', 'obl', 'parataxis',
        'punct', 'root', 'xcomp']
LEXICAL = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
CLAUSAL = {'acl', 'acl:relcl', 'advcl', 'ccomp', 'csubj', 'xcomp'}
MORPH = {'VerbForm': ['Fin', 'Inf', 'Part', 'Ger', 'Conv'], 'Tense': ['Past', 'Pres', 'Fut'],
         'Mood': ['Ind', 'Imp', 'Cnd'], 'Person': ['1', '2', '3'], 'Voice': ['Act', 'Pass', 'Mid'],
         'Aspect': ['Perf', 'Imp']}


def _feats(w):
    return dict(kv.split('=', 1) for kv in (w['feats'] or '').split('|') if '=' in kv)


def _depth(i, head):
    d = 0
    while head[i] != 0:
        i, d = head[i], d + 1
    return d


def _ttr(tokens, n):
    chunks = [tokens[i:i + n] for i in range(0, len(tokens) - n + 1, n)]
    return np.mean([len(set(c)) / n for c in chunks]) if chunks else np.nan


def _dist(counter, keys, prefix, total):
    return {f'{prefix}_{k}': counter.get(k, 0) / total * 100 if total else 0 for k in keys}


def profile(sents):
    """sents: список предложений, предложение это список слов-словарей
    с ключами id, text, lemma, upos, feats, head, deprel."""
    F = {}
    words = [w for s in sents for w in s]
    n_tok = len(words)
    no_punct = [w for w in words if w['upos'] != 'PUNCT']
    F['n_sentences'] = len(sents)
    F['n_tokens'] = n_tok
    F['tokens_per_sent'] = n_tok / max(len(sents), 1)
    F['char_per_tok'] = np.mean([len(w['text']) for w in no_punct]) if no_punct else 0

    forms = [w['text'].lower() for w in no_punct]
    lemmas = [(w['lemma'] or w['text']).lower() for w in no_punct]
    for n in (100, 200):
        F[f'ttr_form_chunks_{n}'] = _ttr(forms, n)
        F[f'ttr_lemma_chunks_{n}'] = _ttr(lemmas, n)

    upos = Counter(w['upos'] for w in words)
    F.update(_dist(upos, UPOS, 'upos_dist', n_tok))
    F['lexical_density'] = sum(upos[p] for p in LEXICAL) / max(len(no_punct), 1)

    verbs = [w for w in words if w['upos'] in ('VERB', 'AUX')]
    for feat, vals in MORPH.items():
        c = Counter(_feats(w).get(feat) for w in verbs)
        F.update(_dist(c, vals, f'verbs_{feat.lower()}_dist', len(verbs)))

    dep = Counter(w['deprel'] for w in words)
    F.update(_dist(dep, DEPS, 'dep_dist', n_tok))

    depths, max_links, links, clause_tok, n_clauses = [], [], [], [], 0
    verb_heads, verb_root, verb_edges = 0, 0, []
    prep_chains, sub_chains = [], []
    subj_pre = subj_post = obj_pre = obj_post = 0
    principal = subordinate = sub_pre = sub_post = 0
    for s in sents:
        head = {w['id']: w['head'] for w in s}
        by_id = {w['id']: w for w in s}
        kids = {}
        for w in s:
            kids.setdefault(w['head'], []).append(w)
        depths.append(max(_depth(w['id'], head) for w in s))
        ll = [abs(w['id'] - w['head']) for w in s if w['head'] != 0 and w['upos'] != 'PUNCT']
        links += ll
        max_links.append(max(ll) if ll else 0)

        for w in s:
            if w['upos'] == 'VERB':
                verb_heads += 1
                verb_edges.append(len([k for k in kids.get(w['id'], []) if k['upos'] != 'PUNCT']))
                if w['head'] == 0:
                    verb_root += 1
            if w['deprel'] in ('nsubj', 'nsubj:pass'):
                subj_pre += w['id'] < w['head']; subj_post += w['id'] > w['head']
            if w['deprel'] == 'obj':
                obj_pre += w['id'] < w['head']; obj_post += w['id'] > w['head']

        # клауза это предикат с его зависимыми, приблизительно число глаголов плюс копулы
        nc = sum(1 for w in s if w['upos'] == 'VERB' or w['deprel'] == 'cop') or 1
        n_clauses += nc
        clause_tok.append(len(s) / nc)

        for w in s:
            if w['deprel'] in CLAUSAL:
                subordinate += 1
                sub_pre += w['id'] < w['head']; sub_post += w['id'] > w['head']
                ln, h = 1, w['head']
                while h in by_id and by_id[h]['deprel'] in CLAUSAL:
                    ln, h = ln + 1, by_id[h]['head']
                sub_chains.append(ln)
            elif w['head'] == 0 or w['deprel'] in ('conj', 'parataxis') and w['upos'] == 'VERB':
                principal += 1

        # цепочки предложных групп: nmod с case, вложенные друг в друга
        for w in s:
            if w['deprel'].startswith('nmod') and any(k['deprel'] == 'case' for k in kids.get(w['id'], [])):
                par = by_id.get(w['head'])
                if par is not None and par['deprel'].startswith('nmod'):
                    continue
                ln, cur = 1, w
                while True:
                    nxt = [k for k in kids.get(cur['id'], []) if k['deprel'].startswith('nmod')
                           and any(g['deprel'] == 'case' for g in kids.get(k['id'], []))]
                    if not nxt:
                        break
                    ln, cur = ln + 1, nxt[0]
                prep_chains.append(ln)

    F['avg_max_depth'] = np.mean(depths)
    F['avg_max_links_len'] = np.mean(max_links)
    F['avg_links_len'] = np.mean(links) if links else 0
    F['max_links_len'] = max(max_links)
    F['avg_token_per_clause'] = np.mean(clause_tok)
    F['verbal_head_per_sent'] = verb_heads / len(sents)
    F['verbal_root_perc'] = verb_root / len(sents) * 100
    F['avg_verb_edges'] = np.mean(verb_edges) if verb_edges else 0
    ve = Counter(min(e, 6) for e in verb_edges)
    F.update(_dist(ve, range(7), 'verb_edges_dist', len(verb_edges)))
    F['n_prepositional_chains'] = len(prep_chains) / len(sents)
    F['avg_prepositional_chain_len'] = np.mean(prep_chains) if prep_chains else 0
    F['subj_pre'] = subj_pre / max(subj_pre + subj_post, 1) * 100
    F['obj_post'] = obj_post / max(obj_pre + obj_post, 1) * 100
    F['principal_proposition_dist'] = principal / max(principal + subordinate, 1) * 100
    F['subordinate_proposition_dist'] = subordinate / max(principal + subordinate, 1) * 100
    F['subordinate_pre'] = sub_pre / max(subordinate, 1) * 100
    F['avg_subordinate_chain_len'] = np.mean(sub_chains) if sub_chains else 0
    return F


def to_conllu(sents):
    out = []
    for s in sents:
        for w in s:
            out.append('\t'.join(str(x) for x in (w['id'], w['text'], w['lemma'] or '_', w['upos'], '_',
                                                   w['feats'] or '_', w['head'], w['deprel'], '_', '_')))
        out.append('')
    return '\n'.join(out) + '\n'


def parse(nlp, texts, batch=64):
    """UD-разбор stanza, на выходе список документов в виде списков предложений."""
    import stanza
    res = []
    for i in range(0, len(texts), batch):
        docs = nlp.bulk_process([stanza.Document([], text=t) for t in texts[i:i + batch]])
        for d in docs:
            res.append([[{'id': w.id, 'text': w.text, 'lemma': w.lemma, 'upos': w.upos, 'feats': w.feats,
                          'head': w.head, 'deprel': w.deprel} for w in s.words] for s in d.sentences])
    return res
