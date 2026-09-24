"""Загрузка и выборка текстов. Общая для ноутбука и фонового разбора."""
import pandas as pd

N_EN_ROWS, N_RU_PER_CLASS, SEED = 1500, 1200, 42
EN_COLS = {'Human-Generated': 'human', 'ChatGPT-Generated': 'ai', 'Mixed Text': 'hybrid'}


def load_en():
    x = pd.read_excel('data/AIGTxt.xlsx').iloc[:, :4]
    x['Domain'] = x.Domain.str.strip().str.title()
    x = x.drop_duplicates('Human-Generated').reset_index(drop=True)
    x['pair_id'] = x.index
    x = x.sample(N_EN_ROWS, random_state=SEED)
    en = x.melt(id_vars=['pair_id', 'Domain'], value_vars=list(EN_COLS), var_name='label', value_name='text')
    en['label'] = en.label.map(EN_COLS)
    en['text'] = en.text.astype(str).str.split().str.join(' ')
    return en.rename(columns={'Domain': 'domain'}).sort_values(['pair_id', 'label']).reset_index(drop=True)


def load_ru():
    d = pd.read_csv('data/AINL-Eval-2025/data/train.csv').drop_duplicates('text')
    d['text'] = d.text.astype(str).str.split().str.join(' ')
    d = d.groupby('label', group_keys=False).sample(N_RU_PER_CLASS, random_state=SEED)
    return d.rename(columns={'label': 'model'}).assign(
        label=lambda t: t.model.where(t.model == 'human', 'ai')).reset_index(drop=True)
