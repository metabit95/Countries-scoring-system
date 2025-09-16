#!/usr/bin/env python3
"""country_ranking_cli.py

Valutazione comparativa di paesi (stipendio, costo vita, qualità, ecc.)
-----------------------------------------------------------------------
Script CLI autosufficiente che:
  1. Pulisce e valida il CSV dei paesi (versione *raw* con note e percentuali).
  2. Applica uno fra 5 template di pesi o un test di robustezza (±15 %).
  3. Salva i ranking in cartelle dedicate e stampa riepiloghi.

Uso rapido (da shell):
    python country_ranking_cli.py

Dipendenze: pandas ≥ 2, numpy ≥ 1, Python ≥ 3.9.

Autore: ChatGPT (o3) – refactor 2025‑06‑30
"""

# ----------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------
from __future__ import annotations

import re
import sys
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# COSTANTI & CONFIG
# ----------------------------------------------------------------------
TEMPLATE_DIR = {
    0: "Reddito_Costo_Mix",
    1: "PPA_Qualita",
    2: "Career_Boost",
    3: "Quality_of_Life",
    4: "Low_Cost_Living",
    5: "Aggregato",
    6: "Robustezza",
}

# === Template 0 – Baseline A =================================================
weights_baselineA = {
    "Reddito": 0.25,
    "CostoVita": 0.20,
    "Salute": 0.15,
    "Sicurezza": 0.10,
    "QualitaVita": 0.10,
    "PotereAcquisto": 0.05,
    "Mercato_IT": 0.05,
    "EFI": 0.05,
    "Gini": 0.05,
}
# === Template 1 – Baseline B =================================================
weights_baselineB = {
    "PotereAcquisto": 0.25,
    "CostoVita": 0.20,
    "Salute": 0.15,
    "Sicurezza": 0.10,
    "QualitaVita": 0.10,
    "Reddito": 0.05,
    "Mercato_IT": 0.05,
    "EFI": 0.05,
    "Gini": 0.05,
}
# === Template 2 – Career Boost ==============================================
weights_career = {
    "Reddito": 0.30,
    "Mercato_IT": 0.20,
    "Tassazione": 0.15,  # ↓ tasse = ↑ punteggio
    "PotereAcquisto": 0.10,
    "CostoVita": 0.05,
    "Salute": 0.05,
    "EFI": 0.10,
    "Gini": -0.05,
}
# === Template 3 – Qualità della vita ========================================
weights_qualita = {
    "Salute": 0.20,
    "Sicurezza": 0.15,
    "QualitaVita": 0.25,
    "CostoVita": 0.10,
    "Reddito": 0.10,
    "EFI": 0.10,
    "Tassazione": 0.05,
    "Gini": 0.05,
}
# === Template 4 – Costi bassi ===============================================
weights_costi = {
    "CostoVita": 0.30,
    "Tassazione": 0.25,
    "CostoAffitti": 0.15,
    "PotereAcquisto": 0.15,
    "Reddito": 0.05,
    "Mercato_IT": 0.05,
    "Salute": 0.03,
    "EFI": 0.02,
    "Gini": 0.05,
}

TEMPLATE_WEIGHTS: Dict[int, Dict[str, float]] = {
    0: weights_baselineA,
    1: weights_baselineB,
    2: weights_career,
    3: weights_qualita,
    4: weights_costi,
}

# --- Cleaner regex ----------------------------------------------------------
TEXT_COLS = [
    "Nome paese",
    "Continente/regione",
    "Valuta",
    "Lingue officiali",
]

URL_RE = re.compile(r"https?://\S+")
SUP_RE = re.compile(r"[\u00B9\u00B2\u00B3\u2070-\u2079]")  # ¹²³…
PAREN_RE = re.compile(r"\([^)]*\)")
NOTE_RE = re.compile(r"\b(news|dato incerto|relocate\.me)\b", flags=re.I)
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

# --- Column map -------------------------------------------------------------
COLUMN_MAP = {
    "Reddito": "RAL media in euro (2024)",
    "Tassazione": "% delle tasse dalla RAL media Salary After Tax",
    "Mercato_IT": "Valore del mercato IT Euro",
    "PotereAcquisto": "Indice potere di acquisto 2025 mid-year (Numbeo)",
    "CostoVita": "Indice del costo della vita 2025 mid-year (Numbeo)",
    "CostoAffitti": "Indice del costo dell'affitto 2025 mid-year (Numbeo)",
    "Sicurezza": "Indice della sicurezza 2025 mid-year (Numbeo)",
    "Salute": "Indice di qualita della sanita 2025 mid-year (Numbeo)",
    "QualitaVita": "Indice qualita della vita  2025 mid-year (Numbeo)",
    "EFI": "English Proficiency Score 2024 (EF EPI)",
    "Indicizzazione": "Indicizzazione stipendi: si =1, no =0",
    "Gini": "Indice di Gini",  # opzionale
}
REQUIRED = set(v for k, v in COLUMN_MAP.items() if k != "Gini")
LOW_IS_BETTER = {"CostoVita", "CostoAffitti", "Tassazione", "Gini_val"}

# ----------------------------------------------------------------------
# FUNZIONI DI UTILITÀ
# ----------------------------------------------------------------------

def clean_text(cell: str) -> str:
    """Rimuove URL, note, parentesi e spazi doppi da celle testuali."""
    s = URL_RE.sub("", str(cell))
    s = SUP_RE.sub("", s)
    s = PAREN_RE.sub("", s)
    s = NOTE_RE.sub("", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def extract_number(cell: str) -> float | np.nan:
    """Estrae il primo numero in una stringa, altrimenti NaN."""
    m = NUMBER_RE.search(str(cell))
    return np.nan if not m else float(m.group(0).replace(",", "."))


def normalize_metric(series: pd.Series, invert: bool = False) -> pd.Series:
    """Scala 0‑100; se *invert* True, 100 va al minimo."""
    if series.dropna().nunique() <= 1:
        return pd.Series(0, index=series.index)
    scaled = (series - series.min()) / (series.max() - series.min()) * 100
    return 100 - scaled if invert else scaled


def generate_random_variants(base: Dict[str, float], n: int = 4, delta: float = 0.15) -> List[Dict[str, float]]:
    """Crea *n* varianti dei pesi, moltiplicando ogni valore per (1±delta)."""
    variants = []
    for _ in range(n):
        v = {k: max(val * random.uniform(1 - delta, 1 + delta), 0) for k, val in base.items()}
        total = sum(v.values())
        variants.append({k: val / total for k, val in v.items()})
    return variants

# ----------------------------------------------------------------------
# PIPELINE DATI
# ----------------------------------------------------------------------

def load_and_clean_csv(path: Path) -> pd.DataFrame:
    """Legge il CSV *raw*, ripulisce testo e converte i numeri."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # pulizia
    for col in df.columns:
        if col in TEXT_COLS:
            df[col] = df[col].apply(clean_text)
        else:
            df[col] = df[col].apply(extract_number)
    # cast numerico + median‑impute
    for col in df.columns.difference(TEXT_COLS):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not df[col].isna().all():
            df[col].fillna(df[col].median(), inplace=True)
    # verifica header
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Mancano colonne obbligatorie: {', '.join(missing)}")
    # rinomina
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items() if v in df.columns})
    df = df[df["Nome paese"].str.strip() != ""].reset_index(drop=True)
    return df


def calc_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Calcola punteggio pesato per ciascun paese."""
    score = pd.Series(0.0, index=df.index)
    for ind, w in weights.items():
        if ind not in df.columns or df[ind].isna().all():
            continue
        col_norm = normalize_metric(df[ind], invert=ind in LOW_IS_BETTER)
        score += col_norm * w
    return score

# ----------------------------------------------------------------------
# OUTPUT RANKING
# ----------------------------------------------------------------------

def save_ranking(df_names: pd.Series, scores: pd.Series, path: Path) -> None:
    res = pd.DataFrame({"Nome Paese": df_names, "TotalScore": scores})
    res["Rank"] = res["TotalScore"].rank(method="min", ascending=False, na_option="bottom").astype("Int64")
    res.sort_values("TotalScore", ascending=False, inplace=True)
    res.to_csv(path, index=False)

# ----------------------------------------------------------------------
# ROBUSTEZZA
# ----------------------------------------------------------------------

def robustezza(df: pd.DataFrame, out_dir: Path) -> None:
    """Esegue 12 scenari (A/B + 8 varianti) e salva riepilogo."""
    variants: Dict[str, Dict[str, float]] = {
        **{f"A_var{i}": w for i, w in enumerate(generate_random_variants(weights_baselineA), 1)},
        **{f"B_var{i}": w for i, w in enumerate(generate_random_variants(weights_baselineB), 1)},
        "BaselineA": weights_baselineA,
        "BaselineB": weights_baselineB,
    }
    all_scores = {}
    for tag, w in variants.items():
        s = calc_scores(df, w)
        all_scores[tag] = s
        save_ranking(df["Nome paese"], s, out_dir / f"Ranking_{tag}.csv")
    # aggregato + riepilogo
    scores_mat = pd.DataFrame(all_scores)
    mean_score = scores_mat.mean(axis=1)
    save_ranking(df["Nome paese"], mean_score, out_dir / "Ranking_Varianti_Aggregato.csv")

    ranks = scores_mat.rank(method="min", ascending=False)
    ranks.index = df["Nome paese"].values
    stability = pd.DataFrame({
        "min": ranks.min(axis=1),
        "max": ranks.max(axis=1),
        "top3%": (ranks <= 3).mean(axis=1) * 100,
    })
    best = stability.sort_values("top3%", ascending=False).iloc[0]
    rep = [
        "=== Riepilogo robustezza ===",
        "Top3 ≥80 %: " + ", ".join(stability[stability["top3%"] >= 80].index),
        "Stabili (≤2 pos.): " + ", ".join(stability[stability["max"] - stability["min"] <= 2].index),
        "Instabili (≥6 pos.): " + ", ".join(stability[stability["max"] - stability["min"] >= 6].index),
        f"{best.name} resta primo nel {best['top3%']:.0f}% degli scenari (range {int(best['min'])}–{int(best['max'])}).",
    ]
    print("\n".join(rep))
    (out_dir / "report_robustezza.txt").write_text("\n".join(rep), encoding="utf-8")

# ----------------------------------------------------------------------
# MAIN CLI
# ----------------------------------------------------------------------

def ask_template() -> int:
    print(" 0 = Reddito & Costo Mix        (Baseline A: reddito 25 %, costo-vita 20 %)")
    print(" 1 = PPA + Qualità della vita   (Baseline B: potere d’acquisto 25 %)")
    print(" 2 = Career Boost               (stipendio + mercato IT, tasse penalizzate)")
    print(" 3 = Quality of Life            (sanità, sicurezza, indice QoL, ecc.)")
    print(" 4 = Low-Cost Living            (costo-vita, tasse, affitti bassi)")
    print(" 5 = Tutti i template           (0-4) con media aggregata")
    print(" 6 = Analisi robustezza         (10 varianti casuali ±15 % sui pesi)")
    while True:
        try:
            c = int(input("Scelta template (0-6): ").strip())
            if 0 <= c <= 6:
                return c
        except ValueError:
            pass
        print("Inserisci un numero da 0 a 6.")


def main() -> None:
    choice = ask_template()

    # opzionale: modifica pesi
    if input("Vuoi modificare i pesi? (s/n): ").lower().startswith("s") and choice != 6:
        sel = TEMPLATE_WEIGHTS[choice] if choice < 5 else TEMPLATE_WEIGHTS[0]
        for k, v in sel.items():
            val = input(f"{k} (attuale {v:.3f}): ").strip()
            if val:
                try:
                    sel[k] = float(val)
                except ValueError:
                    pass
        tot = sum(sel.values())
        for k in sel:
            sel[k] /= tot

    # input CSV con retry
    while True:
        p = Path(input("Percorso CSV dei paesi: ")).expanduser().resolve()
        if p.is_file():
            try:
                df = load_and_clean_csv(p)
                break
            except Exception as e:
                print(f"⚠️  {e}")
        print("File non valido. Riprova…")

    # output base
    out_base = Path(input("Cartella di output (vuoto=./results): ") or "results").expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    # calcoli ----------------------------------------------------------------
    if choice in TEMPLATE_WEIGHTS:  # 0‑4
        out_dir = out_base / TEMPLATE_DIR[choice]
        out_dir.mkdir(parents=True, exist_ok=True)
        scr = calc_scores(df, TEMPLATE_WEIGHTS[choice])
        save_ranking(df["Nome paese"], scr, out_dir / f"Ranking_{TEMPLATE_DIR[choice]}.csv")
        print(f"✅  Salvato ranking in {out_dir}")

    elif choice == 5:  # aggregato
        out_dir = out_base / TEMPLATE_DIR[5]
        out_dir.mkdir(parents=True, exist_ok=True)
        all_scores = {}
        for i, w in TEMPLATE_WEIGHTS.items():
            s = calc_scores(df, w)
            all_scores[i] = s
            save_ranking(df["Nome paese"], s, out_dir / f"Ranking_{TEMPLATE_DIR[i]}.csv")
        mean_s = pd.DataFrame(all_scores).mean(axis=1)
        save_ranking(df["Nome paese"], mean_s, out_dir / "Ranking_Aggregato.csv")
        print(f"✅  Salvate 6 classifiche in {out_dir}")

    else:  # robustezza
        robustezza(df, out_base / TEMPLATE_DIR[6])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
