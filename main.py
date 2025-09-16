import pandas as pd
import numpy as np
import re
import sys
import random
import copy
from pathlib import Path

TEMPLATE_DIR = {
    0: "Reddito_Costo_Mix",
    1: "PPA_Qualita",
    2: "Career_Boost",
    3: "Quality_of_Life",
    4: "Low_Cost_Living",
    5: "Aggregato",
    6: "Robustezza",
}
# 1. Definizione dei pesi per ciascun template (dizionari)
    # === Template 0 : "Reddito & Costo Mix"  (Baseline A) ===
weights_baselineA = {
    'Reddito'        : 0.25,
    'CostoVita'      : 0.20,
    'Salute'         : 0.15,
    'Sicurezza'      : 0.10,
    'QualitaVita'    : 0.10,
    'PotereAcquisto' : 0.05,
    'Mercato_IT'     : 0.05,
    'EFI'            : 0.05,
    'Gini'           : 0.05      # (+) → bassa disuguaglianza premia
}
    # === Template 1 : "Potere d’Acquisto + Qualità" (Baseline B) ===
weights_baselineB = {
    'PotereAcquisto' : 0.25,
    'CostoVita'      : 0.20,
    'Salute'         : 0.15,
    'Sicurezza'      : 0.10,
    'QualitaVita'    : 0.10,
    'Reddito'        : 0.05,
    'Mercato_IT'     : 0.05,
    'EFI'            : 0.05,
    'Gini'           : 0.05
}
    # === Template 2 : Somma pesi positivi = 1.05, Gini −0.05 ⇒ totale 1.00 ===
weights_career = {
    'Reddito'        : 0.30,
    'Mercato_IT'     : 0.20,
    'Tassazione'     : 0.15,   # invertito (↓ tasse = ↑ punteggio)
    'PotereAcquisto' : 0.10,
    'CostoVita'      : 0.05,
    'Salute'         : 0.05,
    'EFI'            : 0.10,
    'Gini'           : -0.05   # penalizza forte disuguaglianza
}
    # === Template 3 : Prientato alla qualità della vita ===
weights_qualita = {
    'Salute'         : 0.20,
    'Sicurezza'      : 0.15,
    'QualitaVita'    : 0.25,   # indice Numbeo complessivo
    'CostoVita'      : 0.10,
    'Reddito'        : 0.10,
    'EFI'            : 0.10,
    'Tassazione'     : 0.05,
    'Gini'           : 0.05
}
    # === Template 4 : Prientato ai costi minori ===
weights_costi = {
    'CostoVita'      : 0.30,   # invertito (↓ costo = ↑ punteggio)
    'Tassazione'     : 0.25,   # invertito
    'CostoAffitti'   : 0.15,   # invertito
    'PotereAcquisto' : 0.15,
    'Reddito'        : 0.05,
    'Mercato_IT'     : 0.05,
    'Salute'         : 0.03,
    'EFI'            : 0.02,
    'Gini'           : 0.05
}

# Dizionario per mappare scelta utente -> weights
template_weights = {
    0: weights_baselineA,
    1: weights_baselineB,
    2: weights_career,
    3: weights_qualita,
    4: weights_costi
}


# --- Pulizia / cleaner ----------------------------------------------------
TEXT_COLS = [
    "Nome paese",
    "Continente/regione",
    "Valuta",
    "Lingue officiali",
]

URL_RE   = re.compile(r"https?://\S+")
SUP_RE   = re.compile(r"[\u00B9\u00B2\u00B3\u2070-\u2079]")   # ¹²³⁴…
PAREN_RE = re.compile(r"\([^)]*\)")
NOTE_RE  = re.compile(r"\b(news|dato incerto|relocate\.me)\b", flags=re.I)
NUMBER_RE= re.compile(r"-?\d+(?:[.,]\d+)?")

def clean_text(cell: str) -> str:
    s = str(cell)
    s = URL_RE.sub("", s)
    s = SUP_RE.sub("", s)
    s = PAREN_RE.sub("", s)
    s = NOTE_RE.sub("", s)
    return re.sub(r"\s{2,}", " ", s).strip()

def extract_number(cell: str):
    m = NUMBER_RE.search(str(cell))
    if not m:
        return np.nan
    return float(m.group(0).replace(",", "."))

# ---- MAPPATURA nomi CSV → nomi logici usati nei calcoli ------------------
COLUMN_MAP = {
    # === indicatori usati nel punteggio ===
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
    "Gini": "Indice di Gini",   # verrà ignorato se non presente
    # === anagrafiche (solo per stampa / info) ===
    #"country_name": "Nome paese",
}
# Set di colonne OBBLIGATORIE per poter calcolare i punteggi
REQUIRED = {v for k, v in COLUMN_MAP.items() if k not in ("Gini",)}  # Gini è opzionale

# Indicatori in cui un valore più BASSO è migliore
LOW_IS_BETTER = {'CostoVita', 'CostoAffitti', 'Tassazione'}

# 2. Funzione per normalizzare una serie in scala 0-100 (gestisce valori costanti)
def normalize_metric(series, invert=False):
    """Restituisce la colonna normalizzata 0-100.
       Se invert=True, 100 va al valore MINIMO (es. costo vita basso)."""
    min_val, max_val = series.min(), series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series([0]*len(series), index=series.index)
    scaled = (series - min_val) / (max_val - min_val) * 100
    return 100 - scaled if invert else scaled

# 3. Input interattivo da CLI
print(" 0 = Reddito & Costo Mix        (Baseline A: reddito 25 %, costo-vita 20 %)")
print(" 1 = PPA + Qualità della vita   (Baseline B: potere d’acquisto 25 %)")
print(" 2 = Career Boost               (stipendio + mercato IT, tasse penalizzate)")
print(" 3 = Quality of Life            (sanità, sicurezza, indice QoL, ecc.)")
print(" 4 = Low-Cost Living            (costo-vita, tasse, affitti bassi)")
print(" 5 = Tutti i template           (0-4) con media aggregata")
print(" 6 = Analisi robustezza         (10 varianti casuali ±15 % sui pesi)")
choice = int(input("Inserisci il numero del template (0-6): ").strip())

modify_weights = input("Vuoi modificare i pesi del template selezionato? (s/n): ").strip().lower()
if modify_weights == 's':
    if choice != 5:
        # Mostra e modifica i pesi per il template scelto
        sel_weights = template_weights[choice]
        print(f"Pesi attuali per il template {choice}:")
        for ind, peso in sel_weights.items():
            sign = "%" if abs(peso) <= 1 else ""  # se sono frazioni, li interpretiamo in percentuale
            print(f" - {ind}: {peso}{sign}")
        print("Inserisci i nuovi pesi per ciascun indicatore (lascia vuoto per mantenere il valore attuale):")
        new_weights = {}
        for ind, peso in sel_weights.items():
            user_val = input(f"{ind} (attuale {peso}): ").strip()
            if user_val == "":
                new_weights[ind] = peso  # nessuna modifica
            else:
                try:
                    val = float(user_val)
                except:
                    val = peso
                new_weights[ind] = val
        # Ricalibrazione se la somma non è 1 (o 100 se dati in percentuale)
        total = sum(new_weights.values())
        if total != 0:  # per sicurezza, evitiamo divisione per zero se utente azzera tutto
            for ind in new_weights:
                new_weights[ind] = new_weights[ind] / total
        template_weights[choice] = new_weights
        print("Nuovi pesi impostati:")
        for ind, peso in new_weights.items():
            print(f" - {ind}: {peso:.3f}")
    else:
        # Se aggregato, potremmo permettere di modificare uno o tutti i template:
        print("Modalità aggregata selezionata: modifica i pesi dei singoli template (0-4) prima di calcolare l'aggregato.")
        # (Per semplicità, in questa implementazione chiederemo quale template modificare e ripetere eventualmente)
        while True:
            tmpl = input("Inserisci l'indice del template da modificare (0-4) o premi Invio per continuare: ").strip()
            if tmpl == "":
                break
            try:
                tmpl = int(tmpl)
                if tmpl in template_weights:
                    sel_weights = template_weights[tmpl]
                    print(f"Pesi attuali per template {tmpl}:")
                    for ind, peso in sel_weights.items():
                        print(f" - {ind}: {peso}")
                    print("Modifica i pesi (lascia vuoto per mantenere):")
                    new_weights = {}
                    for ind, peso in sel_weights.items():
                        val = input(f"{ind} (attuale {peso}): ").strip()
                        if val == "":
                            new_weights[ind] = peso
                        else:
                            try:
                                new_weights[ind] = float(val)
                            except:
                                new_weights[ind] = peso
                    total = sum(new_weights.values())
                    if total != 0:
                        for ind in new_weights:
                            new_weights[ind] = new_weights[ind] / total
                    template_weights[tmpl] = new_weights
                    print("Aggiornati i pesi del template", tmpl)
                else:
                    print("Template non valido.")
            except ValueError:
                print("Inserisci un numero valido o lascia vuoto per uscire.")
        print("Pesi finali dei template aggiornati (per aggregato):")
        for t, w in template_weights.items():
            print(f" Template {t}:")
            for ind, peso in w.items():
                print(f"   - {ind}: {peso:.3f}")

# --- 4. Caricamento dataset con retry -------------------------------------
while True:
    csv_in = input("Percorso CSV dei paesi: ").strip()
    csv_path = Path(csv_in).expanduser().resolve()
    if not csv_path.is_file():
        print(f"❌  File non trovato: {csv_in}\nRiprova oppure Ctrl-C per uscire.")
        continue
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        break          # successo → esci dal while
    except Exception as e:
        print(f"⚠️  Errore nel leggere il CSV: {e}\nRiprova oppure Ctrl-C per uscire.")
    sys.exit(1)              # esci se il file non si apre

# --- cartella di output ------------------------------------------------
out_base = input("Cartella di output (vuoto = ./results): ").strip() or "results"
out_base = Path(out_base).expanduser().resolve()
out_base.mkdir(parents=True, exist_ok=True)
# --- da qui in poi la pulizia e la validazione ---------------------------

# 2a. Pulizia base
for col in df.columns:
    if col in TEXT_COLS:
        df[col] = df[col].apply(clean_text)
    else:
        df[col] = df[col].apply(extract_number)

# 2a bis. Conversione esplicita a float (+ forza NaN su residui non numerici)
for col in df.columns.difference(TEXT_COLS):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in df.columns.difference(TEXT_COLS):
        if not df[col].isna().all():
            df[col] = df[col].fillna(df[col].median())

# 2a ter. Imputazione mediane (evita propagazione di NaN)
for col in df.columns.difference(TEXT_COLS):
    if not df[col].isna().all():           # almeno un valore valido
        df[col] = df[col].fillna(df[col].median())


# 2a quater. Verifica: quante celle non-numeriche restano (solo colonne non testuali)
for col in df.columns.difference(TEXT_COLS):
    if df[col].astype(str).str.contains("[A-Za-z]", regex=True).any():
        bad = df[col].astype(str).str.contains("[A-Za-z]", regex=True).sum()
        print(f"⚠️  Colonna '{col}' contiene testo residuo in {bad} celle")

# 2b. Validazione header obbligatori
missing = REQUIRED - set(df.columns)
if missing:
    print("⚠️  Mancano queste colonne nel CSV:")
    for m in missing:
        print(f"   - {m}")
    sys.exit(1)
print("✓ Colonne verificate: tutto ok.")

# 2c. Rinominazione secondo COLUMN_MAP
df = df.rename(columns={v: k for k, v in COLUMN_MAP.items() if v in df.columns})

# --- 2d. Elimina righe con Nome paese mancante ---------------------------
df = df[df['Nome paese'].astype(str).str.strip() != ''].copy()
df.reset_index(drop=True, inplace=True)

# 5. Applicazione correzioni sui dati grezzi prima della normalizzazione
# 5a. Rimuovere eventuali conversioni valutarie (non necessario se5 dati già convertiti)
# --> Niente da fare qui poiché i dati sono già in euro (assunto).

# 5b. Assegnare punteggio EF EPI = 90 ai paesi anglofoni
anglo_countries = ["USA", "United States", "Regno Unito", "UK", "United Kingdom",
                   "Australia", "Nuova Zelanda", "New Zealand",
                   "Canada", "Irlanda", "Ireland", "Singapore"]
if 'EFI' in df.columns:
    df['EFI'] = df['EFI'].fillna(0)  # se ci fossero NaN
    mask_anglo = df['Nome paese'].isin(anglo_countries)
    if mask_anglo.any():
        df.loc[mask_anglo, 'EFI'] = 90  # assegna 90 ai paesi anglofoni

# 5c. Calcolo dello stipendio netto se richiesto
use_net = input("Vuoi calcolare il reddito netto dal lordo? (s/n): ").strip().lower()
if use_net == 's':
    # Controlla se ci sono le colonne necessarie
    possible_income_cols = ['Reddito']  # possibili nomi della colonna reddito lordo
    possible_tax_cols = ['Tassazione']
    col_income = None
    col_tax = None
    for col in possible_income_cols:
        if col in df.columns:
            col_income = col
            break
    for col in possible_tax_cols:
        if col in df.columns:
            col_tax = col
            break
    if col_income and col_tax:
        df['Reddito_Netto'] = df[col_income] * (1 - df[col_tax].astype(float))
        # Aggiorna eventualmente il nome dell'indicatore di reddito usato nei pesi
        # Se i template usavano 'Reddito', possiamo decidere di usare Reddito_Netto al suo posto
        # Sostituiamo nei dizionari dei pesi se l'utente sceglie netto
        for tw in template_weights.values():
            if 'Reddito' in tw:
                tw['Reddito_Netto'] = tw.pop('Reddito')
    else:
        print("Dati per calcolare il netto non disponibili, userò il reddito lordo.")

# 5d. Applicazione della trasformazione logaritmica a indicatori con outlier (es. Mercato_IT)
if 'Mercato_IT' in df.columns:
    # Applica log(1+x) per sicurezza
    df['Mercato_IT'] = np.log1p(df['Mercato_IT'])
    # Nota: la normalizzazione successiva riporterà poi su scala 0-100

# 5e. Controllo presenza Gini
has_gini = False
for col in df.columns:
    if 'Gini' in col or col.lower() == 'gini':
        has_gini = True
        # Se Gini presente, potremmo invertire il senso (calcoliamo 1 - Gini se serve in positivo)
        df['Gini_val'] = 1 - df[col]  # maggiore è questo, migliore è (0=ineguaglianza massima, 1=perfetta uguaglianza)
        break

# 6. Calcolo punteggi per il/dei template selezionati
# Funzione per calcolare punteggio di un template dato un dizionario di pesi
def calc_scores(df, weights_dict):
    """Calcola il punteggio totale di un template."""
    score = pd.Series(np.zeros(len(df)), index=df.index)

    for indicator, weight in weights_dict.items():

        # salta se la colonna non esiste o è tutta NaN
        if indicator not in df.columns or df[indicator].isna().all():
            continue

        # normalizza (eventualmente invertendo 0↔100)
        invert = indicator in LOW_IS_BETTER
        norm_col = normalize_metric(df[indicator], invert=invert)

        # somma pesata (peso negativo = penalizzazione)
        score += norm_col * weight

    return score

# Se l'utente ha scelto un template specifico (0-4)
if choice in template_weights:
    out_dir = (out_base / TEMPLATE_DIR[choice])
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = template_weights[choice]
    total_score = calc_scores(df, weights)
    # Aggiungi risultati al DataFrame per esportazione
    df_result = pd.DataFrame({
        'Nome Paese': df['Nome paese'] if 'Nome paese' in df.columns else df.iloc[:,0],
        'TotalScore': total_score
    })
    df_result['TotalScore'] = df_result['TotalScore'].fillna(0)   # rimpiazza NaN con 0
    df_result['Rank'] = (
    df_result['TotalScore']
    .rank(method='min', ascending=False, na_option='bottom')   # NaN vanno in fondo
    .astype('Int64')                                          # intero “nullable”
    )
    # Ordina per punteggio decrescente
    df_result = df_result.sort_values('TotalScore', ascending=False).reset_index(drop=True)
    # Salva CSV
    output_name = ""
    if choice == 0:
        output_name = "Ranking_BaselineA.csv"
    elif choice == 1:
        output_name = "Ranking_BaselineB.csv"
    elif choice == 2:
        output_name = "Ranking_Career.csv"
    elif choice == 3:
        output_name = "Ranking_QualitaVita.csv"
    elif choice == 4:
        output_name = "Ranking_Costi.csv"
    df_result.to_csv(out_dir / output_name, index=False)
    print(f"Classifica salvata su {output_name}")
    print(df_result.head(10).to_string(index=False))  # Mostra la top 10 in output per verifica

elif choice == 5:
    out_dir = (out_base / TEMPLATE_DIR[5])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Calcola tutti i template 0-4 e poi la media
    all_scores = {}
    for tmpl, wdict in template_weights.items():
        score = calc_scores(df, wdict)
        all_scores[tmpl] = score
        # Salviamo anche ciascuna classifica singola (opzionale, utile per debug)
        df_single = pd.DataFrame({
            'Nome Paese': df['Nome paese'] if 'Nome paese' in df.columns else df.iloc[:,0],
            'TotalScore': score
        })
        df_single['Rank'] = df_single['TotalScore'].rank(method='min', ascending=False).astype(int)
        df_single = df_single.sort_values('TotalScore', ascending=False).reset_index(drop=True)
        name_map = {0: "BaselineA", 1: "BaselineB", 2: "Career", 3: "QualitaVita", 4: "Costi"}
        df_single.to_csv(out_dir / f"Ranking_{name_map[tmpl]}.csv", index=False)
    # DataFrame per punteggio medio
    scores_matrix = pd.DataFrame(all_scores)
    mean_score = scores_matrix.mean(axis=1)
    df_agg = pd.DataFrame({
        'Nome Paese': df['Nome paese'] if 'Nome paese' in df.columns else df.iloc[:,0],
        'PunteggioMedio': mean_score
    })
    df_agg['Rank'] = df_agg['PunteggioMedio'].rank(method='min', ascending=False).astype(int)
    df_agg = df_agg.sort_values('PunteggioMedio', ascending=False).reset_index(drop=True)
    df_agg.to_csv(out_dir / "Ranking_Aggregato.csv", index=False)
    print("Classifiche per ciascun template salvate (Ranking_*.csv).")
    print("Classifica aggregata (media dei template) salvata su Ranking_Aggregato.csv")
    print(df_agg.head(10).to_string(index=False))
elif choice == 6:
    # ---------------------------------------------------------------
    # Genera 4 varianti casuali di BaselineA e 4 di BaselineB (+ originali)
    # ---------------------------------------------------------------
    out_dir = (out_base / TEMPLATE_DIR[6])
    out_dir.mkdir(parents=True, exist_ok=True)
    def generate_random_variants(base, n=4, delta=0.15):
        import random, copy
        out = []
        for _ in range(n):
            var = {k: max(v * random.uniform(1-delta, 1+delta), 0) for k, v in base.items()}
            tot = sum(var.values())
            for k in var:              # rinormalizza
                var[k] /= tot
            out.append(var)
        return out

    variants = {}
    for i, w in enumerate(generate_random_variants(weights_baselineA, 4), 1):
        variants[f"A_var{i}"] = w
    for i, w in enumerate(generate_random_variants(weights_baselineB, 4), 1):
        variants[f"B_var{i}"] = w
    variants["BaselineA"] = weights_baselineA
    variants["BaselineB"] = weights_baselineB

    print("→ Lancio 10 varianti casuali più i 2 baseline…")

    all_scores = {}
    for tag, wdict in variants.items():
        score = calc_scores(df, wdict)
        all_scores[tag] = score
        df_tmp = pd.DataFrame({"Nome Paese": df["Nome paese"], "TotalScore": score})
        df_tmp["Rank"] = (df_tmp["TotalScore"]
                          .rank(method="min", ascending=False, na_option="bottom")
                          .astype("Int64"))
        df_tmp = df_tmp.sort_values("TotalScore", ascending=False).reset_index(drop=True)
        df_tmp.to_csv(out_dir / f"Ranking_{tag}.csv", index=False)

    # aggregato sulle 12 corse
    m = pd.DataFrame(all_scores).mean(axis=1)
    df_agg = pd.DataFrame({"Nome Paese": df["Nome paese"], "PunteggioMedio": m})
    df_agg["Rank"] = (df_agg["PunteggioMedio"]
                      .rank(method="min", ascending=False, na_option="bottom")
                      .astype("Int64"))
    df_agg = df_agg.sort_values("PunteggioMedio", ascending=False).reset_index(drop=True)
    df_agg.to_csv(out_dir / "Ranking_Varianti_Aggregato.csv", index=False)
    print("✅  Generati Ranking_*.csv + Ranking_Varianti_Aggregato.csv")
    # ---- Riepilogo di robustezza -------------------------------------------
    ranks = pd.DataFrame(all_scores).rank(method='min', ascending=False)
    ranks.index = df['Nome paese'].values
    stability = pd.DataFrame({
    'min_rank'  : ranks.min(axis=1),
    'max_rank'  : ranks.max(axis=1),
    'top3_freq' : (ranks <= 3).mean(axis=1) * 100,
    'top10_freq': (ranks <= 10).mean(axis=1) * 100
    })

    stable   = stability[stability['max_rank'] - stability['min_rank'] <= 2]
    volatile = stability[stability['max_rank'] - stability['min_rank'] >= 6]

    print("\n=== Riepilogo robustezza ===")
    print("Paesi sempre in Top 3 (≥80 % varianti):",
      ", ".join(stability[stability['top3_freq'] >= 80].index))
    print("Paesi stabili (oscillano ≤2 pos.):",
      ", ".join(stability[stability['max_rank'] - stability['min_rank'] <= 2].index))
    print("Paesi instabili (oscillano ≥6 pos.):",
      ", ".join(stability[stability['max_rank'] - stability['min_rank'] >= 6].index))
    best = stability.sort_values('top3_freq', ascending=False).iloc[0]
    print(f" - {best.name} resta al 1° posto nel {best.top3_freq:.0f}% delle varianti "
      f"(range {int(best.min_rank)}–{int(best.max_rank)})")
    print("====================================================\n")
    report_lines = []
    report_lines.append("=== Riepilogo robustezza ===")
    report_lines.append("Paesi sempre in Top 3 (≥80 % varianti): " +
                    ", ".join(stability[stability['top3_freq'] >= 80].index))
    report_lines.append("Paesi stabili (oscillano ≤2 pos.): " +
                    ", ".join(stability[stability['max_rank'] - stability['min_rank'] <= 2].index))
    report_lines.append("Paesi instabili (oscillano ≥6 pos.): " +
                    ", ".join(stability[stability['max_rank'] - stability['min_rank'] >= 6].index))
    report_lines.append(f"{best.name} resta al 1° posto nel {best.top3_freq:.0f}% "
                    f"delle varianti (range {int(best.min_rank)}–{int(best.max_rank)})")
    (out_dir / "report_robustezza.txt").write_text("\n".join(report_lines), encoding="utf-8")
    sys.exit(0)          # chiude dopo aver mostrato il riepilogo
else:
    print("Scelta non valida. Uscita.")