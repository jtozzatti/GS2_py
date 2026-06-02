"""
data_prep.py
============
Projeto GS - Prevenção de Desastres Naturais: Incêndios Florestais
Dataset: NASA FIRMS (Fire Information for Resource Management System)

Responsabilidade:
  - Carregar o CSV bruto da NASA FIRMS
  - Tratar valores nulos e duplicatas
  - Feature engineering (mês, estação, nível de risco, período do dia)
  - Normalização e encoding
  - Salvar dataset limpo em data/fires_clean.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# ─────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────
RAW_PATH   = "data/fires_raw.csv"
CLEAN_PATH = "data/fires_clean.csv"


# ─────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Carrega o CSV da NASA FIRMS."""
    print(f"[1/6] Carregando dataset: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"      Registros carregados: {len(df):,}")
    print(f"      Colunas: {list(df.columns)}\n")
    return df


# ─────────────────────────────────────────
# 2. LIMPEZA
# ─────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e trata nulos."""
    print(f"[2/6] Limpeza dos dados")

    # Duplicatas
    before = len(df)
    df = df.drop_duplicates()
    print(f"      Duplicatas removidas: {before - len(df)}")

    # Nulos por coluna
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        print(f"      Colunas com nulos:\n{nulos}")
    else:
        print("      Nenhum valor nulo encontrado.")

    # Preencher nulos numéricos com mediana
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Preencher nulos categóricos com moda
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"      Registros após limpeza: {len(df):,}\n")
    return df


# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas features relevantes para prevenção de incêndios."""
    print("[3/6] Feature engineering")

    # Detecta coluna de data (NASA FIRMS usa 'acq_date')
    date_col = None
    for c in ["acq_date", "date", "datetime"]:
        if c in df.columns:
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["mes"]       = df[date_col].dt.month
        df["ano"]       = df[date_col].dt.year
        df["dia_semana"] = df[date_col].dt.dayofweek  # 0=segunda, 6=domingo

        # Estação do ano (hemisfério sul)
        def estacao(mes):
            if mes in [12, 1, 2]:  return "Verão"
            elif mes in [3, 4, 5]: return "Outono"
            elif mes in [6, 7, 8]: return "Inverno"
            else:                  return "Primavera"

        df["estacao"] = df["mes"].apply(estacao)
        print("      Features de data criadas: mes, ano, dia_semana, estacao")

    # Período do dia a partir de 'acq_time' (NASA FIRMS: HHMM inteiro)
    if "acq_time" in df.columns:
        df["hora"] = df["acq_time"].astype(str).str.zfill(4).str[:2].astype(int)

        def periodo(h):
            if 6 <= h < 12:  return "Manhã"
            elif 12 <= h < 18: return "Tarde"
            elif 18 <= h < 24: return "Noite"
            else:              return "Madrugada"

        df["periodo_dia"] = df["hora"].apply(periodo)
        print("      Feature de período criada: periodo_dia")

    # Nível de risco baseado em brightness (temperatura de brilho do satélite)
    # brightness > 370K = alto risco; 340-370K = médio; < 340K = baixo
    if "brightness" in df.columns:
        def nivel_risco(b):
            if b >= 370:   return "Alto"
            elif b >= 340: return "Médio"
            else:          return "Baixo"

        df["nivel_risco"] = df["brightness"].apply(nivel_risco)
        print("      Feature de risco criada: nivel_risco (Baixo/Médio/Alto)")

    # Alternativa: usar FRP (Fire Radiative Power) se disponível
    elif "frp" in df.columns:
        q33 = df["frp"].quantile(0.33)
        q66 = df["frp"].quantile(0.66)

        def nivel_risco_frp(f):
            if f >= q66:   return "Alto"
            elif f >= q33: return "Médio"
            else:          return "Baixo"

        df["nivel_risco"] = df["frp"].apply(nivel_risco_frp)
        print("      Feature de risco criada via FRP: nivel_risco")

    print()
    return df


# ─────────────────────────────────────────
# 4. ENCODING
# ─────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica Label Encoding em variáveis categóricas."""
    print("[4/6] Encoding de variáveis categóricas")

    cat_cols = ["estacao", "periodo_dia", "nivel_risco", "daynight", "satellite"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    le = LabelEncoder()
    for col in cat_cols:
        encoded_col = col + "_enc"
        df[encoded_col] = le.fit_transform(df[col].astype(str))
        print(f"      {col} → {encoded_col}  | Classes: {list(le.classes_)}")

    print()
    return df


# ─────────────────────────────────────────
# 5. NORMALIZAÇÃO
# ─────────────────────────────────────────
def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas numéricas para o intervalo [0, 1]."""
    print("[5/6] Normalização Min-Max")

    # Colunas da NASA FIRMS mais relevantes para normalizar
    cols_para_norm = ["brightness", "frp", "bright_t31", "confidence",
                      "latitude", "longitude"]
    cols_para_norm = [c for c in cols_para_norm if c in df.columns]

    scaler = MinMaxScaler()
    norm_names = [c + "_norm" for c in cols_para_norm]
    df[norm_names] = scaler.fit_transform(df[cols_para_norm])

    print(f"      Colunas normalizadas: {cols_para_norm}\n")
    return df


# ─────────────────────────────────────────
# 6. SALVAR
# ─────────────────────────────────────────
def save_data(df: pd.DataFrame, path: str) -> None:
    """Salva o dataset limpo e preparado."""
    print(f"[6/6] Salvando dataset limpo em: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"      ✓ {len(df):,} registros salvos.")
    print(f"      ✓ {len(df.columns)} colunas no total.")
    print(f"\n      Colunas finais:\n      {list(df.columns)}")


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────
def run_pipeline():
    print("=" * 55)
    print("  GS PYTHON - Prevenção de Incêndios Florestais")
    print("  Etapa: Preparação dos Dados (NASA FIRMS)")
    print("=" * 55 + "\n")

    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_features(df)
    df = normalize_features(df)
    save_data(df, CLEAN_PATH)

    print("\n" + "=" * 55)
    print("  ✓ Pipeline concluído com sucesso!")
    print("  Próximo passo: execute eda.py")
    print("=" * 55)

    return df


if __name__ == "__main__":
    df = run_pipeline()