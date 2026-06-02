"""
model.py
========
Projeto GS - Prevenção de Desastres Naturais: Incêndios Florestais
Dataset: NASA FIRMS - MODIS 2024 Brasil

Modelos implementados:
  1. Random Forest Classifier — previsão de nível de risco (Baixo/Médio/Alto)
  2. K-Means Clustering       — agrupamento de regiões por perfil de incêndio

Métricas:
  - Classificação : accuracy, precision, recall, F1-score
  - Clustering    : silhouette score, inertia (elbow method)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────
CLEAN_PATH  = "data/fires_clean.csv"
OUTPUT_DIR  = "data/graficos"
MODEL_DIR   = "data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.labelcolor":  "white",
    "axes.titlecolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "grid.color":       "#2a2a4a",
})

RANDOM_STATE = 42


def salvar(fig, nome):
    path = os.path.join(OUTPUT_DIR, nome)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓ Gráfico salvo: {path}")
    plt.close(fig)


# ─────────────────────────────────────────
# 1. CARREGAR E PREPARAR
# ─────────────────────────────────────────
def carregar_dados():
    print(f"[1/5] Carregando dataset: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    print(f"      Registros: {len(df):,}")
    return df


def preparar_features(df):
    """Seleciona e prepara features para os modelos."""
    print("\n[2/5] Preparando features")

    # Features numéricas disponíveis no dataset
    feature_cols = ["brightness", "frp", "bright_t31", "confidence",
                    "scan", "track"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Adiciona features de engenharia se existirem
    extras = ["mes", "hora", "estacao_enc", "periodo_dia_enc", "daynight_enc"]
    for c in extras:
        if c in df.columns:
            feature_cols.append(c)

    # Target: nivel_risco
    if "nivel_risco" not in df.columns:
        print("  ERRO: coluna 'nivel_risco' não encontrada!")
        return None, None, None

    # Remove linhas com nulos nas features
    dados = df[feature_cols + ["nivel_risco"]].dropna()

    X = dados[feature_cols].values
    y_raw = dados["nivel_risco"].values

    # Encode do target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"      Features usadas ({len(feature_cols)}): {feature_cols}")
    print(f"      Classes: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    print(f"      Total de amostras: {len(X):,}")

    # Amostra para performance (500k registros é muito para RF)
    if len(X) > 100_000:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X), 100_000, replace=False)
        X, y = X[idx], y[idx]
        print(f"      Amostra aplicada: 100.000 registros para treino/teste")

    return X, y, le.classes_


# ─────────────────────────────────────────
# 2. RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────
def treinar_random_forest(X, y, classes):
    print("\n[3/5] Treinando Random Forest Classifier")

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Treino: {len(X_train):,} | Teste: {len(X_test):,}")

    # Modelo
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("      Modelo treinado!")

    # Predições
    y_pred = rf.predict(X_test)

    # Métricas
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n      ── Métricas Random Forest ──")
    print(f"      Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall   : {rec:.4f}")
    print(f"      F1-Score : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=classes, zero_division=0)}")

    # Salva modelo
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    print(f"      Modelo salvo em: {MODEL_DIR}/random_forest.pkl")

    # ── Gráfico: Matriz de Confusão ──
    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax,
                annot_kws={"color": "white"})
    ax.set_title("Matriz de Confusão — Random Forest", fontsize=13, pad=10)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    salvar(fig, "08_matriz_confusao.png")

    # ── Gráfico: Feature Importance ──
    feature_names = [
        "brightness", "frp", "bright_t31", "confidence",
        "scan", "track", "mes", "hora",
        "estacao_enc", "periodo_dia_enc", "daynight_enc"
    ][:X.shape[1]]

    importancias = pd.Series(rf.feature_importances_, index=feature_names)
    importancias = importancias.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(importancias.index, importancias.values,
                   color="#e94560", edgecolor="none")
    ax.set_title("Importância das Features — Random Forest", fontsize=13, pad=10)
    ax.set_xlabel("Importância")
    for bar, val in zip(bars, importancias.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    salvar(fig, "09_feature_importance.png")

    return rf, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────
# 3. K-MEANS CLUSTERING
# ─────────────────────────────────────────
def treinar_kmeans(df):
    print("\n[4/5] Treinando K-Means Clustering")

    # Features geográficas + intensidade para clustering
    cols_cluster = ["latitude", "longitude", "brightness", "frp"]
    cols_cluster = [c for c in cols_cluster if c in df.columns]

    dados = df[cols_cluster].dropna()

    # Amostra para performance
    if len(dados) > 50_000:
        dados = dados.sample(50_000, random_state=RANDOM_STATE)
        print(f"      Amostra: 50.000 registros para clustering")

    X_cl = dados.values

    # ── Elbow Method: encontrar K ideal ──
    print("      Calculando Elbow Method (K=2 a 8)...")
    inercias = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_cl)
        inercias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K_range, inercias, "o-", color="#e94560", linewidth=2, markersize=7,
            markerfacecolor="#f5a623")
    ax.axvline(3, color="#7ed321", linestyle="--", linewidth=1.2, label="K escolhido = 3")
    ax.set_title("Elbow Method — K-Means", fontsize=13, pad=10)
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Inércia")
    ax.legend(labelcolor="white", facecolor="#1a1a2e", edgecolor="#444")
    salvar(fig, "10_elbow_method.png")

    # ── Modelo final com K=3 ──
    K_FINAL = 3
    print(f"      Treinando com K={K_FINAL}...")
    km_final = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10)
    labels = km_final.fit_predict(X_cl)

    sil = silhouette_score(X_cl, labels, sample_size=10_000, random_state=RANDOM_STATE)
    print(f"\n      ── Métricas K-Means ──")
    print(f"      K (clusters)     : {K_FINAL}")
    print(f"      Inércia          : {km_final.inertia_:,.0f}")
    print(f"      Silhouette Score : {sil:.4f}  (0=ruim, 1=ótimo)")

    # Salva modelo
    joblib.dump(km_final, os.path.join(MODEL_DIR, "kmeans.pkl"))
    print(f"      Modelo salvo em: {MODEL_DIR}/kmeans.pkl")

    # ── Gráfico: Clusters no mapa ──
    dados_plot = dados.copy()
    dados_plot["cluster"] = labels

    fig, ax = plt.subplots(figsize=(10, 8))
    cores_cluster = ["#e94560", "#4a90e2", "#7ed321"]
    nomes_cluster = ["Cluster A — Alta Intensidade",
                     "Cluster B — Intensidade Moderada",
                     "Cluster C — Baixa Intensidade"]

    for i in range(K_FINAL):
        mask = dados_plot["cluster"] == i
        ax.scatter(
            dados_plot.loc[mask, "longitude"],
            dados_plot.loc[mask, "latitude"],
            c=cores_cluster[i], s=1.5, alpha=0.4,
            label=nomes_cluster[i]
        )

    ax.set_title(f"K-Means Clustering (K={K_FINAL}) — Focos no Brasil 2024",
                 fontsize=13, pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-75, -28)
    ax.set_ylim(-35, 6)
    legend = ax.legend(facecolor="#1a1a2e", edgecolor="#444",
                       labelcolor="white", markerscale=6)
    salvar(fig, "11_clusters_mapa.png")

    return km_final, sil


# ─────────────────────────────────────────
# 4. RESUMO FINAL
# ─────────────────────────────────────────
def exibir_resumo(metricas_rf, sil_score):
    print("\n" + "=" * 55)
    print("  RESUMO DOS MODELOS")
    print("=" * 55)
    print(f"  Random Forest Classifier:")
    print(f"    Accuracy  : {metricas_rf['accuracy']*100:.1f}%")
    print(f"    Precision : {metricas_rf['precision']:.4f}")
    print(f"    Recall    : {metricas_rf['recall']:.4f}")
    print(f"    F1-Score  : {metricas_rf['f1']:.4f}")
    print(f"\n  K-Means Clustering:")
    print(f"    Clusters        : 3")
    print(f"    Silhouette Score: {sil_score:.4f}")
    print("=" * 55)
    print("  Gráficos gerados:")
    print("    08_matriz_confusao.png")
    print("    09_feature_importance.png")
    print("    10_elbow_method.png")
    print("    11_clusters_mapa.png")
    print("=" * 55)
    print("  Próximo passo: execute src/avaliacao.py")
    print("=" * 55)


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────
def run_modelos():
    print("=" * 55)
    print("  GS PYTHON - Modelagem Inteligente")
    print("  Random Forest + K-Means Clustering")
    print("=" * 55)

    df = carregar_dados()
    X, y, classes = preparar_features(df)

    if X is None:
        print("ERRO: Não foi possível preparar as features.")
        return

    rf, metricas_rf  = treinar_random_forest(X, y, classes)
    km, sil_score    = treinar_kmeans(df)

    exibir_resumo(metricas_rf, sil_score)

    return rf, km


if __name__ == "__main__":
    run_modelos()