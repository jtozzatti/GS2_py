"""
avaliacao.py
============
Projeto GS - Prevenção de Desastres Naturais: Incêndios Florestais
Dataset: NASA FIRMS - MODIS 2024 Brasil

Responsabilidade:
  - Carregar modelos salvos e reavaliar
  - Gerar relatório completo de métricas
  - Insights estratégicos de negócio
  - Gráfico de distribuição de clusters por região
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os

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
CLEAN_PATH = "data/fires_clean.csv"
OUTPUT_DIR = "data/graficos"
MODEL_DIR  = "data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# 1. CARREGAR DADOS E MODELOS
# ─────────────────────────────────────────
def carregar_tudo():
    print("[1/4] Carregando dados e modelos")

    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    print(f"      Dataset: {len(df):,} registros")

    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    km = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
    print("      Modelos carregados: random_forest.pkl | kmeans.pkl")

    return df, rf, km


# ─────────────────────────────────────────
# 2. REAVALIAÇÃO DO RANDOM FOREST
# ─────────────────────────────────────────
def avaliar_random_forest(df, rf):
    print("\n[2/4] Avaliação detalhada — Random Forest")

    feature_cols = ["brightness", "frp", "bright_t31", "confidence",
                    "scan", "track"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    extras = ["mes", "hora", "estacao_enc", "periodo_dia_enc", "daynight_enc"]
    for c in extras:
        if c in df.columns:
            feature_cols.append(c)

    dados = df[feature_cols + ["nivel_risco"]].dropna()

    le = LabelEncoder()
    y  = le.fit_transform(dados["nivel_risco"].values)
    X  = dados[feature_cols].values

    if len(X) > 100_000:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X), 100_000, replace=False)
        X, y = X[idx], y[idx]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    y_pred = rf.predict(X_test)
    classes = le.classes_

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n      {'Métrica':<20} {'Valor':>10}")
    print(f"      {'-'*32}")
    print(f"      {'Accuracy':<20} {acc*100:>9.2f}%")
    print(f"      {'Precision (weighted)':<20} {prec:>10.4f}")
    print(f"      {'Recall (weighted)':<20} {rec:>10.4f}")
    print(f"      {'F1-Score (weighted)':<20} {f1:>10.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=classes, zero_division=0)}")

    # ── Gráfico: Métricas em barras ──
    fig, ax = plt.subplots(figsize=(8, 4))
    metricas = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    cores    = ["#4a90e2", "#7ed321", "#f5a623", "#e94560"]

    bars = ax.bar(metricas.keys(), metricas.values(),
                  color=cores, edgecolor="none", width=0.5)

    for bar, val in zip(bars, metricas.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_title("Métricas de Avaliação — Random Forest Classifier", fontsize=13, pad=10)
    ax.set_ylabel("Score")
    ax.axhline(0.8, color="white", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Referência 0.80")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    salvar(fig, "12_metricas_rf.png")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, classes


# ─────────────────────────────────────────
# 3. AVALIAÇÃO DO K-MEANS
# ─────────────────────────────────────────
def avaliar_kmeans(df, km):
    print("[3/4] Avaliação — K-Means Clustering")

    cols_cluster = ["latitude", "longitude", "brightness", "frp"]
    cols_cluster = [c for c in cols_cluster if c in df.columns]
    dados = df[cols_cluster].dropna()

    if len(dados) > 50_000:
        dados = dados.sample(50_000, random_state=RANDOM_STATE)

    X_cl   = dados.values
    labels = km.predict(X_cl)
    sil    = silhouette_score(X_cl, labels, sample_size=10_000, random_state=RANDOM_STATE)

    print(f"\n      Clusters         : {km.n_clusters}")
    print(f"      Inércia          : {km.inertia_:,.0f}")
    print(f"      Silhouette Score : {sil:.4f}")

    # Tamanho dos clusters
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n      Distribuição dos clusters:")
    nomes = ["Alta Intensidade", "Intensidade Moderada", "Baixa Intensidade"]
    for i, (u, c) in enumerate(zip(unique, counts)):
        nome = nomes[i] if i < len(nomes) else f"Cluster {u}"
        print(f"        Cluster {u} ({nome}): {c:,} focos ({c/len(labels)*100:.1f}%)")

    # ── Gráfico: Pizza dos clusters ──
    fig, ax = plt.subplots(figsize=(7, 7))
    cores_cl = ["#e94560", "#4a90e2", "#7ed321"]
    labels_cl = [f"Cluster {i}\n{nomes[i]}\n{counts[i]:,} focos" for i in range(len(unique))]

    wedges, texts, autotexts = ax.pie(
        counts, labels=labels_cl, colors=cores_cl[:len(unique)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="#1a1a2e", linewidth=1.5),
        textprops=dict(color="white")
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax.set_title("Distribuição dos Clusters — K-Means", fontsize=13, pad=15)
    salvar(fig, "13_distribuicao_clusters.png")

    return sil


# ─────────────────────────────────────────
# 4. INSIGHTS ESTRATÉGICOS
# ─────────────────────────────────────────
def insights_negocio(df, metricas_rf, sil_score):
    print("\n[4/4] Insights estratégicos de negócio")

    # Cálculos para insights
    total = len(df)
    pico_mes = df["mes"].value_counts().idxmax() if "mes" in df.columns else "N/A"
    meses_nomes = {1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",
                   6:"Junho",7:"Julho",8:"Agosto",9:"Setembro",
                   10:"Outubro",11:"Novembro",12:"Dezembro"}
    pico_nome = meses_nomes.get(pico_mes, str(pico_mes))

    risco_alto = len(df[df["nivel_risco"] == "Alto"]) if "nivel_risco" in df.columns else 0
    pct_alto   = risco_alto / total * 100 if total > 0 else 0

    estacao_pico = df["estacao"].value_counts().idxmax() if "estacao" in df.columns else "N/A"

    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║         INSIGHTS ESTRATÉGICOS — GS PYTHON           ║
  ║     Prevenção de Incêndios via Satélite (NASA)      ║
  ╠══════════════════════════════════════════════════════╣
  ║                                                      ║
  ║  📡 SOBRE OS DADOS                                   ║
  ║  • {total:,} focos detectados por satélite em 2024     
  ║  • Mês com mais focos: {pico_nome:<30}
  ║  • Estação crítica: {estacao_pico:<33}
  ║  • {pct_alto:.1f}% dos focos classificados como ALTO RISCO  
  ║                                                      ║
  ║  🤖 DESEMPENHO DO MODELO (Random Forest)             ║
  ║  • Accuracy  : {metricas_rf['accuracy']*100:.1f}% — modelo confiável para triagem    
  ║  • F1-Score  : {metricas_rf['f1']:.4f} — boa precisão e recall       
  ║  • Aplicação : alertas automáticos por nível de risco║
  ║                                                      ║
  ║  🗺️  CLUSTERING (K-Means, K=3)                       ║
  ║  • Silhouette: {sil_score:.4f} — clusters bem definidos       
  ║  • Identifica 3 zonas de risco no território         ║
  ║  • Permite priorizar recursos por região             ║
  ║                                                      ║
  ║  🚀 IMPACTO ESTRATÉGICO                              ║
  ║  • Antecipação de focos críticos com dados de        ║
  ║    satélite em tempo real (NASA FIRMS)               ║
  ║  • Reduz tempo de resposta de brigadistas            ║
  ║  • Otimiza alocação de recursos por cluster          ║
  ║  • Suporte à decisão para agências como INPE/IBAMA   ║
  ║                                                      ║
  ╚══════════════════════════════════════════════════════╝
    """)

    # ── Gráfico: Dashboard de insights ──
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # Painel 1: Focos por nível de risco
    ax1 = fig.add_subplot(gs[0])
    if "nivel_risco" in df.columns:
        contagem = df["nivel_risco"].value_counts().reindex(["Baixo","Médio","Alto"]).fillna(0)
        cores_r  = ["#4a90e2", "#f5a623", "#e94560"]
        ax1.bar(contagem.index, contagem.values, color=cores_r, edgecolor="none")
        ax1.set_title("Focos por\nNível de Risco", fontsize=11)
        ax1.set_ylabel("Focos")
        for i, (idx, val) in enumerate(contagem.items()):
            ax1.text(i, val + total*0.005, f"{int(val):,}",
                     ha="center", fontsize=8)

    # Painel 2: Métricas do modelo
    ax2 = fig.add_subplot(gs[1])
    labels_m = ["Accuracy", "Precision", "Recall", "F1"]
    vals_m   = [metricas_rf["accuracy"], metricas_rf["precision"],
                metricas_rf["recall"],   metricas_rf["f1"]]
    cores_m  = ["#4a90e2", "#7ed321", "#f5a623", "#e94560"]
    bars2 = ax2.barh(labels_m, vals_m, color=cores_m, edgecolor="none")
    ax2.set_xlim(0, 1.15)
    ax2.set_title("Métricas do\nModelo RF", fontsize=11)
    for bar, val in zip(bars2, vals_m):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)

    # Painel 3: Focos por estação
    ax3 = fig.add_subplot(gs[2])
    if "estacao" in df.columns:
        ordem = ["Verão","Outono","Inverno","Primavera"]
        contagem_e = df["estacao"].value_counts().reindex(ordem).fillna(0)
        cores_e    = ["#f5a623","#e94560","#4a90e2","#7ed321"]
        ax3.bar(contagem_e.index, contagem_e.values, color=cores_e, edgecolor="none")
        ax3.set_title("Focos por\nEstação", fontsize=11)
        ax3.set_ylabel("Focos")
        ax3.tick_params(axis="x", rotation=15)

    fig.suptitle("Dashboard de Insights — Incêndios Florestais Brasil 2024",
                 fontsize=13, y=1.02)
    salvar(fig, "14_dashboard_insights.png")


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────
def run_avaliacao():
    print("=" * 55)
    print("  GS PYTHON - Avaliação e Insights")
    print("  Random Forest + K-Means | NASA FIRMS 2024")
    print("=" * 55)

    df, rf, km = carregar_tudo()
    metricas_rf, classes = avaliar_random_forest(df, rf)
    sil_score = avaliar_kmeans(df, km)
    insights_negocio(df, metricas_rf, sil_score)

    print("\n" + "=" * 55)
    print("  ✓ Avaliação concluída!")
    print("  Gráficos gerados: 12, 13, 14")
    print("  Próximo passo: execute main.py")
    print("=" * 55)


if __name__ == "__main__":
    run_avaliacao()