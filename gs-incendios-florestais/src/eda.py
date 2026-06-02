"""
eda.py
======
Projeto GS - Prevenção de Desastres Naturais: Incêndios Florestais
Dataset: NASA FIRMS - MODIS 2024 Brasil

Análise Exploratória de Dados (EDA)
Gráficos obrigatórios (mínimo 5):
  1. Distribuição de focos por mês (barplot)
  2. Distribuição de brightness (histograma + KDE)
  3. Correlação entre variáveis numéricas (heatmap)
  4. FRP por nível de risco (boxplot)
  5. Focos por estação do ano (barplot)
  6. Série temporal de focos por mês (lineplot)
  7. Mapa de calor geográfico - latitude x longitude (scatter)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ─────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────
CLEAN_PATH  = "data/fires_clean.csv"
OUTPUT_DIR  = "data/graficos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo visual
sns.set_theme(style="darkgrid", palette="flare")
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
    "font.family":      "sans-serif",
})

CORES = ["#e94560", "#f5a623", "#f8e71c", "#7ed321", "#4a90e2", "#9b59b6"]


def carregar_dados():
    print(f"Carregando: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    print(f"Registros: {len(df):,} | Colunas: {len(df.columns)}")
    return df


def salvar(fig, nome):
    path = os.path.join(OUTPUT_DIR, nome)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓ Salvo: {path}")
    plt.close(fig)


# ─────────────────────────────────────────
# GRÁFICO 1 — Focos por Mês (Barplot)
# ─────────────────────────────────────────
def grafico_focos_por_mes(df):
    print("\n[1/7] Focos de incêndio por mês")
    if "mes" not in df.columns:
        print("  Coluna 'mes' não encontrada, pulando.")
        return

    contagem = df["mes"].value_counts().sort_index()
    meses_nomes = ["Jan","Fev","Mar","Abr","Mai","Jun",
                   "Jul","Ago","Set","Out","Nov","Dez"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(contagem.index, contagem.values,
                  color=CORES[0], edgecolor="#ff6b8a", linewidth=0.5)

    # Destaca o pico
    pico = contagem.idxmax()
    bars[pico - 1].set_color(CORES[1])
    bars[pico - 1].set_edgecolor("#ffcc00")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(meses_nomes)
    ax.set_title("Focos de Incêndio por Mês — Brasil 2024", fontsize=14, pad=12)
    ax.set_xlabel("Mês")
    ax.set_ylabel("Número de Focos")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Anotação no pico
    ax.annotate(f"Pico: {meses_nomes[pico-1]}\n{contagem[pico]:,} focos",
                xy=(pico, contagem[pico]),
                xytext=(pico + 1.5, contagem[pico] * 0.95),
                arrowprops=dict(arrowstyle="->", color="white"),
                color="white", fontsize=9)

    salvar(fig, "01_focos_por_mes.png")


# ─────────────────────────────────────────
# GRÁFICO 2 — Distribuição de Brightness
# ─────────────────────────────────────────
def grafico_brightness(df):
    print("[2/7] Distribuição de Brightness (temperatura de brilho)")
    if "brightness" not in df.columns:
        print("  Coluna 'brightness' não encontrada, pulando.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["brightness"].dropna(), bins=80, color=CORES[0],
            edgecolor="none", alpha=0.7, density=True, label="Histograma")

    # KDE sobreposto
    from scipy.stats import gaussian_kde
    dados = df["brightness"].dropna().values
    kde = gaussian_kde(dados[::50])  # amostra para performance
    x = np.linspace(dados.min(), dados.max(), 300)
    ax.plot(x, kde(x), color=CORES[1], linewidth=2, label="KDE")

    # Linhas de threshold de risco
    ax.axvline(340, color=CORES[2], linestyle="--", linewidth=1.2, label="Risco Médio (340K)")
    ax.axvline(370, color=CORES[3], linestyle="--", linewidth=1.2, label="Risco Alto (370K)")

    ax.set_title("Distribuição de Temperatura de Brilho (Brightness)", fontsize=14, pad=12)
    ax.set_xlabel("Brightness (Kelvin)")
    ax.set_ylabel("Densidade")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    salvar(fig, "02_distribuicao_brightness.png")


# ─────────────────────────────────────────
# GRÁFICO 3 — Heatmap de Correlação
# ─────────────────────────────────────────
def grafico_correlacao(df):
    print("[3/7] Heatmap de correlação entre variáveis numéricas")

    cols = ["brightness", "frp", "bright_t31", "confidence", "scan", "track"]
    cols = [c for c in cols if c in df.columns]

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#2a2a4a",
                ax=ax, cbar_kws={"shrink": 0.8})

    ax.set_title("Correlação entre Variáveis — NASA FIRMS", fontsize=14, pad=12)
    ax.set_facecolor("#16213e")

    salvar(fig, "03_heatmap_correlacao.png")


# ─────────────────────────────────────────
# GRÁFICO 4 — FRP por Nível de Risco (Boxplot)
# ─────────────────────────────────────────
def grafico_boxplot_frp(df):
    print("[4/7] Boxplot de FRP por nível de risco")

    if "frp" not in df.columns or "nivel_risco" not in df.columns:
        print("  Colunas necessárias não encontradas, pulando.")
        return

    ordem = ["Baixo", "Médio", "Alto"]
    ordem = [o for o in ordem if o in df["nivel_risco"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="nivel_risco", y="frp",
                order=ordem, palette=["#4a90e2", "#f5a623", "#e94560"],
                flierprops=dict(marker=".", color="white", alpha=0.3, markersize=2),
                ax=ax)

    ax.set_title("Fire Radiative Power (FRP) por Nível de Risco", fontsize=14, pad=12)
    ax.set_xlabel("Nível de Risco")
    ax.set_ylabel("FRP (MW)")
    ax.set_ylim(0, df["frp"].quantile(0.97))  # remove outliers extremos do eixo

    salvar(fig, "04_boxplot_frp_risco.png")


# ─────────────────────────────────────────
# GRÁFICO 5 — Focos por Estação do Ano
# ─────────────────────────────────────────
def grafico_estacao(df):
    print("[5/7] Focos por estação do ano")

    if "estacao" not in df.columns:
        print("  Coluna 'estacao' não encontrada, pulando.")
        return

    ordem = ["Verão", "Outono", "Inverno", "Primavera"]
    ordem = [e for e in ordem if e in df["estacao"].unique()]
    contagem = df["estacao"].value_counts().reindex(ordem).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(contagem.index, contagem.values,
                  color=["#f5a623", "#e94560", "#4a90e2", "#7ed321"],
                  edgecolor="none", width=0.5)

    for bar, val in zip(bars, contagem.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + contagem.max() * 0.01,
                f"{int(val):,}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Focos de Incêndio por Estação do Ano — Brasil 2024", fontsize=14, pad=12)
    ax.set_xlabel("Estação")
    ax.set_ylabel("Número de Focos")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    salvar(fig, "05_focos_por_estacao.png")


# ─────────────────────────────────────────
# GRÁFICO 6 — Série Temporal (Lineplot)
# ─────────────────────────────────────────
def grafico_serie_temporal(df):
    print("[6/7] Série temporal de focos por mês")

    if "acq_date" not in df.columns:
        print("  Coluna 'acq_date' não encontrada, pulando.")
        return

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    serie = df.groupby(df["acq_date"].dt.to_period("M")).size()
    serie.index = serie.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(serie.index, serie.values, alpha=0.3, color=CORES[0])
    ax.plot(serie.index, serie.values, color=CORES[0], linewidth=2, marker="o",
            markersize=5, markerfacecolor=CORES[1])

    # Marca o pico
    idx_pico = serie.idxmax()
    ax.annotate(f"Pico: {idx_pico.strftime('%b/%Y')}\n{serie[idx_pico]:,} focos",
                xy=(idx_pico, serie[idx_pico]),
                xytext=(idx_pico, serie[idx_pico] * 0.75),
                arrowprops=dict(arrowstyle="->", color="white"),
                color="white", fontsize=9, ha="center")

    ax.set_title("Série Temporal — Focos de Incêndio no Brasil 2024", fontsize=14, pad=12)
    ax.set_xlabel("Mês")
    ax.set_ylabel("Número de Focos")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    salvar(fig, "06_serie_temporal.png")


# ─────────────────────────────────────────
# GRÁFICO 7 — Mapa Geográfico de Focos
# ─────────────────────────────────────────
def grafico_mapa(df):
    print("[7/7] Mapa geográfico de focos (lat x lon)")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("  Colunas de coordenadas não encontradas, pulando.")
        return

    # Amostra para não travar (dataset tem 500k+ registros)
    amostra = df.sample(min(30_000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 9))

    sc = ax.scatter(
        amostra["longitude"], amostra["latitude"],
        c=amostra["brightness"] if "brightness" in amostra.columns else CORES[0],
        cmap="YlOrRd", alpha=0.4, s=1.5, linewidths=0
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Brightness (K)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Distribuição Geográfica dos Focos — Brasil 2024\n(amostra 30k pontos)",
                 fontsize=13, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Contorno aproximado do Brasil
    ax.set_xlim(-75, -28)
    ax.set_ylim(-35, 6)

    salvar(fig, "07_mapa_geografico.png")


# ─────────────────────────────────────────
# PIPELINE EDA
# ─────────────────────────────────────────
def run_eda():
    print("=" * 55)
    print("  GS PYTHON - Análise Exploratória (EDA)")
    print("  Dataset: NASA FIRMS - MODIS 2024 Brasil")
    print("=" * 55 + "\n")

    df = carregar_dados()

    print(f"\nResumo estatístico:")
    print(df[["brightness", "frp", "confidence", "bright_t31"]].describe().round(2))

    grafico_focos_por_mes(df)
    grafico_brightness(df)
    grafico_correlacao(df)
    grafico_boxplot_frp(df)
    grafico_estacao(df)
    grafico_serie_temporal(df)
    grafico_mapa(df)

    print("\n" + "=" * 55)
    print(f"  ✓ 7 gráficos salvos em: {OUTPUT_DIR}/")
    print("  Próximo passo: execute src/model.py")
    print("=" * 55)


if __name__ == "__main__":
    run_eda()