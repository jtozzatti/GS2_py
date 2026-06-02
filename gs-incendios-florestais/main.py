"""
main.py
=======
Projeto GS - Prevenção de Desastres Naturais: Incêndios Florestais
Dataset: NASA FIRMS - MODIS 2024 Brasil

Pipeline completo end-to-end:
  1. Preparação dos dados  (data_prep.py)
  2. Análise exploratória  (eda.py)
  3. Modelagem             (model.py)
  4. Avaliação e insights  (avaliacao.py)
"""

import sys
import os
import time

# Garante que src/ está no path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_prep  import run_pipeline
from eda        import run_eda
from model      import run_modelos
from avaliacao  import run_avaliacao


def separador(titulo):
    print("\n" + "█" * 55)
    print(f"  {titulo}")
    print("█" * 55 + "\n")


def main():
    inicio = time.time()

    print("\n" + "=" * 55)
    print("  GS PYTHON — FIAP 2026")
    print("  Prevenção de Desastres: Incêndios Florestais")
    print("  Dados: NASA FIRMS | MODIS 2024 | Brasil")
    print("=" * 55)

    # ── ETAPA 1 ──
    separador("ETAPA 1/4 — Preparação dos Dados")
    run_pipeline()

    # ── ETAPA 2 ──
    separador("ETAPA 2/4 — Análise Exploratória (EDA)")
    run_eda()

    # ── ETAPA 3 ──
    separador("ETAPA 3/4 — Modelagem Inteligente")
    run_modelos()

    # ── ETAPA 4 ──
    separador("ETAPA 4/4 — Avaliação e Insights")
    run_avaliacao()

    # ── CONCLUSÃO ──
    tempo = time.time() - inicio
    print("\n" + "=" * 55)
    print("  ✓ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print(f"  Tempo total: {tempo:.1f}s")
    print("\n  Entregáveis gerados:")
    print("    data/fires_clean.csv       — dataset tratado")
    print("    data/graficos/             — 14 gráficos EDA + modelos")
    print("    data/models/               — modelos treinados (.pkl)")
    print("\n  Critérios da GS cobertos:")
    print("    ✓ Problema definido (incêndios via satélite)")
    print("    ✓ Dataset real NASA FIRMS 500k+ registros")
    print("    ✓ Tratamento e feature engineering")
    print("    ✓ EDA com 7+ gráficos")
    print("    ✓ Random Forest + K-Means")
    print("    ✓ Métricas: Accuracy, F1, Silhouette")
    print("    ✓ Insights estratégicos")
    print("    ✓ Pipeline automatizado")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()