#!/usr/bin/env python3
"""
Rotina de pré‑processamento e predição para dados de exoplanetas.

Este módulo oferece uma função de alto nível para ler um arquivo
(CSV/Excel) fornecido por um usuário, identificar e normalizar as
colunas presentes, alinhar os dados às features esperadas por um
modelo RandomForest previamente treinado e calcular a probabilidade
de cada linha ser um planeta real.  O objetivo é permitir que
catálogos completos como KOIFULL, K2FULL ou TOIFULL, bem como
variantes com nomes de colunas semelhantes, possam ser processados
automaticamente sem intervenção manual.

Uso típico:

    python predict_user_data.py --input K2FULL.csv \
        --model rf_model.pkl --features rf_features.pkl \
        --output resultados.csv

Argumentos:
    --input      Caminho para o arquivo do usuário (CSV ou XLSX).
    --model      Caminho para o arquivo .pkl contendo o modelo treinado.
    --features   Caminho opcional para um .pkl com a lista de features.
                  Se omitido, tenta usar ``model.feature_names_in_``.
    --min-nonnull Número mínimo de atributos numéricos não vazios por linha
                  para que a observação seja considerada (default: 3).
    --output     Caminho opcional para salvar um CSV com os resultados.

O script identifica as colunas correspondentes às features de forma
robusta, buscando tanto por correspondência exata como parcial (por
substrings).  Valores textuais são convertidos para numéricos quando
possível e NaNs são imputados com a mediana de cada feature.  Por
fim, o modelo é utilizado para prever as probabilidades e os
resultados são retornados como DataFrame ou gravados em arquivo.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd


def _to_numeric(s: pd.Series) -> pd.Series:
    """Converte uma Series para numérico, substituindo vírgula por ponto e
    extraindo apenas números, inclusive notação científica.  Valores
    não numéricos resultam em NaN.
    """
    s = s.astype("object").astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(
        s.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0],
        errors="coerce",
    )


def _match_column(df: pd.DataFrame, feature: str) -> pd.Series:
    """Procura, de forma case-insensitive, a coluna do DataFrame que melhor
    corresponde a uma feature desejada.  A função tenta correspondência
    direta (`feature.lower() == coluna.lower()`), depois por prefixo e
    finalmente por substring.  Se nada for encontrado, retorna uma
    Series de NaNs com o mesmo comprimento do DataFrame.
    """
    f_low = feature.lower().strip()
    lower_cols = {c.lower().strip(): c for c in df.columns}
    # correspondência exata
    if f_low in lower_cols:
        return _to_numeric(df[lower_cols[f_low]])
    # busca por prefixo ou substring
    for key, orig in lower_cols.items():
        if key.startswith(f_low) or f_low in key:
            return _to_numeric(df[orig])
    # não encontrada
    return pd.Series([np.nan] * len(df), index=df.index)


def build_feature_matrix(
    df_in: pd.DataFrame, features: Iterable[str], min_raw_nonnull: int = 3
) -> pd.DataFrame:
    """Constrói uma matriz de atributos para predição.

    Para cada feature esperada pelo modelo, tenta localizar a coluna
    correspondente no DataFrame de entrada utilizando `_match_column`.
    Depois descarta as linhas que possuem menos que `min_raw_nonnull`
    valores não nulos em suas features brutas e imputa os faltantes
    restantes com a mediana por coluna.

    Parameters
    ----------
    df_in : pd.DataFrame
        DataFrame original com os dados fornecidos pelo usuário.
    features : Iterable[str]
        Lista de nomes de features que o modelo espera.
    min_raw_nonnull : int
        Número mínimo de colunas não nulas para considerar uma linha.

    Returns
    -------
    pd.DataFrame
        DataFrame numérico contendo somente as features esperadas, com
        valores NaN imputados pela mediana.
    """
    feats = list(features)
    out = pd.DataFrame(index=df_in.index)
    # construir colunas individuais
    for f in feats:
        out[f] = _match_column(df_in, f)
    # remover linhas com muitas ausências
    mask_keep = out.notna().sum(axis=1) >= min_raw_nonnull
    out = out.loc[mask_keep].copy()
    if out.empty:
        raise ValueError(
            f"Nenhuma linha com pelo menos {min_raw_nonnull} valores válidos."
        )
    # substituir inf/-inf por NaN
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    # imputação por mediana
    medians = out.median(numeric_only=True)
    for c in out.columns:
        out[c] = out[c].fillna(medians[c])
    return out


def predict_file(
    input_path: str,
    model_path: str,
    features_path: Optional[str] = None,
    min_raw_nonnull: int = 3,
    output_path: Optional[str] = None,
    prob_format: str = "percent",
    prob_decimals: int = 4,
    keep_float: bool = False,
) -> pd.DataFrame:
    """Lê um arquivo de entrada, aplica o pré‑processamento e calcula
    probabilidades usando um modelo salvo em disco.

    Parameters
    ----------
    input_path : str
        Caminho para o arquivo CSV ou XLSX contendo os dados brutos.
    model_path : str
        Caminho para o arquivo pickle com o modelo treinado
        (deve implementar ``predict_proba``).
    features_path : str, optional
        Caminho para um pickle contendo a lista de features.  Se
        None, tenta usar o atributo ``feature_names_in_`` do modelo.
    min_raw_nonnull : int, default=3
        Número mínimo de colunas não vazias para manter uma linha.
    output_path : str, optional
        Caminho para gravar o resultado em CSV.  Se None, não salva.
    prob_format : str, default="percent"
        Formato da coluna de probabilidade retornada: ``percent`` (0–100%)
        ou ``float`` (0–1).  A formatação textual é aplicada conforme
        ``prob_decimals``.
    prob_decimals : int, default=4
        Número de casas decimais ao formatar a probabilidade.
    keep_float : bool, default=False
        Se True, inclui coluna "p_planet_float" com o valor
        numérico bruto em 0–1.

    Returns
    -------
    pd.DataFrame
        DataFrame com as features utilizadas e a coluna de probabilidade
        ``p_planet`` (e opcionalmente ``p_planet_float``).
    """
    # 1) Ler arquivo de entrada (CSV/XLSX)
    name_low = input_path.lower()
    if name_low.endswith(('.xlsx', '.xls')):
        df_in = pd.read_excel(input_path)
    else:
        df_in = pd.read_csv(input_path, sep=",", skipinitialspace=True, comment="#", engine="python")
    # 2) Carregar modelo
    model = joblib.load(model_path)
    # 3) Determinar lista de features
    if features_path:
        features = joblib.load(features_path)
    elif hasattr(model, 'feature_names_in_'):
        features = list(getattr(model, 'feature_names_in_'))
    else:
        raise ValueError(
            "Não foi possível determinar as features. Forneça um arquivo de features ou utilize um modelo compatível."
        )
    # 4) Construir matriz de atributos e alinhar ordem
    X = build_feature_matrix(df_in, features, min_raw_nonnull=min_raw_nonnull)
    # Importante: reindexar para garantir ordem/colunas exatamente como o modelo espera
    X_aligned = X.reindex(columns=features, fill_value=0)
    # 5) Calcular probabilidades
    if not hasattr(model, 'predict_proba'):
        raise ValueError("O modelo fornecido não possui método predict_proba().")
    probs = model.predict_proba(X_aligned)[:, 1]
    # 6) Preparar saída
    out = X_aligned.copy()
    # colunas originais podem ser mantidas se desejado; aqui mantemos apenas as features
    # adicionar probabilidade
    out['p_planet_float'] = pd.Series(probs, index=out.index)
    # formatação da coluna de probabilidade
    if prob_format == 'percent':
        pct = (out['p_planet_float'] * 100).round(prob_decimals)
        out['p_planet'] = pct.map(lambda x: f"{int(x)}%" if prob_decimals == 0 else f"{x:.{prob_decimals}f}%")
    else:
        out['p_planet'] = out['p_planet_float'].round(prob_decimals)
    if not keep_float:
        out.drop(columns=['p_planet_float'], inplace=True)
    # 7) Classificar por probabilidade decrescente
    out = out.sort_values('p_planet_float' if keep_float else 'p_planet', ascending=False)
    # 8) Salvar, se solicitado
    if output_path:
        out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predição de exoplanetas em arquivos de usuário usando RandomForest treinado.")
    parser.add_argument("--input", required=True, help="Caminho para o CSV/XLSX de entrada.")
    parser.add_argument("--model", required=True, help="Caminho para o modelo .pkl treinado.")
    parser.add_argument("--features", help="Caminho opcional para o arquivo .pkl contendo as features.")
    parser.add_argument("--min-nonnull", type=int, default=3, help="Mínimo de valores não nulos por linha (default 3).")
    parser.add_argument("--output", help="Caminho para salvar o CSV de saída (opcional).")
    parser.add_argument("--prob-format", default="percent", choices=["percent", "float"], help="Formato da coluna de probabilidade.")
    parser.add_argument("--prob-decimals", type=int, default=4, help="Casas decimais para probabilidade (default 4).")
    parser.add_argument("--keep-float", action="store_true", help="Manter coluna de probabilidade em valor numérico.")
    args = parser.parse_args()
    result = predict_file(
        input_path=args.input,
        model_path=args.model,
        features_path=args.features,
        min_raw_nonnull=args.min_nonnull,
        output_path=args.output,
        prob_format=args.prob_format,
        prob_decimals=args.prob_decimals,
        keep_float=args.keep_float,
    )
    # Imprime primeiro registros para visualização
    print(result.head())


if __name__ == "__main__":
    main()