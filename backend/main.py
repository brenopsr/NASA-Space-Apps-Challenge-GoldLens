#!/usr/bin/env python3
from flask import Flask, request, jsonify, Response
import pandas as pd
import numpy as np
import joblib, os, io, argparse, sys
from flask_cors import CORS
from typing import List, Optional

app = Flask(__name__)
CORS(app)


def _first_existing_path(candidates):
    """Return the first existing path from *candidates* (ignoring ``None``)."""
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _default_model_path():
    """Try to infer a usable model path bundled with the repository."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    candidates = [
        os.getenv("MODEL_PATH"),
        os.path.join(repo_root, "models", "rf_model.pkl"),
        os.path.join(here, "rf_300.pkl"),
        os.path.join(here, "artifacts", "modelo_final.pkl"),
        os.path.join(repo_root, "model.pkl"),
    ]
    return _first_existing_path(candidates)


def _default_features_path():
    """Infer a default features file if one is available."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    candidates = [
        os.getenv("FEATURES_PATH"),
        os.path.join(repo_root, "models", "rf_features.pkl"),
        os.path.join(here, "features.pkl"),
        os.path.join(here, "artifacts", "features.pkl"),
    ]
    return _first_existing_path(candidates)

# ---------------------------
# CLI / ENV config
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GoldLens - API de predição (Flask)")
    p.add_argument("--model", type=str, default=_default_model_path(),
                   help="Caminho do modelo .pkl (ou env MODEL_PATH)")
    p.add_argument("--features", type=str, default=_default_features_path(),
                   help="Caminho do .pkl com lista de features (opcional). Se ausente, usa feature_names_in_ do modelo.")
    p.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    p.add_argument("--debug", action="store_true", default=os.getenv("DEBUG", "0") in ("1","true","True"))
    return p.parse_args()

ARGS = None
rf = None
FEATURES: List[str] = []

# ---------------------------
# Helpers de pré-processamento
# ---------------------------
def _to_num(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype="object").astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(
        s.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0],
        errors="coerce",
    )

def _normalize_df_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = _to_num(out[c])
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def build_feature_matrix(df_in: pd.DataFrame, features: List[str], min_raw_nonnull: int = 3) -> pd.DataFrame:
    """
    1) Converte colunas para numérico onde possível
    2) Seleciona/intersecta com 'features'
    3) Imputa faltantes com mediana por coluna
    4) Descarta linhas com menos de 'min_raw_nonnull' não-nulos nas features
    """
    if not features:
        raise ValueError("Lista de FEATURES está vazia.")
    df_num = _normalize_df_numeric(df_in)
    X = df_num.reindex(columns=features)
    keep = X.notna().sum(axis=1) >= min_raw_nonnull
    X = X.loc[keep].copy()
    if len(X) == 0:
        raise ValueError(f"Nenhuma linha com informação suficiente (min_raw_nonnull={min_raw_nonnull}).")
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X

def read_payload_to_df(req) -> pd.DataFrame:
    if "file" in req.files:
        file = req.files["file"]
        name = (file.filename or "").lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(file)
        # CSV genérico
        return pd.read_csv(file, sep=",", skipinitialspace=True, on_bad_lines="skip", engine="python", comment="#")
    if req.is_json:
        payload = req.get_json()
        if isinstance(payload, dict) and "data" in payload:
            return pd.DataFrame(payload["data"])
        return pd.DataFrame(payload)
    raise ValueError("Envie um arquivo CSV/XLSX em 'file' ou JSON válido.")

def format_prob_column(out: pd.DataFrame, prob_format: str, prob_decimals: int, keep_float: bool):
    # p_planet_float ∈ [0..1]
    if prob_format == "percent":
        pct = (out["p_planet_float"] * 100).round(prob_decimals)
        out["p_planet"] = pct.map(lambda x: f"{int(x)}%" if prob_decimals == 0 else f"{x:.{prob_decimals}f}%")
    else:
        out["p_planet"] = out["p_planet_float"].round(prob_decimals)
    if not keep_float:
        out.drop(columns=["p_planet_float"], inplace=True)
    return out

# ---------------------------
# Rotas
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        fmt = (request.args.get("format") or "json").lower()                 # json|csv
        top = request.args.get("top")
        include_index = request.args.get("include_index", "0").lower() in ("1", "true")
        min_raw_nonnull = int(request.args.get("min_raw_nonnull", 3))
        prob_format = (request.args.get("prob_format") or "percent").lower() # percent|float
        prob_decimals = int(request.args.get("prob_decimals", 4))
        keep_float = request.args.get("keep_float", "0").lower() in ("1", "true")

        if rf is None or not hasattr(rf, "predict_proba"):
            return jsonify({"error": "Modelo não carregado ou sem predict_proba()."}), 500
        if not FEATURES:
            return jsonify({"error": "Lista de FEATURES vazia."}), 500

        # 1) input
        df_in = read_payload_to_df(request)

        # 2) features
        X = build_feature_matrix(df_in, FEATURES, min_raw_nonnull=min_raw_nonnull)

        # 3) prever
        X_aligned = X.reindex(columns=FEATURES, fill_value=0)
        p1 = rf.predict_proba(X_aligned)[:, 1]

        # 4) output
        out = X.copy()
        out["p_planet_float"] = pd.Series(p1, index=out.index)

        if include_index:
            out = out.reset_index(names="orig_idx")
        out = out.sort_values("p_planet_float", ascending=False)

        if top is not None:
            try:
                n = int(top)
                if n > 0:
                    out = out.head(n)
            except ValueError:
                pass

        out = format_prob_column(out, prob_format, prob_decimals, keep_float)

        keep_cols = FEATURES + (["orig_idx"] if include_index else []) + (["p_planet"] + (["p_planet_float"] if keep_float else []))
        keep_cols = [c for c in keep_cols if c in out.columns]  # só as existentes
        out = out[keep_cols]

        if fmt == "csv":
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            buf.seek(0)
            return Response(
                buf.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=predicoes.csv"},
            )
        return jsonify(out.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": f"Erro ao processar: {str(e)}"}), 400

@app.route("/metrics_summary", methods=["GET"])
def get_metrics_summary():
    try:
        model_path = ARGS.model or _default_model_path()
        if not model_path:
            return jsonify({"error": "MODEL_PATH/--model não definido."}), 400
        model_dir = os.path.dirname(os.path.abspath(model_path))
        file_path = os.path.join(model_dir, "metrics_summary.txt")
        if not os.path.exists(file_path):
            return jsonify({"error": "Resumo de métricas não encontrado."}), 404
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        return Response(content, mimetype="text/plain")
    except Exception as e:
        return jsonify({"error": f"Erro ao ler métricas: {str(e)}"}), 400

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Backend Flask ativo"})

# ---------------------------
# Bootstrap
# ---------------------------
def _load_features(features_path: Optional[str], model) -> List[str]:
    # 1) se arquivo de features foi passado, tenta carregar
    if features_path and os.path.exists(features_path):
        feats = joblib.load(features_path)
        if isinstance(feats, (list, tuple, np.ndarray)):
            return list(feats)
    # 2) senão, tenta pegar do modelo
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # 3) fallback: erro explícito
    raise ValueError("Não foi possível determinar a lista de FEATURES. Passe --features ou salve o modelo com feature_names_in_.")

def bootstrap():
    global ARGS, rf, FEATURES
    ARGS = parse_args()
    model_path = ARGS.model or _default_model_path()
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    rf = joblib.load(model_path)
    if not hasattr(rf, "predict_proba"):
        raise ValueError("O modelo carregado não possui predict_proba().")
    features_path = ARGS.features or _default_features_path()
    FEATURES = _load_features(features_path, rf)

if __name__ == "__main__":
    bootstrap()
    app.run(host=ARGS.host, port=ARGS.port, debug=ARGS.debug)
