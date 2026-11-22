"""Predice la clase del modelo de cáncer de mama con un archivo JSON."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "modelo_cancer_mama.pkl"
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]
LABELS = {0: "maligno", 1: "benigno"}


def main():
    parser = argparse.ArgumentParser(
        description="Clasifica una muestra con el Random Forest de cáncer de mama."
    )
    parser.add_argument("archivo_json", nargs="?", help="JSON con las 30 características.")
    parser.add_argument(
        "--plantilla", metavar="RUTA",
        help="Crea un JSON de ejemplo en RUTA y termina.",
    )
    args = parser.parse_args()

    if args.plantilla:
        Path(args.plantilla).write_text(
            json.dumps({name: 0.0 for name in FEATURES}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Plantilla creada: {args.plantilla}")
        return 0
    if not args.archivo_json:
        parser.error("indica un archivo JSON o usa --plantilla RUTA")

    try:
        payload = json.loads(Path(args.archivo_json).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("El JSON debe ser un objeto con nombre de característica y valor.")

        missing = [name for name in FEATURES if name not in payload]
        extra = [name for name in payload if name not in FEATURES]
        if missing or extra:
            details = []
            if missing:
                details.append("faltan: " + ", ".join(missing))
            if extra:
                details.append("no reconocidas: " + ", ".join(extra))
            raise ValueError("; ".join(details))

        row = {}
        for name in FEATURES:
            value = payload[name]
            if isinstance(value, bool):
                raise ValueError(f"'{name}' debe ser un número, no un booleano.")
            try:
                row[name] = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"'{name}' debe ser un valor numérico.") from None
            if not pd.notna(row[name]) or row[name] in (float("inf"), float("-inf")):
                raise ValueError(f"'{name}' debe ser un número finito.")

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        sample = pd.DataFrame([row], columns=FEATURES)
        prediction = int(model.predict(sample)[0])
        result = {"clase": prediction, "resultado": LABELS.get(prediction, "desconocido")}
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            result["probabilidades"] = {
                LABELS.get(int(label), str(label)): float(probability)
                for label, probability in zip(model.classes_, probabilities)
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"No se pudo cargar o ejecutar el modelo: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
