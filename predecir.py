"""Predicción interactiva reutilizando una única carga del modelo."""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent / "modelo_cancer_mama.pkl"
DEFAULT_INPUT = Path(__file__).resolve().parent / "entrada.json"
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


def write_template(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({name: 0.0 for name in FEATURES}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_model():
    """Importa dependencias y deserializa el modelo una vez con indicador visual."""
    stop_spinner = threading.Event()

    def show_spinner():
        frames = itertools.cycle("|/-\\")
        while not stop_spinner.wait(0.12):
            sys.stdout.write(f"\rCargando el modelo en memoria... {next(frames)}")
            sys.stdout.flush()

    spinner = threading.Thread(target=show_spinner, daemon=True)
    print("Preparando el clasificador. Esto solo ocurre una vez por sesión...", flush=True)
    spinner.start()
    try:
        import joblib

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    finally:
        stop_spinner.set()
        spinner.join()
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
    print("Modelo cargado y listo.\n")
    return model


def read_payload(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
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
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"'{name}' debe ser un valor numérico.") from None
        if not math.isfinite(row[name]):
            raise ValueError(f"'{name}' debe ser un número finito.")
    return row


def predict(model, path):
    import pandas as pd

    sample = pd.DataFrame([read_payload(path)], columns=FEATURES)
    prediction = int(model.predict(sample)[0])
    result = {"clase": prediction, "resultado": LABELS.get(prediction, "desconocido")}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        result["probabilidades"] = {
            LABELS.get(int(label), str(label)): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }
    print("\nPredicción:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def open_for_editing(path):
    if os.name == "nt":
        try:
            subprocess.run(["notepad.exe", str(path)], check=False)
            return
        except OSError:
            pass
    print(f"Edita y guarda este archivo antes de continuar: {path}")
    input("Cuando esté listo, presiona Enter...")


def interactive_session(model):
    print("=" * 58)
    print("  PREDICCIÓN DE CÁNCER DE MAMA — SESIÓN INTERACTIVA")
    print("=" * 58)
    print("El modelo permanece cargado hasta que elijas salir.")
    print("Las predicciones son educativas y no constituyen un diagnóstico.\n")

    while True:
        print("\n1. Realizar una predicción")
        print("2. Salir")
        choice = input("Selecciona una opción [1/2]: ").strip().lower()
        if choice in {"2", "salir", "s"}:
            print("Sesión finalizada.")
            return 0
        if choice not in {"", "1", "predecir", "p"}:
            print("Opción no reconocida. Elige 1 o 2.")
            continue

        supplied_path = input(
            f"Archivo JSON [{DEFAULT_INPUT.name}] (Enter para usarlo): "
        ).strip().strip('"')
        path = Path(supplied_path).expanduser() if supplied_path else DEFAULT_INPUT
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            try:
                write_template(path)
                print(f"Se creó una plantilla con 30 campos: {path}")
                print("Completa los valores numéricos; los ceros son solo marcadores.")
                open_for_editing(path)
            except OSError as exc:
                print(f"No se pudo crear o abrir la plantilla: {exc}")
                continue
        else:
            edit = input("¿Abrir el JSON para revisarlo antes de predecir? [S/n]: ").strip().lower()
            if edit not in {"n", "no"}:
                open_for_editing(path)

        try:
            predict(model, path)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"No se pudo hacer la predicción: {exc}")
            print("Corrige el JSON y vuelve a intentarlo; el modelo sigue cargado.")
        except Exception as exc:
            print(f"Error durante la predicción: {type(exc).__name__}: {exc}")
            print("Puedes corregir la entrada o intentar otra predicción.")


def main():
    parser = argparse.ArgumentParser(
        description="Clasificador de cáncer de mama con modo interactivo."
    )
    parser.add_argument(
        "archivo_json", nargs="?", help="JSON de una muestra (modo de una sola predicción)."
    )
    parser.add_argument(
        "--plantilla", metavar="RUTA",
        help="Crea un JSON de ejemplo en RUTA y termina.",
    )
    args = parser.parse_args()

    if args.plantilla:
        try:
            path = Path(args.plantilla)
            write_template(path)
            print(f"Plantilla creada: {path.resolve()}")
            return 0
        except OSError as exc:
            print(f"No se pudo crear la plantilla: {exc}", file=sys.stderr)
            return 2

    try:
        model = load_model()
    except Exception as exc:
        print("No se pudo cargar el modelo.", file=sys.stderr)
        print(f"Detalle: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Revisa que modelo_cancer_mama.pkl esté junto a predecir.py y "
            "ejecuta el proyecto con las dependencias de requirements.txt.",
            file=sys.stderr,
        )
        return 2

    if args.archivo_json:
        try:
            predict(model, Path(args.archivo_json).expanduser())
            return 0
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Error durante la predicción: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    try:
        return interactive_session(model)
    except (EOFError, KeyboardInterrupt):
        print("\nSesión finalizada.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
