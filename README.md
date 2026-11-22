# Predicción de cáncer de mama

Clasificador educativo de Machine Learning que usa un **Random Forest** entrenado con el conjunto `load_breast_cancer` de scikit-learn. Incluye el modelo serializado, el cuaderno de entrenamiento y una interfaz de línea de comandos para clasificar una muestra desde un archivo JSON.

> [!CAUTION]
> Este proyecto es educativo y **no es un dispositivo médico ni una herramienta de diagnóstico**. No uses sus resultados para tomar decisiones clínicas. Consulta con un profesional de salud.

## Estructura

```text
.
├── modelo_cancer_mama.pkl
├── INICIAR_MODELO.bat
├── notebooks/
│   └── entrenamiento_modelo.ipynb
├── predecir.py
├── README.md
└── requirements.txt
```

## Instalación

### Inicio sencillo en Windows

Haz doble clic en `INICIAR_MODELO.bat`. El lanzador prepara un entorno Python 3.12, instala las dependencias y crea/abre `entrada.json` en el Bloc de notas. Completa las 30 mediciones y guarda el archivo; al cerrar el Bloc de notas, aparecerá la predicción en la terminal. En usos posteriores puedes editar el mismo archivo.

La primera ejecución requiere `uv` y conexión a Internet para descargar Python y los paquetes. Para usar la terminal manualmente, se requiere Python 3.9 o posterior. Instala las dependencias así:

```bash
python -m pip install -r requirements.txt
```

## Uso

Genera un archivo JSON con las 30 claves de entrada:

```bash
python predecir.py --plantilla entrada.json
```

La plantilla inicial usa `0.0` como marcador; **reemplaza cada valor** por la medición correspondiente antes de predecir. Esos ceros no representan una muestra real.

Ejecuta la clasificación:

```bash
python predecir.py entrada.json
```

La salida incluye la clase y, si el modelo las ofrece, las probabilidades estimadas:

```json
{
  "clase": 1,
  "resultado": "benigno",
  "probabilidades": {
    "maligno": 0.12,
    "benigno": 0.88
  }
}
```

Las probabilidades son las puntuaciones de probabilidad del clasificador y no constituyen probabilidades clínicas calibradas.

## Datos que espera el modelo

El archivo JSON debe ser un único objeto con **exactamente 30 valores numéricos**. Las claves respetan los nombres de las características originales del dataset; no traduzcas ni cambies mayúsculas, espacios o guiones.

| Clave JSON | Significado |
| --- | --- |
| `mean radius` | Radio medio |
| `mean texture` | Textura media |
| `mean perimeter` | Perímetro medio |
| `mean area` | Área media |
| `mean smoothness` | Suavidad media |
| `mean compactness` | Compacidad media |
| `mean concavity` | Concavidad media |
| `mean concave points` | Puntos cóncavos medios |
| `mean symmetry` | Simetría media |
| `mean fractal dimension` | Dimensión fractal media |
| `radius error` | Error estándar del radio |
| `texture error` | Error estándar de la textura |
| `perimeter error` | Error estándar del perímetro |
| `area error` | Error estándar del área |
| `smoothness error` | Error estándar de la suavidad |
| `compactness error` | Error estándar de la compacidad |
| `concavity error` | Error estándar de la concavidad |
| `concave points error` | Error estándar de puntos cóncavos |
| `symmetry error` | Error estándar de la simetría |
| `fractal dimension error` | Error estándar de la dimensión fractal |
| `worst radius` | Radio en el valor `worst` del dataset |
| `worst texture` | Textura en el valor `worst` |
| `worst perimeter` | Perímetro en el valor `worst` |
| `worst area` | Área en el valor `worst` |
| `worst smoothness` | Suavidad en el valor `worst` |
| `worst compactness` | Compacidad en el valor `worst` |
| `worst concavity` | Concavidad en el valor `worst` |
| `worst concave points` | Puntos cóncavos en el valor `worst` |
| `worst symmetry` | Simetría en el valor `worst` |
| `worst fractal dimension` | Dimensión fractal en el valor `worst` |

`worst` es el término usado por el dataset para sus valores de mayor magnitud; no se debe estimar manualmente. Los valores deben proceder de mediciones con las mismas definiciones, unidades y método que los datos de entrenamiento. El script rechaza claves faltantes, extras y valores no numéricos.

## Interpretación de clases

El cuaderno asigna las etiquetas así:

- **0 → maligno**
- **1 → benigno**

El modelo se entrenó con `RandomForestClassifier(n_estimators=100, random_state=42)` y una partición estratificada de 80 % para entrenamiento y 20 % para prueba. Una predicción no confirma ni descarta una enfermedad.

## Archivos principales

- `predecir.py`: validación de entrada y predicción desde terminal.
- `modelo_cancer_mama.pkl`: modelo entrenado.
- `notebooks/entrenamiento_modelo.ipynb`: preparación de datos, entrenamiento y evaluación.
- `requirements.txt`: dependencias de ejecución; scikit-learn está fijado a la versión usada para serializar el modelo.

### Seguridad del modelo

Los modelos serializados con `joblib`/pickle pueden ejecutar código al cargarse. Usa solo este archivo del repositorio o modelos de una fuente de confianza.
