"""
main.py
-------
API REST (FastAPI) que sirve un modelo de clasificacion de sensores
industriales (temperatura, vibracion -> riesgo de retraso), incluye
monitoreo de drift con Evidently y un dashboard interactivo con Gradio
montado en /ui.

Ejecutar en local:
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health   -> estado del servicio
    POST /predict  -> prediccion puntual (temperatura, vibracion)
    POST /monitor  -> genera reporte de drift (reports/reporte_drift.html)
    GET  /ui       -> dashboard Gradio (sliders + histogramas)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# --------------------------------------------------------------------
# Rutas base y archivos
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

for d in (MODELS_DIR, DATA_DIR, REPORTS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # en HF/Docker ya se crean con permisos en el Dockerfile

CSV_PATH = DATA_DIR / "datos_sensor.csv"
MODEL_PATH = MODELS_DIR / "modelo_sensores.pkl"

# Si no hay CSV, genera uno pequeno para que los graficos no queden vacios
if not CSV_PATH.exists():
    rng = np.random.default_rng(42)
    normal = pd.DataFrame({
        "temperatura": rng.normal(70, 5, 300),
        "vibracion": rng.normal(0.5, 0.1, 300),
        "retraso": 0,
    })
    falla = pd.DataFrame({
        "temperatura": rng.normal(85, 5, 60),
        "vibracion": rng.normal(0.9, 0.1, 60),
        "retraso": 1,
    })
    df0 = pd.concat([normal, falla], ignore_index=True)
    df0.to_csv(CSV_PATH, index=False)

# --------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------
if not MODEL_PATH.exists():
    # Entrena algo minimo si el .pkl no llego
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(CSV_PATH)
    X = df[["temperatura", "vibracion"]]
    y = df["retraso"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
    joblib.dump(clf, MODEL_PATH)

modelo = joblib.load(MODEL_PATH)

# --------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------
app = FastAPI(title="API Sensores + Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "csv": CSV_PATH.exists()}


@app.post("/predict")
def predict(temperatura: float, vibracion: float):
    X = pd.DataFrame([[temperatura, vibracion]], columns=["temperatura", "vibracion"])
    proba = float(modelo.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)
    return {"prediccion": pred, "prob_retraso": round(proba, 4)}


@app.post("/monitor")
def monitor(size: int, t_mean: float, t_std: float, v_mean: float, v_std: float):
    """
    Genera un lote de datos 'actuales' simulados y los compara contra el
    dataset de referencia (data/datos_sensor.csv) usando Evidently para
    detectar drift. Guarda un reporte HTML navegable en
    reports/reporte_drift.html y devuelve un resumen con alertas simples
    basadas en la regla de 2 sigmas.
    """
    rng = np.random.default_rng(7)
    curr = pd.DataFrame({
        "temperatura": rng.normal(t_mean, t_std, size),
        "vibracion": rng.normal(v_mean, v_std, size),
    })
    ref = pd.read_csv(CSV_PATH)[["temperatura", "vibracion"]]

    # --- Regla simple: media actual > 2 sigmas sobre la referencia -----
    t_upper = ref["temperatura"].mean() + 2 * ref["temperatura"].std()
    v_upper = ref["vibracion"].mean() + 2 * ref["vibracion"].std()

    resumen = {
        "limits": {
            "temp_upper": round(float(t_upper), 3),
            "vib_upper": round(float(v_upper), 3),
        },
        "current_means": {
            "temp": round(float(curr["temperatura"].mean()), 3),
            "vib": round(float(curr["vibracion"].mean()), 3),
        },
        "alerts": {
            "temp": bool(curr["temperatura"].mean() > t_upper),
            "vib": bool(curr["vibracion"].mean() > v_upper),
        },
    }

    # --- Reporte de drift con Evidently ---------------------------------
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=curr)
        report_path = REPORTS_DIR / "reporte_drift.html"
        report.save_html(str(report_path))
        resumen["reporte_html"] = str(report_path.name)
    except Exception as e:
        resumen["reporte_html_error"] = str(e)

    return resumen


# --------------------------------------------------------------------
# Dashboard (Gradio)
# --------------------------------------------------------------------
import gradio as gr
import plotly.express as px
from gradio.routes import mount_gradio_app


def make_dashboard():
    def _predecir_ui(t, v):
        r = predict(temperatura=t, vibracion=v)
        return f"{'Retraso' if r['prediccion'] == 1 else 'Sin retraso'} (p={r['prob_retraso']})"

    def _monitor_ui(sz, tm, ts, vm, vs):
        r = monitor(sz, tm, ts, vm, vs)
        a = r["alerts"]
        return f"Temp>{a['temp']} | Vib>{a['vib']} | limites={r['limits']} actuales={r['current_means']}"

    def _graficos():
        df = pd.read_csv(CSV_PATH)
        f1 = px.histogram(df, x="temperatura", nbins=40, title="Hist. temperatura")
        f2 = px.histogram(df, x="vibracion", nbins=40, title="Hist. vibracion")
        return f1, f2

    with gr.Blocks(title="Panel de Sensores", analytics_enabled=False) as demo:
        gr.Markdown("## Panel de Sensores")
        with gr.Row():
            with gr.Column():
                temperatura = gr.Slider(50, 110, value=75, step=1, label="Temperatura")
                vibracion = gr.Slider(0.2, 1.5, value=0.8, step=0.01, label="Vibracion")
                salida_pred = gr.Label(label="Resultado")
                gr.Button("Predecir").click(
                    _predecir_ui, inputs=[temperatura, vibracion], outputs=salida_pred
                )
            with gr.Column():
                size = gr.Slider(50, 1000, value=200, step=10, label="Tamano batch")
                t_mean = gr.Slider(60, 110, value=90, step=1, label="Temp. media batch")
                t_std = gr.Slider(1, 10, value=5, step=0.5, label="Temp. std batch")
                v_mean = gr.Slider(0.3, 1.5, value=1.0, step=0.05, label="Vib. media batch")
                v_std = gr.Slider(0.05, 0.5, value=0.1, step=0.01, label="Vib. std batch")
                salida_mon = gr.Label(label="Resumen drift")
                gr.Button("Calcular drift y guardar reporte").click(
                    _monitor_ui,
                    inputs=[size, t_mean, t_std, v_mean, v_std],
                    outputs=salida_mon,
                )
        gr.Markdown("## Distribuciones historicas")
        fig_t = gr.Plot()
        fig_v = gr.Plot()
        demo.load(_graficos, None, [fig_t, fig_v])
        gr.Button("Actualizar graficos").click(_graficos, None, [fig_t, fig_v])
    return demo


demo = make_dashboard()
app = mount_gradio_app(app, demo, path="/ui")
