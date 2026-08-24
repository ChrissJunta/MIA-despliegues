# Ejercicio Práctico Guiado — Semana 2/3
## Monitoreo del Modelo e IA aplicada a la Industria 4.0

Este proyecto ya viene **probado y funcionando** (fue verificado end-to-end
antes de entregártelo: entrenamiento, `/health`, `/predict`, `/monitor` con
reporte de drift de Evidently, y el dashboard Gradio en `/ui`). Solo debes
seguir estos pasos en tu computador para reproducirlo y tomar tus propias
capturas para el informe.

---

## Paso 1 — Preparar el entorno virtual

Abre una terminal dentro de la carpeta del proyecto.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sabrás que está activo porque el prompt de tu terminal mostrará `(.venv)`
al inicio. **📸 Captura 1:** este momento (creación + activación del entorno).

---

## Paso 2 — Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Esto instala `fastapi, uvicorn, scikit-learn, pandas, joblib, evidently,
gradio` y dos dependencias de compatibilidad (`plotly`, `python-multipart`,
`huggingface_hub` fijado en `0.25.2` — sin este pin, Gradio 4.x falla al
importar por un cambio de API en versiones nuevas de `huggingface_hub`).

**📸 Captura 2:** la terminal mostrando la instalación completada sin errores.

---

## Paso 3 — Generar datos y entrenar el modelo

```bash
python train.py
```

Deberías ver algo como:
```
OK -> data/datos_sensor.csv (1200 filas)
Precision holdout: 0.994
Modelo guardado -> models/modelo_sensores.pkl
```

**📸 Captura 3:** salida de la terminal con la precisión alcanzada.
**📸 Captura 4:** el archivo `data/datos_sensor.csv` abierto (Excel o editor).
**📸 Captura 5:** el archivo `models/modelo_sensores.pkl` visible en el explorador de archivos.

> Las variables simuladas son: `temperatura` (°C del sensor), `vibracion`
> (nivel normalizado de vibración de la máquina) y `retraso` (etiqueta 0/1
> que indica si el lote de producción tuvo un retraso). El modelo es un
> `RandomForestClassifier` que aprende a predecir `retraso` a partir de
> `temperatura` y `vibracion`.

---

## Paso 4 — Levantar la API localmente

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Deja esta terminal abierta (el servidor queda corriendo). Abre tu navegador en:

- Swagger (documentación interactiva): **http://127.0.0.1:8000/docs**
- Dashboard Gradio: **http://127.0.0.1:8000/ui**

**📸 Captura 6:** `/docs` en Swagger mostrando el endpoint `/health` expandido, con clic en "Try it out" → "Execute" y la respuesta `200 OK`.

**📸 Captura 7 y 8:** dos ejecuciones distintas de `/predict` en Swagger,
por ejemplo:
- Ejemplo A (proceso normal): `temperatura=70`, `vibracion=0.5` → debería devolver `prediccion: 0`
- Ejemplo B (riesgo de retraso): `temperatura=88`, `vibracion=0.95` → debería devolver `prediccion: 1`

> **Cómo se sirve el modelo:** `train.py` serializa el `RandomForestClassifier`
> entrenado con `joblib.dump()` en `models/modelo_sensores.pkl`. Al arrancar,
> `main.py` lo carga una sola vez con `joblib.load()` y lo mantiene en
> memoria; cada solicitud a `/predict` solo arma un `DataFrame` con los
> valores recibidos y llama a `modelo.predict_proba()`, sin volver a
> entrenar ni releer el disco.

---

## Paso 5 — Probar el monitoreo de drift

En Swagger, ejecuta `/monitor` dos veces con parámetros distintos:

- **Dentro de rango** (sin alerta): `size=200, t_mean=71, t_std=5, v_mean=0.5, v_std=0.1`
- **Fuera de rango** (con alerta / drift): `size=200, t_mean=92, t_std=6, v_mean=1.1, v_std=0.1`

**📸 Captura 9 y 10:** ambas ejecuciones y sus respuestas JSON (fíjate en el campo `alerts`).

Cada llamada a `/monitor` genera (o sobrescribe) `reports/reporte_drift.html`
usando Evidently (`DataDriftPreset`), comparando el dataset de referencia
(`data/datos_sensor.csv`) contra el lote "actual" simulado.

**📸 Captura 11:** abre `reports/reporte_drift.html` directamente en tu navegador (doble clic en el archivo) y captura la vista general.

> **Cómo interpretar el reporte:** la sección superior muestra si se
> detectó *dataset drift* en conjunto (comparación estadística entre la
> distribución de referencia y la actual). Debajo, por cada columna
> (`temperatura`, `vibracion`) se muestra un histograma superpuesto
> (referencia vs. actual) y un puntaje de drift; si la distribución actual
> se desplaza significativamente respecto a la de referencia, la columna
> se marca como "Drift detected". La regla adicional de `/monitor` (media
> actual > referencia + 2 desviaciones estándar) es una alerta operativa
> simple y explicable, complementaria al análisis estadístico de Evidently.

---

## Paso 6 — Probar el dashboard Gradio

Ve a **http://127.0.0.1:8000/ui**.

**📸 Captura 12:** el dashboard mostrando los sliders de temperatura/vibración y el resultado de una predicción.
**📸 Captura 13:** los histogramas de temperatura y vibración (botón "Actualizar gráficos").

> **Conexión Gradio–FastAPI:** `mount_gradio_app(app, demo, path="/ui")`
> monta la aplicación Gradio (`demo`) como una sub-aplicación dentro de la
> misma instancia de FastAPI, bajo la ruta `/ui`. Ambas comparten proceso
> y puerto: los botones de Gradio no hacen una petición HTTP al endpoint
> `/predict`, sino que llaman directamente a la función Python `predict()`
> en memoria — es una integración a nivel de código, no de red.

---

## Paso 7 — Dockerización

Detén el servidor local (`Ctrl+C`) y en la misma carpeta:

```bash
docker build -t mi-api-ml:latest .
```

**📸 Captura 14:** la terminal mostrando el build completado.

```bash
docker run --rm -p 8000:7860 mi-api-ml:latest
```

**📸 Captura 15:** la terminal con el contenedor corriendo.

Abre **http://127.0.0.1:8000/docs** en el navegador (el contenedor expone
el puerto 7860 internamente, pero lo mapeamos a 8000 en tu máquina).

**📸 Captura 16:** la API funcionando desde el navegador, ahora servida por Docker.

---

## Paso 8 — Despliegue en la nube (Render.com)

> **Nota:** la guía original proponía Hugging Face Spaces, pero HF cambió su
> política y el SDK Docker de Spaces ahora requiere un plan de pago (PRO).
> Este proyecto se despliega en **Render.com**, que sí ofrece un nivel
> gratuito real para servicios Docker.

### 8.1 Ajustar el Dockerfile

Render asigna el puerto dinámicamente mediante la variable de entorno
`PORT`. Cambia la última línea del `Dockerfile` de forma exec a forma shell:

```dockerfile
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
```

**📸 Captura 17 (a):** el `Dockerfile` editado.

### 8.2 Subir el proyecto a GitHub

```bash
git init
git add .
git commit -m "Proyecto inicial"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/proyecto-ia-industria.git
git push -u origin main
```

### 8.3 Crear el Web Service en Render

1. Crea una cuenta en [render.com](https://render.com) (puedes usar tu cuenta de GitHub).
2. **New** → **Web Service** → selecciona tu repositorio.
3. Render detecta el `Dockerfile` automáticamente.
4. Verifica que **Instance Type** esté en **Free**.
5. Clic en **Create Web Service**.

**📸 Captura 17:** el repositorio conectado y el Web Service creado en Render (pantalla como la del dashboard, con el estado del deploy).

### 8.4 Verificar el despliegue

Cuando el estado pase a **Live**, abre en el navegador:

- `https://TU-SERVICIO.onrender.com/health`
- `https://TU-SERVICIO.onrender.com/docs`
- `https://TU-SERVICIO.onrender.com/ui`

> El plan gratuito "duerme" el servicio tras ~15 minutos sin tráfico; la
> primera petición después de eso puede tardar 30–50 segundos en responder.

**📸 Captura 18:** la app funcionando en la URL pública de Render.

> **Local vs. cloud:** en local, el proceso corre sin restricciones de red
> ni gestión de recursos externos, ideal para desarrollar y depurar rápido.
> En Render, la misma imagen Docker corre en infraestructura administrada:
> URL pública con HTTPS, reinicio automático ante fallos, pero con límites
> de CPU/RAM del plan gratuito (512 MB RAM) y sin persistencia de archivos
> entre reinicios — por eso el proyecto regenera el CSV y el modelo si no
> los encuentra al arrancar.