# App Streamlit — Analizador de imagen satelital

Instrucciones rápidas:

1. Crear un entorno e instalar dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
```

2. Ejecutar la app:

```bash
streamlit run app/streamlit_app.py
```

La app permite subir una imagen o usar una imagen por defecto desde `data/`. Tiene tres pestañas: Original, Detecciones y Estadísticas.
