import os
import sys
import json
import time
import logging
import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from routes import scan, health
# from services.dataset_lookup import load_datasets   # DISABLED: using ML-only prediction
# from services.cache import ResultCache               # DISABLED: using ML-only prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger('phishguard')

app = FastAPI(title='PhishGuard API', description='Real-time phishing URL detection API powered by XGBoost', version='2.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

app.mount('/static', StaticFiles(directory=os.path.join(_BACKEND_DIR, 'static')), name='static')

@app.middleware('http')
async def response_time_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers['X-Response-Time-Ms'] = f'{elapsed_ms:.2f}'
    logger.info('%s %s → %s (%.2f ms)', request.method, request.url.path, response.status_code, elapsed_ms)
    return response

@app.on_event('startup')
async def load_model():
    # ── 1. Load ML model ──
    model_path = os.path.join(_PROJECT_ROOT, 'ml', 'model.pkl')
    features_path = os.path.join(_PROJECT_ROOT, 'ml', 'features.json')
    if os.path.isfile(model_path):
        try:
            app.state.model = joblib.load(model_path)
            logger.info('Model loaded from %s', model_path)
        except Exception as exc:
            app.state.model = None
            logger.warning('Failed to load model from %s: %s — ML inference disabled.', model_path, exc)
    else:
        app.state.model = None
        logger.warning('model.pkl not found at %s — /scan will return stub responses. Run `python ml/train.py` to train the model first.', model_path)
    if os.path.isfile(features_path):
        with open(features_path, 'r') as f:
            data = json.load(f)
        app.state.feature_order = data.get('features', [])
        logger.info('Feature order loaded (%d features): %s', len(app.state.feature_order), app.state.feature_order)
    else:
        app.state.feature_order = []
        logger.warning('features.json not found at %s — feature alignment disabled.', features_path)

    # ── 2. Load datasets into memory (O(1) lookup sets) ──
    # DISABLED: using ML-only prediction — uncomment to re-enable dataset lookup
    # logger.info('Loading datasets for URL lookup...')
    # ds_stats = load_datasets()
    # logger.info('Dataset loading complete: %s', ds_stats)

    # ── 3. Initialise server-side result cache ──
    # DISABLED: using ML-only prediction — uncomment to re-enable caching
    # app.state.cache = ResultCache(ttl=3600, max_size=10_000)
    # logger.info('Server-side result cache initialised (TTL=3600s, max=10000)')

    app.state.scan_history = []

app.include_router(health.router)
app.include_router(scan.router)

@app.get('/', response_class=HTMLResponse)
async def serve_dashboard():
    template_path = os.path.join(_BACKEND_DIR, 'templates', 'index.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)