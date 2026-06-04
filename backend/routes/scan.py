"""
/scan endpoint
==============
Pure ML-only prediction pipeline:
  1. Normalise & decode URL
  2. Extract 36 features from the URL string
  3. Run XGBoost model inference
  4. Run threat intel + forensics (in parallel)
  5. Combine into final score & return

NOTE: Dataset lookup and server-side cache are commented out.
      Uncomment the marked sections to re-enable them.
"""

import asyncio
import time
from typing import Any
from urllib.parse import urlparse
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from ml.feature_extractor import extract_features, decode_obfuscation
from services.threat_intel import run_threat_intel, combine_final_score
from services.domain_intel import get_whois_info, get_dns_records, detect_typosquatting
# from services.dataset_lookup import lookup_url  # DISABLED: using ML-only prediction

router = APIRouter(prefix='/scan', tags=['Scan'])


# ──────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    url: str = Field(..., description='The URL to analyse for phishing')


class MLResult(BaseModel):
    is_phishing: bool
    confidence: float


class ThreatScoreResponse(BaseModel):
    threat_level: str
    score: int
    reasons: list[str]


class DatasetInfo(BaseModel):
    found: bool
    source: str = ""
    label: str = ""
    confidence: float = 0.0


class ScanResponse(BaseModel):
    url: str
    cleaned_url: str
    obfuscation_found: list[str]
    ml: MLResult
    threat_intel: dict[str, Any]
    forensics: dict[str, Any]
    final: ThreatScoreResponse
    dataset_match: DatasetInfo
    source: str                     # "cache", "dataset", "ml"
    response_time_ms: float


# ──────────────────────────────────────────────────────────────
# Helper: build a quick result from a dataset match
# DISABLED: using ML-only prediction — uncomment to re-enable
# ──────────────────────────────────────────────────────────────
# def _build_dataset_result(url: str, cleaned_url: str, obfuscation_found: list[str],
#                           ds_label: str, ds_source: str, ds_confidence: float,
#                           elapsed_ms: float) -> dict:
#     """Construct a ScanResponse-compatible dict from a dataset hit."""
#     # Score semantics: 100 = safest, 0 = most dangerous
#     if ds_label in ("phishing", "malware"):
#         is_phishing = True
#         ml_confidence = ds_confidence
#         threat_level = "malicious"
#         score = 5 if ds_label == "phishing" else 0
#         reasons = [f"URL found in {ds_source} dataset as '{ds_label}' (exact match)"]
#     else:
#         # benign / genuine
#         is_phishing = False
#         ml_confidence = ds_confidence
#         threat_level = "safe"
#         score = 95 if ds_source == "iscx" else 90
#         reasons = [f"URL/domain found in {ds_source} dataset as '{ds_label}' (exact match)"]
#
#     return {
#         "url": url,
#         "cleaned_url": cleaned_url,
#         "obfuscation_found": obfuscation_found,
#         "ml": {"is_phishing": is_phishing, "confidence": round(ml_confidence, 4)},
#         "threat_intel": {},
#         "forensics": {"whois": {}, "dns": {}, "typosquatting": {}},
#         "final": {"threat_level": threat_level, "score": score, "reasons": reasons},
#         "dataset_match": {
#             "found": True,
#             "source": ds_source,
#             "label": ds_label,
#             "confidence": ds_confidence,
#         },
#         "source": "dataset",
#         "response_time_ms": round(elapsed_ms, 2),
#     }


# ──────────────────────────────────────────────────────────────
# POST /scan/
# ──────────────────────────────────────────────────────────────
@router.post('/')
async def scan_url(req: ScanRequest, request: Request):
    start = time.perf_counter()
    # cache = request.app.state.cache  # DISABLED: using ML-only prediction

    # ── 0. Normalise & decode ──
    url_to_scan = req.url.strip()
    if not url_to_scan.startswith(('http://', 'https://')):
        url_to_scan = 'http://' + url_to_scan
    obfus = decode_obfuscation(url_to_scan)
    cleaned_url: str = obfus['cleaned_url']
    obfuscation_found: list[str] = obfus['obfuscation_techniques_found']

    # ── 1. CHECK SERVER-SIDE CACHE ──
    # DISABLED: using ML-only prediction — uncomment to re-enable caching
    # cached = cache.get(cleaned_url)
    # if cached is not None:
    #     elapsed_ms = (time.perf_counter() - start) * 1000.0
    #     cached["response_time_ms"] = round(elapsed_ms, 2)
    #     cached["source"] = "cache"
    #     return cached

    # ── 2. DATASET LOOKUP (O(1) hash-set check) ──
    # DISABLED: using ML-only prediction — uncomment to re-enable dataset lookup
    # ds_match = lookup_url(cleaned_url)
    # if ds_match.found:
    #     elapsed_ms = (time.perf_counter() - start) * 1000.0
    #     result = _build_dataset_result(
    #         url=req.url,
    #         cleaned_url=cleaned_url,
    #         obfuscation_found=obfuscation_found,
    #         ds_label=ds_match.label,
    #         ds_source=ds_match.source,
    #         ds_confidence=ds_match.confidence,
    #         elapsed_ms=elapsed_ms,
    #     )
    #     # Store in cache for subsequent requests
    #     cache.put(cleaned_url, result, source="dataset")
    #     # Store in history
    #     history: list = request.app.state.scan_history
    #     history.append(result)
    #     if len(history) > 100:
    #         request.app.state.scan_history = history[-100:]
    #     return result

    # ── 3. ML INFERENCE (pure ML prediction) ──
    features: dict = extract_features(cleaned_url)
    feature_order: list[str] = request.app.state.feature_order
    aligned_values = [features.get(col, 0) for col in feature_order]

    model = request.app.state.model
    if model is not None and feature_order:
        proba = model.predict_proba([aligned_values])[0]
        ml_confidence = float(proba[1])
        is_phishing = ml_confidence >= 0.5
    else:
        ml_confidence = 0.0
        is_phishing = False

    # ── 3b. Threat intel + forensics (in parallel) ──
    parsed = urlparse(cleaned_url)
    bare_domain = (parsed.netloc or '').split('@')[-1].split(':')[0]
    if bare_domain.startswith('www.'):
        bare_domain = bare_domain[4:]

    loop = asyncio.get_event_loop()

    async def run_sync(func, *args):
        return await loop.run_in_executor(None, func, *args)

    threat_intel_result, whois_result, dns_result, typo_result = await asyncio.gather(
        asyncio.wait_for(run_threat_intel(cleaned_url), timeout=0.4),
        asyncio.wait_for(run_sync(get_whois_info, bare_domain), timeout=0.3),
        asyncio.wait_for(run_sync(get_dns_records, bare_domain), timeout=0.2),
        run_sync(detect_typosquatting, bare_domain),
        return_exceptions=True,
    )

    if isinstance(threat_intel_result, Exception):
        threat_intel_result = {'virustotal': {}, 'urlhaus': {}, 'error': str(threat_intel_result)}
    if isinstance(whois_result, Exception):
        whois_result = {'error': str(whois_result)}
    if isinstance(dns_result, Exception):
        dns_result = {'error': str(dns_result)}
    if isinstance(typo_result, Exception):
        typo_result = {'error': str(typo_result)}

    forensics_data = {'whois': whois_result, 'dns': dns_result, 'typosquatting': typo_result}
    threat_score_data = combine_final_score(ml_confidence, threat_intel_result, forensics_data)

    if whois_result.get('is_newly_registered'):
        age = whois_result.get('domain_age_days', '?')
        threat_score_data['reasons'].append(
            f'Domain is newly registered ({age} days old — phishing domains are typically < 180 days)'
        )
    if typo_result.get('is_typosquat'):
        brand = typo_result.get('original_brand', '?')
        dist = typo_result.get('edit_distance', '?')
        threat_score_data['reasons'].append(f"Possible typosquat of '{brand}' (edit distance: {dist})")

    domain_age = whois_result.get('domain_age_days')
    is_new = domain_age is not None and domain_age < 365
    if dns_result.get('available') and (not dns_result.get('has_mx_record')) and is_new:
        threat_score_data['reasons'].append(
            'Domain has no MX record and is relatively new — suspicious for a business'
        )

    if threat_score_data['threat_level'] == 'malicious':
        is_phishing = True

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    result = {
        "url": req.url,
        "cleaned_url": cleaned_url,
        "obfuscation_found": obfuscation_found,
        "ml": {"is_phishing": is_phishing, "confidence": round(ml_confidence, 4)},
        "threat_intel": threat_intel_result,
        "forensics": forensics_data,
        "final": threat_score_data,
        "dataset_match": {"found": False, "source": "", "label": "", "confidence": 0.0},
        "source": "ml",
        "response_time_ms": round(elapsed_ms, 2),
    }

    # ── 4. STORE IN CACHE ──
    # DISABLED: using ML-only prediction — uncomment to re-enable caching
    # cache.put(cleaned_url, result, source="ml")

    # ── 5. Store in history & return ──
    history: list = request.app.state.scan_history
    history.append(result)
    if len(history) > 100:
        request.app.state.scan_history = history[-100:]

    return result


# ──────────────────────────────────────────────────────────────
# GET /scan/history
# ──────────────────────────────────────────────────────────────
@router.get('/history')
async def scan_history(request: Request):
    history: list = request.app.state.scan_history
    return {'count': len(history), 'scans': history}


# ──────────────────────────────────────────────────────────────
# GET /scan/cache-stats
# DISABLED: using ML-only prediction — uncomment to re-enable
# ──────────────────────────────────────────────────────────────
# @router.get('/cache-stats')
# async def cache_stats(request: Request):
#     cache = request.app.state.cache
#     return cache.stats()


# ──────────────────────────────────────────────────────────────
# DELETE /scan/cache  — flush the server cache
# DISABLED: using ML-only prediction — uncomment to re-enable
# ──────────────────────────────────────────────────────────────
# @router.delete('/cache')
# async def clear_cache(request: Request):
#     cache = request.app.state.cache
#     count = cache.clear()
#     return {"cleared": count}