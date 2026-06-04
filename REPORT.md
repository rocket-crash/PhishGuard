# PhishGuard — Project Report

## Real-Time Phishing URL Detection System

**Team:** Rocket_Crash
**Date:** May 2026

---

## 1. Introduction

PhishGuard is a real-time phishing URL detection system that combines machine learning inference with live Cyber Threat Intelligence (CTI) lookups and domain forensics. The system is designed to classify URLs as **safe**, **suspicious**, or **malicious** using a multi-layered analysis pipeline.

### Key Goals
- Achieve ≥95% classification accuracy on phishing/malicious URLs
- Deliver predictions in under 500ms end-to-end
- Provide analysts with a dashboard for visual risk assessment and IoC export

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Browser Extension                      │
│        (Chrome Extension — auto-scans on page load)      │
└───────────────────────┬─────────────────────────────────┘
                        │  POST /scan/ {url}
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 FastAPI REST Backend                      │
│                  (main.py — port 8000)                    │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │   URL Decode │  │  Feature     │  │  XGBoost       │  │
│  │   & Normalise│→ │  Extraction  │→ │  Inference     │  │
│  │              │  │  (36 feats)  │  │  (model.pkl)   │  │
│  └─────────────┘  └──────────────┘  └───────┬────────┘  │
│                                              │           │
│  ┌───────────────────────────────────────────┼────────┐  │
│  │         Parallel Enrichment (asyncio)     │        │  │
│  │  ┌─────────────┐ ┌────────┐ ┌──────────┐ │        │  │
│  │  │ VirusTotal  │ │ WHOIS  │ │   DNS    │ │        │  │
│  │  │ URLhaus API │ │ Lookup │ │ Records  │ │        │  │
│  │  └─────────────┘ └────────┘ └──────────┘ │        │  │
│  │  ┌──────────────────────────────────────┐ │        │  │
│  │  │       Typosquatting Detection        │ │        │  │
│  │  │    (Levenshtein vs Tranco Top-10K)   │ │        │  │
│  │  └──────────────────────────────────────┘ │        │  │
│  └───────────────────────────────────────────┘        │  │
│                        │                               │  │
│                        ▼                               │  │
│            ┌───────────────────────┐                   │  │
│            │   Score Combiner      │                   │  │
│            │   ML(40%) + VT(40%)   │                   │  │
│            │   + URLhaus(20%)      │                   │  │
│            │   + Forensic Penalties │                   │  │
│            └───────────┬───────────┘                   │  │
│                        │                               │  │
│                        ▼                               │  │
│              JSON Response → Client                    │  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Analyst Web Dashboard                       │
│   (Scan UI, Stats Bar, Charts, History, IoC Export)      │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Machine Learning Model

### 3.1 Datasets Used

| Dataset | Type | Records | Role |
|---------|------|---------|------|
| **ISCX-URL-2016** | Labelled URLs | 651,191 | Benign (428K), Phishing (94K), Malware (32K), Defacement (96K) |
| **PhishTank** | Verified phishing | 1,108 | Additional phishing URLs from community reports |
| **URLhaus** | Malware URLs | 12,324 | Active malware distribution URLs from abuse.ch |
| **Tranco Top-1M** | Legitimate domains | 1,000,000 | Top legitimate websites for safe-class training |

**Total combined (deduplicated):** 1,654,547 URLs
**After 1:1 class balancing:** 400,000 samples (200K safe + 200K malicious)

### 3.2 Feature Engineering (36 Features)

All features are extracted purely from the URL string — **no network requests** are made during feature extraction, ensuring sub-millisecond extraction time.

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | `url_length` | Structure | Total URL character count |
| 2 | `num_dots` | Structure | Count of '.' in URL |
| 3 | `num_hyphens` | Structure | Count of '-' in URL |
| 4 | `num_at_symbols` | Structure | Count of '@' (credential trick) |
| 5 | `num_special_chars` | Structure | Count of ?=&%#/ characters |
| 6 | `subdomain_depth` | Domain | Number of subdomain levels |
| 7 | `has_ip_address` | Security | IPv4 address instead of domain |
| 8 | `has_https` | Security | Whether URL uses HTTPS |
| 9 | `domain_length` | Domain | Length of the domain name |
| 10 | `is_common_tld` | Domain | .com/.net/.org/.edu/.gov |
| 11 | `shannon_entropy` | Statistical | Information entropy of URL string |
| 12 | `path_length` | Structure | Length of URL path |
| 13 | `path_depth` | Structure | Number of '/' in path |
| 14 | `num_digits_in_domain` | Domain | Digit count in domain |
| 15 | `digit_ratio` | Domain | Ratio of digits to domain length |
| 16 | `num_subdomains` | Domain | Number of subdomain parts |
| 17 | `has_port` | Security | Non-standard port detected |
| 18 | `has_double_slash_redirect` | Security | '//' redirect trick in path |
| 19 | `has_shortener` | Security | Known URL shortener domain |
| 20 | `vowel_consonant_ratio` | Statistical | Vowel/consonant ratio (DGA detection) |
| 21 | `brand_in_subdomain` | Phishing | Brand keyword in subdomain |
| 22 | `is_suspicious_tld` | Domain | TLD in suspicious set (.tk, .xyz, etc.) |
| 23 | `has_http_no_s` | **Phishing** | HTTP without TLS |
| 24 | `url_length_bucket` | **Phishing** | Bucketed URL length (0-3 severity) |
| 25 | `num_encoded_chars` | **Phishing** | Count of %xx encoded characters |
| 26 | `has_login_keyword` | **Phishing** | login/verify/secure in URL |
| 27 | `query_length` | **Phishing** | Length of query string |
| 28 | `num_query_params` | **Phishing** | Number of query parameters |
| 29 | `has_suspicious_extension` | **Phishing** | .exe/.php/.zip in path |
| 30 | `brand_in_domain` | **Phishing** | Brand keyword in main domain |
| 31 | `num_redirects` | **Phishing** | redirect/goto/url= patterns |
| 32 | `has_punycode` | **Phishing** | xn-- internationalized domain |
| 33 | `avg_domain_token_length` | **Statistical** | Average token length (DGA signal) |
| 34 | `tld_length` | **Domain** | Length of TLD |
| 35 | `num_underscores` | **Structure** | Underscore count (uncommon in legit) |
| 36 | `num_digits_in_url` | **Statistical** | Total digits in entire URL |

### 3.3 Model: XGBoost Classifier

**Algorithm:** XGBoost (eXtreme Gradient Boosting) — chosen for its:
- High accuracy on tabular/structured data
- Fast inference time (~0.01ms per prediction)
- Built-in regularisation to prevent overfitting
- Feature importance ranking for interpretability

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 800 | Sufficient ensemble depth |
| `max_depth` | 10 | Captures complex feature interactions |
| `learning_rate` | 0.03 | Slow learning for better generalisation |
| `reg_alpha` | 0.1 | L1 regularisation |
| `reg_lambda` | 1.5 | L2 regularisation |
| `min_child_weight` | 5 | Prevents overfitting on sparse splits |
| `subsample` | 0.8 | Row sampling for robustness |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `gamma` | 0.1 | Minimum split loss |
| `tree_method` | hist | Fast histogram-based splits |
| `early_stopping_rounds` | 30 | Prevents unnecessary iterations |

### 3.4 Evaluation Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **96.71%** |
| **Precision** | **96.67%** |
| **Recall** | **96.75%** |
| **F1 Score** | **96.71%** |

**Confusion Matrix (80,000 test samples):**

|  | Predicted Legit | Predicted Phishing |
|---|---|---|
| **Actual Legit** | TN = 38,667 | FP = 1,333 |
| **Actual Phishing** | FN = 1,301 | TP = 38,699 |

**Top Feature Importances:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `has_https` | 0.6440 |
| 2 | `has_http_no_s` | 0.1759 |
| 3 | `brand_in_subdomain` | 0.0220 |
| 4 | `domain_length` | 0.0192 |
| 5 | `num_dots` | 0.0183 |

**Inference Speed:** 0.012ms per URL (batch of 1000 in 12.4ms)

---

## 4. REST API Backend

### 4.1 Technology Stack

- **Framework:** FastAPI (Python) — async, high-performance, auto-docs
- **ML Runtime:** XGBoost + joblib serialisation
- **Server:** Uvicorn (ASGI)

### 4.2 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/scan/` | Scan a URL — returns ML prediction, CTI, forensics, final verdict |
| `GET` | `/scan/history` | Retrieve scan history (last 100 scans) |
| `GET` | `/health/` | Health check — returns `{"status": "healthy"}` |
| `GET` | `/` | Serve the analyst dashboard |

### 4.3 Scan Pipeline

Each `POST /scan/` request executes the following pipeline:

1. **URL Normalisation & Decoding** — Detects percent-encoding, `@` tricks, hex/octal IP obfuscation
2. **Feature Extraction** — Extracts 36 URL-based features (no network calls, <1ms)
3. **XGBoost Inference** — Predicts phishing probability (<0.05ms)
4. **Live CTI Lookups** (parallel via `asyncio.gather`):
   - **VirusTotal API** — Multi-engine scan (when API key configured)
   - **URLhaus API** — Known malware URL check (abuse.ch)
5. **Domain Forensics** (parallel):
   - **WHOIS Lookup** — Domain age, registrar, newly-registered flag
   - **DNS Records** — A, MX, NS record checks
   - **Typosquatting Detection** — Levenshtein distance against Tranco Top-10K
6. **Score Combination** — Weighted formula: `ML(40%) + VirusTotal(40%) + URLhaus(20%) + forensic penalties`
7. **Response** — JSON with verdict, confidence, CTI data, forensics, and response time

**Average response time:** ~400ms (dominated by network-based CTI lookups)

### 4.4 Sample Response

```json
{
  "url": "http://login-paypal.suspicious.tk/account/verify",
  "cleaned_url": "http://login-paypal.suspicious.tk/account/verify",
  "obfuscation_found": [],
  "ml": {
    "is_phishing": true,
    "confidence": 1.0
  },
  "threat_intel": { "virustotal": {}, "urlhaus": {} },
  "forensics": {
    "whois": {},
    "dns": {},
    "typosquatting": { "is_typosquat": false }
  },
  "final": {
    "threat_level": "safe",
    "score": 60,
    "reasons": ["ML model flagged as phishing (100.0% confidence)"]
  },
  "source": "ml",
  "response_time_ms": 403.68
}
```

---

## 5. Analyst Dashboard

The web-based analyst dashboard is served at `http://localhost:8000/` and provides:

### 5.1 Features

| Component | Description |
|-----------|-------------|
| **URL Scanner** | Input field with real-time scan results |
| **Stats Bar** | Live counters: Total Scans, Safe, Suspicious, Malicious, Avg Response Time |
| **Result Card** | Detailed verdict with ML confidence bar, findings, CTI tiles, obfuscation alerts |
| **Risk Distribution** | Doughnut chart (verdict proportions) + Bar chart (per-scan score timeline) |
| **Scan History** | Sortable table: #, URL, Verdict, Score, ML Confidence, Source, Time |
| **IoC Export** | Download structured reports in **JSON** or **CSV** format |

### 5.2 Design

- **Dark glassmorphism** theme with `backdrop-filter: blur()`
- **Gradient accents** using purple-violet palette
- **Micro-animations** — fade-slide-up for results, hover effects on buttons
- **Responsive** — Grid-based layout adapts to mobile/tablet/desktop
- **Chart.js** — Interactive doughnut and bar charts with tooltips

### 5.3 IoC Report Format

**JSON report** includes:
- Report metadata (timestamp, total scans)
- Summary counts (safe/suspicious/malicious)
- Per-URL indicators: URL, threat level, ML confidence, CTI results, forensics, response time

**CSV report** includes:
- One row per scanned URL with 17 columns covering threat level, scores, CTI flags, and domain forensics

---

## 6. Project Structure

```
PhisGuard/
├── ml/                          # Machine Learning
│   ├── data/                    # Training datasets
│   │   ├── iscx_url_2016.csv    # ISCX URL dataset (651K URLs)
│   │   ├── phishtank.csv        # PhishTank verified phishing
│   │   ├── urlhaus.csv          # URLhaus malware URLs
│   │   └── tranco_top1m.csv     # Tranco Top 1M legitimate
│   ├── feature_extractor.py     # 36-feature extraction engine
│   ├── train.py                 # Training pipeline (XGBoost)
│   ├── model.pkl                # Serialised trained model
│   └── features.json            # Feature column order
├── backend/                     # FastAPI REST API
│   ├── main.py                  # Application entry point
│   ├── routes/
│   │   ├── scan.py              # /scan endpoint (ML inference)
│   │   └── health.py            # /health endpoint
│   ├── services/
│   │   ├── threat_intel.py      # VirusTotal + URLhaus APIs
│   │   ├── domain_intel.py      # WHOIS, DNS, typosquatting
│   │   ├── dataset_lookup.py    # (disabled) Dataset lookup service
│   │   └── cache.py             # (disabled) Server-side LRU cache
│   ├── templates/
│   │   └── index.html           # Dashboard HTML
│   └── static/
│       ├── style.css            # Dashboard styles
│       └── app.js               # Dashboard JavaScript
├── extension/                   # Chrome browser extension
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 7. How to Run

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the ML model (optional — model.pkl is included)
cd ml && python train.py && cd ..

# 4. Start the backend server
cd backend && python main.py
```

The dashboard is available at **http://localhost:8000/**

### Optional: VirusTotal Integration
```bash
export VIRUSTOTAL_API_KEY=your_api_key_here
```

---

## 8. Conclusion

PhishGuard demonstrates a practical, production-ready approach to real-time phishing detection that combines:

1. **Fast ML inference** (0.012ms/URL) using XGBoost with 36 URL-based features
2. **Live threat intelligence** from VirusTotal and URLhaus
3. **Domain forensics** including WHOIS age, DNS records, and typosquatting detection
4. **Weighted score combination** that merges all signals into a unified threat assessment

The system achieves **96.71% accuracy** with an end-to-end response time of **~400ms**, meeting both the accuracy target (≥95%) and latency target (<500ms).
