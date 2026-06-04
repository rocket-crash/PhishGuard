/* ================================================================
   PhishGuard — Analyst Dashboard (app.js)
   ================================================================
   Handles: URL scanning, result rendering, stats bar, dual charts
            (doughnut + bar), scan history table, and IoC report
            export in JSON and CSV formats.
   ================================================================ */

const API_URL = '/scan/';
const scanHistory = [];

// ── DOM Handles ──
const $urlInput      = document.getElementById('url-input');
const $scanBtn       = document.getElementById('scan-btn');
const $scanLabel     = document.getElementById('scan-label');
const $spinner       = document.getElementById('scan-spinner');
const $errorMsg      = document.getElementById('error-msg');
const $resultWrap    = document.getElementById('result-wrap');
const $resultCard    = document.getElementById('result-card');
const $chartSection  = document.getElementById('chart-section');
const $historySection = document.getElementById('history-section');
const $exportSection = document.getElementById('export-section');
const $emptyState    = document.getElementById('empty-state');
const $statsBar      = document.getElementById('stats-bar');

$urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleScan(); });
$scanBtn.addEventListener('click', handleScan);

// ── Main Scan Handler ──
async function handleScan() {
  const url = $urlInput.value.trim();
  if (!url) return;

  setLoading(true);
  $errorMsg.classList.add('hidden');
  $resultCard.classList.add('hidden');

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) throw new Error('Server returned ' + res.status);
    const data = await res.json();
    scanHistory.push(data);
    renderResult(data);
    renderStats();
    renderCharts();
    renderHistory();
    renderExport();
    $emptyState.classList.add('hidden');
  } catch (err) {
    const msg = (err.message.includes('Failed to fetch') || err.message.includes('NetworkError'))
      ? 'Backend not running — start with: python backend/main.py'
      : 'Scan failed: ' + err.message;
    $errorMsg.textContent = msg;
    $errorMsg.classList.remove('hidden');
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  $scanBtn.disabled = on;
  $scanLabel.textContent = on ? 'Scanning…' : '🔍 Scan URL';
  $spinner.classList.toggle('hidden', !on);
  if (on) $resultWrap.style.opacity = '0.4';
  else $resultWrap.style.opacity = '1';
}

// ── Result Card ──
function renderResult(data) {
  const { url, cleaned_url, obfuscation_found = [], ml = {}, threat_intel = {}, final: ts = {}, forensics = {}, response_time_ms } = data;
  const level = ts.threat_level || 'safe';
  const score = ts.score ?? 0;
  const confidence = ml.confidence ?? 0;
  const reasons = ts.reasons || [];
  const vt = threat_intel.virustotal || {};
  const uh = threat_intel.urlhaus || {};
  const whois = forensics.whois || {};
  const dns = forensics.dns || {};
  const typo = forensics.typosquatting || {};

  const icons = { safe: '✅', suspicious: '⚠️', malicious: '🚨' };

  let html = `
    <div class="result-header">
      <div class="left">
        <span class="icon">${icons[level] || '✅'}</span>
        <div>
          <div class="level level-${level}">${level}</div>
          <div class="score-label">Threat Score: ${score}/100</div>
        </div>
      </div>
      <div class="time">${response_time_ms?.toFixed(0) || '?'} ms · source: ml</div>
    </div>

    <div class="confidence-section">
      <div class="confidence-label">
        <span>ML Confidence (phishing)</span>
        <span class="level-${level}">${(confidence * 100).toFixed(1)}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill bar-${level}" style="width:${Math.min(confidence * 100, 100)}%"></div>
      </div>
    </div>`;

  if (reasons.length > 0) {
    html += `<div class="findings"><div class="section-title">Findings</div><ul>`;
    reasons.forEach(r => { html += `<li>${escapeHtml(r)}</li>`; });
    html += `</ul></div>`;
  }

  let tiles = '';
  if (vt.available) tiles += infoTile('VirusTotal', `${vt.malicious_count||0}/${vt.total_engines||0} flagged`, vt.malicious_count > 0);
  if (uh.available) tiles += infoTile('URLhaus', uh.is_known_malicious ? 'Known malicious' : 'Not listed', uh.is_known_malicious);
  if (whois.available) tiles += infoTile('Domain Age', whois.domain_age_days != null ? `${whois.domain_age_days} days` : 'Unknown', whois.is_newly_registered);
  if (dns.available !== false) tiles += infoTile('MX Record', dns.has_mx_record ? 'Present ✓' : 'Missing ✗', !dns.has_mx_record);
  if (typo.is_typosquat) tiles += infoTile('Typosquat', `→ ${typo.original_brand} (dist ${typo.edit_distance})`, true);
  if (tiles) html += `<div class="info-grid">${tiles}</div>`;

  if (obfuscation_found.length > 0) {
    html += `<div><div class="section-title">Obfuscation Detected</div><div class="obfuscation-tags">`;
    obfuscation_found.forEach(t => { html += `<span class="obfuscation-tag">${escapeHtml(t)}</span>`; });
    html += `</div></div>`;
  }

  html += `<div class="scanned-url">${escapeHtml(url)}`;
  if (cleaned_url !== url) html += `<br>Cleaned: ${escapeHtml(cleaned_url)}`;
  html += `</div>`;

  $resultCard.className = `result-card glass-strong border-${level} bg-${level}`;
  $resultCard.innerHTML = html;
  $resultCard.classList.remove('hidden');
}

function infoTile(label, value, warn) {
  return `<div class="info-tile ${warn ? 'warn' : ''}">
    <div class="tile-label">${label}</div>
    <div class="tile-value">${escapeHtml(value)}</div>
  </div>`;
}

// ── Stats Bar ──
function renderStats() {
  $statsBar.classList.remove('hidden');
  const counts = { safe: 0, suspicious: 0, malicious: 0 };
  let totalTime = 0;

  scanHistory.forEach(s => {
    const l = s.final?.threat_level;
    if (l === 'safe') counts.safe++;
    else if (l === 'suspicious') counts.suspicious++;
    else if (l === 'malicious') counts.malicious++;
    totalTime += s.response_time_ms || 0;
  });

  document.getElementById('stat-total').textContent = scanHistory.length;
  document.getElementById('stat-safe').textContent = counts.safe;
  document.getElementById('stat-suspicious').textContent = counts.suspicious;
  document.getElementById('stat-malicious').textContent = counts.malicious;
  document.getElementById('stat-avg-time').textContent =
    scanHistory.length > 0 ? `${(totalTime / scanHistory.length).toFixed(0)}ms` : '—';
}

// ── Charts ──
function renderCharts() {
  $chartSection.classList.remove('hidden');

  const counts = { Safe: 0, Suspicious: 0, Malicious: 0 };
  scanHistory.forEach(s => {
    const l = s.final?.threat_level;
    if (l === 'safe') counts.Safe++;
    else if (l === 'suspicious') counts.Suspicious++;
    else if (l === 'malicious') counts.Malicious++;
  });

  // ── Doughnut Chart ──
  const doughnutCanvas = document.getElementById('risk-doughnut');
  const doughnutCtx = doughnutCanvas.getContext('2d');
  if (window._doughnutChart) window._doughnutChart.destroy();

  window._doughnutChart = new Chart(doughnutCtx, {
    type: 'doughnut',
    data: {
      labels: ['Safe', 'Suspicious', 'Malicious'],
      datasets: [{
        data: [counts.Safe, counts.Suspicious, counts.Malicious],
        backgroundColor: ['#4ade80', '#fbbf24', '#f87171'],
        borderColor: 'rgba(6,6,12,0.8)',
        borderWidth: 3,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: 'rgba(255,255,255,0.6)',
            font: { family: 'Inter', size: 11 },
            padding: 16,
            usePointStyle: true,
            pointStyleWidth: 10,
          }
        },
        tooltip: {
          backgroundColor: 'rgba(10,10,15,0.9)',
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1,
          titleFont: { family: 'Inter' },
          bodyFont: { family: 'Inter' },
        }
      },
    }
  });

  // ── Bar Chart (Timeline) ──
  const barCanvas = document.getElementById('risk-bar');
  const barCtx = barCanvas.getContext('2d');
  if (window._barChart) window._barChart.destroy();

  // Build per-scan score timeline
  const labels = scanHistory.map((_, i) => `#${i + 1}`);
  const scores = scanHistory.map(s => s.final?.score ?? 50);
  const bgColors = scanHistory.map(s => {
    const l = s.final?.threat_level;
    if (l === 'malicious') return '#f87171';
    if (l === 'suspicious') return '#fbbf24';
    return '#4ade80';
  });

  window._barChart = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Threat Score',
        data: scores,
        backgroundColor: bgColors,
        borderRadius: 4,
        maxBarThickness: 28,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(10,10,15,0.9)',
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1,
          titleFont: { family: 'Inter' },
          bodyFont: { family: 'Inter' },
          callbacks: {
            label: (ctx) => {
              const scan = scanHistory[ctx.dataIndex];
              const url = scan?.url || '?';
              const short = url.length > 40 ? url.substring(0, 37) + '…' : url;
              return `${short} — Score: ${ctx.raw}/100`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: 'rgba(255,255,255,0.4)', font: { family: 'Inter', size: 10 } },
          grid: { display: false },
          border: { display: false },
        },
        y: {
          min: 0, max: 100,
          ticks: { color: 'rgba(255,255,255,0.3)', font: { family: 'Inter', size: 10 }, stepSize: 25 },
          grid: { color: 'rgba(255,255,255,0.05)' },
          border: { display: false },
        }
      }
    }
  });
}

// ── History Table ──
function renderHistory() {
  $historySection.classList.remove('hidden');
  const $count = document.getElementById('history-count');
  const $tbody = document.getElementById('history-body');
  $count.textContent = scanHistory.length;
  $tbody.innerHTML = '';

  [...scanHistory].reverse().forEach((scan, i) => {
    const idx = scanHistory.length - i;
    const level = scan.final?.threat_level || 'safe';
    const score = scan.final?.score ?? 0;
    const conf = scan.ml?.confidence ?? 0;
    const ms = scan.response_time_ms?.toFixed(0) ?? '?';
    const source = scan.source || 'ml';
    const displayUrl = scan.url?.length > 50 ? scan.url.substring(0, 47) + '…' : (scan.url || '—');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${idx}</td>
      <td title="${escapeHtml(scan.url || '')}">${escapeHtml(displayUrl)}</td>
      <td><span class="badge badge-${level}">${level}</span></td>
      <td>${score}/100</td>
      <td class="level-${level}">${(conf * 100).toFixed(1)}%</td>
      <td><span class="badge badge-ml">${source}</span></td>
      <td>${ms} ms</td>`;
    $tbody.appendChild(tr);
  });
}

// ── Export Section ──
function renderExport() {
  $exportSection.classList.remove('hidden');
  const $jsonBtn = document.getElementById('export-json-btn');
  const $csvBtn = document.getElementById('export-csv-btn');
  $jsonBtn.textContent = `📄 Download IoC Report (JSON) — ${scanHistory.length} scans`;
  $csvBtn.textContent = `📊 Download IoC Report (CSV) — ${scanHistory.length} scans`;
  $jsonBtn.onclick = exportIoCJSON;
  $csvBtn.onclick = exportIoCCSV;
}

function exportIoCJSON() {
  const report = {
    report_type: 'PhishGuard IoC Report',
    generated_at: new Date().toISOString(),
    total_scans: scanHistory.length,
    summary: {
      safe: scanHistory.filter(s => s.final?.threat_level === 'safe').length,
      suspicious: scanHistory.filter(s => s.final?.threat_level === 'suspicious').length,
      malicious: scanHistory.filter(s => s.final?.threat_level === 'malicious').length,
    },
    indicators: scanHistory.map(scan => ({
      url: scan.url,
      cleaned_url: scan.cleaned_url,
      threat_level: scan.final?.threat_level,
      threat_score: scan.final?.score,
      ml_confidence: scan.ml?.confidence,
      is_phishing: scan.ml?.is_phishing,
      reasons: scan.final?.reasons,
      obfuscation: scan.obfuscation_found,
      source: scan.source,
      virustotal: scan.threat_intel?.virustotal,
      urlhaus: scan.threat_intel?.urlhaus,
      forensics: scan.forensics,
      response_time_ms: scan.response_time_ms,
    })),
  };

  downloadBlob(
    JSON.stringify(report, null, 2),
    'application/json',
    `phishguard-ioc-report-${Date.now()}.json`
  );
}

function exportIoCCSV() {
  const headers = [
    'url', 'cleaned_url', 'threat_level', 'threat_score', 'ml_confidence',
    'is_phishing', 'source', 'response_time_ms', 'reasons',
    'vt_malicious', 'vt_total', 'urlhaus_known',
    'domain_age_days', 'newly_registered', 'has_mx_record',
    'is_typosquat', 'obfuscation'
  ];

  const rows = scanHistory.map(scan => {
    const vt = scan.threat_intel?.virustotal || {};
    const uh = scan.threat_intel?.urlhaus || {};
    const whois = scan.forensics?.whois || {};
    const dns = scan.forensics?.dns || {};
    const typo = scan.forensics?.typosquatting || {};
    return [
      csvEscape(scan.url),
      csvEscape(scan.cleaned_url),
      scan.final?.threat_level || '',
      scan.final?.score ?? '',
      scan.ml?.confidence ?? '',
      scan.ml?.is_phishing ?? '',
      scan.source || '',
      scan.response_time_ms?.toFixed(2) ?? '',
      csvEscape((scan.final?.reasons || []).join('; ')),
      vt.malicious_count ?? '',
      vt.total_engines ?? '',
      uh.is_known_malicious ?? '',
      whois.domain_age_days ?? '',
      whois.is_newly_registered ?? '',
      dns.has_mx_record ?? '',
      typo.is_typosquat ?? '',
      csvEscape((scan.obfuscation_found || []).join('; ')),
    ].join(',');
  });

  const csv = [headers.join(','), ...rows].join('\n');
  downloadBlob(csv, 'text/csv', `phishguard-ioc-report-${Date.now()}.csv`);
}

// ── Utilities ──
function downloadBlob(content, mimeType, filename) {
  const blob = new Blob([content], { type: mimeType });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}

function csvEscape(str) {
  if (str == null) return '';
  str = String(str);
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
