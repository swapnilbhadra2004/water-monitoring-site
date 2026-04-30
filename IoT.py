import streamlit as st
import serial
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ─────────────────────────────────────────────
#  PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TDS Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0e17;
    color: #c9d4e8;
}
.stApp { background-color: #0a0e17; }

section[data-testid="stSidebar"] {
    background: #0d1321;
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] * { color: #8aa3c1 !important; }
section[data-testid="stSidebar"] .stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #0f4c75, #1b6ca8);
    color: #e0f0ff !important;
    border: none;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    padding: 0.55rem;
    letter-spacing: 0.08em;
    transition: filter 0.2s;
}
section[data-testid="stSidebar"] .stButton>button:hover { filter: brightness(1.2); }

div[data-testid="metric-container"] {
    background: #101828;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 20px rgba(0,100,200,0.08);
}
div[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #4d7ea8 !important;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem !important;
    color: #48cae4 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
}

.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4d7ea8;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.badge-safe   { background: #0d3320; color: #39d97a; border: 1px solid #1a6640; }
.badge-warn   { background: #2d2000; color: #f4c430; border: 1px solid #7a5500; }
.badge-danger { background: #2d0a0a; color: #ff4d4d; border: 1px solid #7a1a1a; }
.badge-idle   { background: #151f2e; color: #5a7a99; border: 1px solid #1e3a5f; }

.gauge-wrap {
    background: #101828;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.alert-box {
    background: #2d0a0a;
    border-left: 4px solid #ff4d4d;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #ff9999;
    margin-top: 0.5rem;
}

/* ── Health Info Cards ── */
.info-card {
    background: #101828;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.info-card h4 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 0 0 0.6rem 0;
}
.info-card ul {
    margin: 0;
    padding-left: 1.2rem;
    font-size: 0.85rem;
    line-height: 1.9;
    color: #a0b8d4;
}
.info-card li { margin-bottom: 0.1rem; }

.solution-item {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1a2a40;
    font-size: 0.84rem;
    color: #a0b8d4;
    line-height: 1.5;
}
.solution-icon {
    font-size: 1.1rem;
    min-width: 1.4rem;
    margin-top: 0.05rem;
}

/* ── Prediction Box ── */
.pred-box {
    background: #0d1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.pred-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.5rem;
    letter-spacing: 0.05em;
}
.pred-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #4d7ea8;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.trend-up   { color: #ff6b6b; }
.trend-down { color: #39d97a; }
.trend-flat { color: #f4c430; }

.stDataFrame { border: 1px solid #1e3a5f; border-radius: 8px; }
.stSlider > div { color: #4d7ea8; }
h1, h2, h3 { color: #c9d4e8 !important; }
.stTextInput input {
    background: #0d1321 !important;
    border: 1px solid #1e3a5f !important;
    color: #c9d4e8 !important;
    font-family: 'Share Tech Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
defaults = {
    "connected": False,
    "paused": False,
    "data": [],
    "alert_triggered": False,
    "ser": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
#  TDS KNOWLEDGE BASE
# ─────────────────────────────────────────────
TDS_RANGES = [
    {
        "range": (0, 50),
        "label": "TOO PURE",
        "color": "#48cae4",
        "health_effects": [
            "Lacks essential minerals (Ca, Mg, K) your body needs",
            "May leach minerals from teeth and bones over time",
            "Flat, tasteless — often from reverse osmosis without remineralization",
            "Not ideal for long-term sole consumption",
        ],
        "suitable_for": ["Laboratory & pharmaceutical use", "Steam irons / humidifiers", "Car batteries / cooling systems"],
        "not_suitable_for": ["Daily drinking (long-term)", "Cooking — reduces mineral intake from food"],
        "solutions": [
            ("💊", "Add a remineralization filter or mineral drops after RO treatment"),
            ("🧂", "Add a pinch of Himalayan/sea salt and a squeeze of lemon per litre"),
            ("🔄", "Blend with mineral-rich water to reach 100–300 ppm range"),
        ],
    },
    {
        "range": (50, 150),
        "label": "EXCELLENT",
        "color": "#39d97a",
        "health_effects": [
            "Ideal mineral balance for drinking and cooking",
            "Promotes healthy hydration and electrolyte balance",
            "WHO-recommended sweet spot for potable water",
            "Low risk of scale formation in pipes and appliances",
        ],
        "suitable_for": ["Daily drinking ✓", "Cooking & beverages ✓", "Baby formula ✓", "Coffee & tea (optimal extraction)"],
        "not_suitable_for": [],
        "solutions": [
            ("✅", "Water quality is excellent — maintain your filtration system regularly"),
            ("🗓️", "Schedule filter replacements every 6 months to sustain this level"),
        ],
    },
    {
        "range": (150, 300),
        "label": "GOOD",
        "color": "#6fe89e",
        "health_effects": [
            "Generally safe and pleasant to drink",
            "Adequate mineral content for daily hydration",
            "Slight mineral taste may be noticeable",
            "Suitable for all household uses",
        ],
        "suitable_for": ["Daily drinking ✓", "Cooking ✓", "Most household appliances"],
        "not_suitable_for": ["Sensitive aquarium fish (may need filtering)"],
        "solutions": [
            ("✅", "Water is in the good range — no immediate action needed"),
            ("🔍", "Monitor periodically; consider a carbon filter to improve taste"),
        ],
    },
    {
        "range": (300, 500),
        "label": "ACCEPTABLE",
        "color": "#f4c430",
        "health_effects": [
            "Meets WHO drinking water guidelines (max 500 ppm)",
            "Noticeable mineral/salty taste",
            "May cause mild scale buildup in kettles and pipes over time",
            "Could affect taste of tea, coffee, and soups",
        ],
        "suitable_for": ["Drinking (within limits)", "Cooking (some taste impact)", "Irrigation"],
        "not_suitable_for": ["Steam appliances (scale risk)", "Sensitive aquarium fish", "Hydroponics"],
        "solutions": [
            ("🚰", "Install an activated carbon + sediment pre-filter to reduce dissolved solids"),
            ("💧", "Consider a countertop RO unit for drinking water specifically"),
            ("🫧", "Descale kettles and appliances monthly with citric acid solution"),
            ("📊", "Test for specific contaminants (hardness, nitrates, chlorides)"),
        ],
    },
    {
        "range": (500, 1000),
        "label": "POOR",
        "color": "#ff9900",
        "health_effects": [
            "Above WHO recommended limit of 500 ppm for drinking water",
            "High risk of kidney stones with prolonged consumption",
            "Significant scale buildup damages plumbing and appliances",
            "May contain elevated heavy metals, nitrates, or sulfates",
            "Bitter/salty taste — often unpleasant to drink",
            "Not safe for infants, elderly, or immunocompromised individuals",
        ],
        "suitable_for": ["Flushing toilets", "Garden irrigation (non-edible plants)", "Industrial cleaning"],
        "not_suitable_for": [
            "Drinking ✗", "Cooking ✗", "Baby formula ✗",
            "Aquariums ✗", "Steam appliances ✗",
        ],
        "solutions": [
            ("🚨", "Stop drinking this water immediately — use bottled water as interim"),
            ("🔬", "Get a full water quality lab test to identify contaminant profile"),
            ("💧", "Install a reverse osmosis (RO) system — can reduce TDS by 90–99%"),
            ("⚗️", "Ion exchange water softener if hardness is the primary cause"),
            ("🏠", "Check source: pipe corrosion, nearby industrial runoff, or seasonal contamination"),
            ("🛑", "Flush pipes for 2–3 minutes before use if water has been standing"),
        ],
    },
    {
        "range": (1000, 2000),
        "label": "UNSAFE",
        "color": "#ff4d4d",
        "health_effects": [
            "Severely contaminated — poses immediate health risks",
            "High risk of gastrointestinal illness and diarrhea",
            "Toxic heavy metals (lead, arsenic, mercury) likely present",
            "Causes rapid and severe scale/corrosion in all plumbing",
            "Linked to hypertension and cardiovascular disease with chronic exposure",
            "Can be lethal for pets, fish, and sensitive plants",
            "May indicate sewage or industrial waste contamination",
        ],
        "suitable_for": ["Concrete mixing", "Non-potable industrial processes only"],
        "not_suitable_for": [
            "Drinking ✗✗", "Cooking ✗✗", "Bathing (prolonged) ✗",
            "Irrigation (edible crops) ✗✗", "Any domestic use ✗",
        ],
        "solutions": [
            ("🆘", "CRITICAL: Do not use this water for any consumable purpose"),
            ("📞", "Report to local water authority / municipal corporation immediately"),
            ("🔬", "Emergency lab testing — identify specific toxins (heavy metals, pathogens)"),
            ("💧", "Multi-stage RO with UV sterilization is the minimum treatment needed"),
            ("🏭", "If from a well/borewell: check for nearby industrial or agricultural runoff"),
            ("🧪", "Distillation can be used as an emergency backup if RO is unavailable"),
            ("🚫", "Seal the source until a certified water treatment technician inspects it"),
        ],
    },
]


def get_tds_info(tds_val):
    """Return the knowledge-base entry for the given TDS value."""
    for entry in TDS_RANGES:
        lo, hi = entry["range"]
        if lo <= tds_val < hi:
            return entry
    return TDS_RANGES[-1]  # fallback: unsafe


# ─────────────────────────────────────────────
#  PREDICTION MODEL
# ─────────────────────────────────────────────
def run_prediction(df, horizon_steps=10):
    """
    Fit a polynomial regression on the TDS time-series and predict
    the next `horizon_steps` readings (1 reading/sec cadence).

    Returns:
        pred_values  : np.array of predicted TDS (length = horizon_steps)
        trend        : 'up' | 'down' | 'flat'
        slope_ppm_min: rate of change in ppm per minute
        r2           : model fit quality
        confidence   : 'HIGH' | 'MEDIUM' | 'LOW'
    """
    n = len(df)
    if n < 5:
        return None, "flat", 0.0, 0.0, "LOW"

    X = np.arange(n).reshape(-1, 1)
    y = df["tds"].values

    # Exponential smoothing to reduce noise before fitting
    alpha = 0.3
    y_smooth = np.zeros_like(y, dtype=float)
    y_smooth[0] = y[0]
    for i in range(1, len(y)):
        y_smooth[i] = alpha * y[i] + (1 - alpha) * y_smooth[i - 1]

    # Use degree-2 polynomial for short series, degree-3 for longer ones
    degree = 2 if n < 30 else 3
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y_smooth)

    # Predict next horizon_steps
    X_future = np.arange(n, n + horizon_steps).reshape(-1, 1)
    pred_values = model.predict(X_future)
    pred_values = np.clip(pred_values, 0, 3000)

    # R² on smoothed series
    y_pred_train = model.predict(X)
    ss_res = np.sum((y_smooth - y_pred_train) ** 2)
    ss_tot = np.sum((y_smooth - y_smooth.mean()) ** 2)
    r2 = max(0.0, 1 - ss_res / (ss_tot + 1e-9))

    # Slope: ppm per minute (60 readings/min at 1 Hz)
    slope_per_reading = (pred_values[-1] - pred_values[0]) / horizon_steps
    slope_ppm_min = slope_per_reading * 60

    # Trend classification
    if slope_ppm_min > 5:
        trend = "up"
    elif slope_ppm_min < -5:
        trend = "down"
    else:
        trend = "flat"

    # Confidence
    if r2 > 0.85 and n >= 20:
        confidence = "HIGH"
    elif r2 > 0.60 and n >= 10:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return pred_values, trend, slope_ppm_min, r2, confidence


def render_prediction_section(df, threshold):
    """Render the prediction panel inside the dashboard."""
    st.markdown('<div class="section-header">▸ PREDICTIVE ANALYSIS · POLYNOMIAL REGRESSION MODEL</div>',
                unsafe_allow_html=True)

    if len(df) < 5:
        st.info("⏳ Need at least 5 readings to activate the prediction model…")
        return

    pred_vals, trend, slope, r2, confidence = run_prediction(df, horizon_steps=10)

    if pred_vals is None:
        st.info("Insufficient data for prediction.")
        return

    pred_60s  = pred_vals[0]         # ~10 readings ahead
    predicted_tds_1min = pred_vals[-1]
    will_breach = any(v > threshold for v in pred_vals)

    trend_icon  = {"up": "📈", "down": "📉", "flat": "➡️"}[trend]
    trend_class = {"up": "trend-up", "down": "trend-down", "flat": "trend-flat"}[trend]
    trend_label = {"up": "RISING", "down": "FALLING", "flat": "STABLE"}[trend]
    conf_color  = {"HIGH": "#39d97a", "MEDIUM": "#f4c430", "LOW": "#ff6b6b"}[confidence]

    # ── KPI row ──
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        color = "#ff4d4d" if pred_60s > threshold else "#48cae4"
        st.markdown(f"""
<div class="pred-box">
  <div class="pred-label">Predicted TDS (~10s)</div>
  <div class="pred-value" style="color:{color}">{pred_60s:.1f} <span style="font-size:0.9rem">ppm</span></div>
</div>""", unsafe_allow_html=True)

    with p2:
        color2 = "#ff4d4d" if predicted_tds_1min > threshold else "#48cae4"
        st.markdown(f"""
<div class="pred-box">
  <div class="pred-label">Projected TDS (1 min)</div>
  <div class="pred-value" style="color:{color2}">{predicted_tds_1min:.1f} <span style="font-size:0.9rem">ppm</span></div>
</div>""", unsafe_allow_html=True)

    with p3:
        st.markdown(f"""
<div class="pred-box">
  <div class="pred-label">Rate of Change</div>
  <div class="pred-value {trend_class}">{trend_icon} {abs(slope):.1f} <span style="font-size:0.8rem">ppm/min</span></div>
  <div style="font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#4d7ea8;margin-top:0.3rem">{trend_label}</div>
</div>""", unsafe_allow_html=True)

    with p4:
        st.markdown(f"""
<div class="pred-box">
  <div class="pred-label">Model Confidence</div>
  <div class="pred-value" style="color:{conf_color}">{confidence}</div>
  <div style="font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#4d7ea8;margin-top:0.3rem">R² = {r2:.3f}</div>
</div>""", unsafe_allow_html=True)

    # ── Breach warning ──
    if will_breach:
        st.markdown(f"""
<div class="alert-box" style="margin-top:0.8rem">
  🔮 MODEL ALERT — Predicted TDS will exceed the {threshold} ppm threshold
  within the next 10 readings. Recommend immediate inspection of water source.
</div>""", unsafe_allow_html=True)

    # ── Prediction chart (combine historical + forecast) ──
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;'
        'color:#4d7ea8;letter-spacing:0.12em;margin:0.8rem 0 0.3rem">HISTORICAL + FORECAST OVERLAY</div>',
        unsafe_allow_html=True,
    )

    # Build a unified dataframe: last 40 actual + 10 predicted
    hist = df["tds"].tail(40).reset_index(drop=True)
    future_idx_start = len(hist)
    pred_series = pd.Series(pred_vals, index=range(future_idx_start, future_idx_start + len(pred_vals)))

    combined = pd.DataFrame({
        "Actual TDS (ppm)": hist,
        "Predicted TDS (ppm)": pd.Series(dtype=float),
    })
    for i, v in zip(pred_series.index, pred_series.values):
        combined.loc[i, "Predicted TDS (ppm)"] = v

    # Mark handoff point
    combined.loc[future_idx_start, "Predicted TDS (ppm)"] = hist.iloc[-1]

    st.line_chart(combined, height=200, use_container_width=True)

    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;'
        'color:#3a5a7a;letter-spacing:0.08em;margin-top:0.3rem">'
        '⚙ Polynomial regression (degree 2/3) with exponential smoothing (α=0.3) · '
        'Updates every reading cycle</div>',
        unsafe_allow_html=True,
    )


def render_health_section(latest_tds):
    """Render the health effects + solutions panel."""
    st.markdown('<div class="section-header">▸ HEALTH IMPACT & USAGE ANALYSIS</div>',
                unsafe_allow_html=True)

    if latest_tds is None:
        st.info("Connect to sensor to see health analysis.")
        return

    info = get_tds_info(latest_tds)
    color = info["color"]

    h1, h2 = st.columns([1, 1])

    # ── Left: Health effects + suitability ──
    with h1:
        st.markdown(f"""
<div class="info-card" style="border-color:{color}33">
  <h4 style="color:{color}">⚕ Health & Usage Effects at {latest_tds:.0f} ppm ({info['label']})</h4>
  <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;
              letter-spacing:0.12em;color:#4d7ea8;margin-bottom:0.5rem">
    WHAT HAPPENS WHEN YOU USE THIS WATER
  </div>
  <ul>
""" + "".join(f"<li>{e}</li>" for e in info["health_effects"]) + """
  </ul>
</div>""", unsafe_allow_html=True)

        # Suitable / not suitable
        if info["suitable_for"] or info["not_suitable_for"]:
            suitable_html = ""
            if info["suitable_for"]:
                suitable_html += f"""
<div style="margin-bottom:0.5rem">
  <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;
              letter-spacing:0.12em;color:#39d97a;margin-bottom:0.3rem">✔ SAFE TO USE FOR</div>
  <ul style="margin:0;padding-left:1.2rem;font-size:0.83rem;
              line-height:1.9;color:#a0b8d4">
    {"".join(f"<li>{u}</li>" for u in info['suitable_for'])}
  </ul>
</div>"""
            if info["not_suitable_for"]:
                suitable_html += f"""
<div>
  <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;
              letter-spacing:0.12em;color:#ff4d4d;margin-bottom:0.3rem">✘ AVOID USING FOR</div>
  <ul style="margin:0;padding-left:1.2rem;font-size:0.83rem;
              line-height:1.9;color:#a0b8d4">
    {"".join(f"<li>{u}</li>" for u in info['not_suitable_for'])}
  </ul>
</div>"""
            st.markdown(f'<div class="info-card">{suitable_html}</div>', unsafe_allow_html=True)

    # ── Right: Solutions ──
    with h2:
        sol_items = "".join(
            f'<div class="solution-item"><span class="solution-icon">{icon}</span><span>{text}</span></div>'
            for icon, text in info["solutions"]
        )
        st.markdown(f"""
<div class="info-card" style="border-color:{color}33">
  <h4 style="color:{color}">🛠 Recommended Solutions & Actions</h4>
  <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;
              letter-spacing:0.12em;color:#4d7ea8;margin-bottom:0.6rem">
    CORRECTIVE MEASURES FOR {latest_tds:.0f} PPM WATER
  </div>
  {sol_items}
</div>""", unsafe_allow_html=True)

        # TDS scale reference
        st.markdown("""
<div class="info-card">
  <h4 style="color:#4d7ea8">📋 TDS Reference Scale</h4>
  <table style="width:100%;font-family:Share Tech Mono,monospace;
                font-size:0.75rem;border-collapse:collapse">
    <tr style="border-bottom:1px solid #1e3a5f">
      <th style="text-align:left;color:#4d7ea8;padding:0.25rem 0">Range (ppm)</th>
      <th style="text-align:left;color:#4d7ea8;padding:0.25rem 0">Classification</th>
      <th style="text-align:left;color:#4d7ea8;padding:0.25rem 0">WHO Status</th>
    </tr>
    <tr><td style="padding:0.2rem 0;color:#48cae4">0 – 50</td>
        <td style="color:#48cae4">Too Pure</td><td style="color:#f4c430">⚠ Monitor</td></tr>
    <tr><td style="padding:0.2rem 0;color:#39d97a">50 – 150</td>
        <td style="color:#39d97a">Excellent</td><td style="color:#39d97a">✔ Ideal</td></tr>
    <tr><td style="padding:0.2rem 0;color:#6fe89e">150 – 300</td>
        <td style="color:#6fe89e">Good</td><td style="color:#39d97a">✔ Safe</td></tr>
    <tr><td style="padding:0.2rem 0;color:#f4c430">300 – 500</td>
        <td style="color:#f4c430">Acceptable</td><td style="color:#f4c430">⚠ Limit</td></tr>
    <tr><td style="padding:0.2rem 0;color:#ff9900">500 – 1000</td>
        <td style="color:#ff9900">Poor</td><td style="color:#ff4d4d">✘ Exceeds</td></tr>
    <tr><td style="padding:0.2rem 0;color:#ff4d4d">1000+</td>
        <td style="color:#ff4d4d">Unsafe</td><td style="color:#ff4d4d">✘✘ Hazardous</td></tr>
  </table>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    port = st.text_input("COM Port", "COM12")
    baud = st.selectbox("Baud Rate", [9600, 115200, 57600], index=0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 Connect"):
            try:
                ser = serial.Serial(port, baud, timeout=1)
                st.session_state.ser = ser
                st.session_state.connected = True
                st.success("Connected")
            except Exception as e:
                st.error(f"Failed: {e}")
    with col2:
        if st.button("⛔ Disconnect"):
            if st.session_state.ser:
                try:
                    st.session_state.ser.close()
                except:
                    pass
            st.session_state.connected = False
            st.session_state.ser = None
            st.info("Disconnected")

    st.markdown("---")
    st.markdown("### 🚨 Alert Threshold")
    threshold = st.slider("Max TDS (ppm)", min_value=100, max_value=2000, value=500, step=10)

    st.markdown("---")
    st.markdown("### 📊 Chart Window")
    window = st.slider("Last N readings", 10, 200, 50, step=10)

    st.markdown("---")
    st.markdown("### ⏸ Controls")
    pause_label = "▶ Resume" if st.session_state.paused else "⏸ Pause"
    if st.button(pause_label):
        st.session_state.paused = not st.session_state.paused

    if st.button("🗑 Clear Data"):
        st.session_state.data = []
        st.session_state.alert_triggered = False

    if st.session_state.data:
        df_export = pd.DataFrame(st.session_state.data)
        csv_bytes = df_export.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Export CSV",
            data=csv_bytes,
            file_name=f"tds_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.markdown("---")
    status_dot = "🟢" if st.session_state.connected else "🔴"
    st.markdown(f"**Status:** {status_dot} {'Connected' if st.session_state.connected else 'Not connected'}")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:0.2rem'>
  <span style='font-size:2rem'>💧</span>
  <div>
    <div style='font-family:Share Tech Mono,monospace; font-size:1.6rem;
                color:#48cae4; letter-spacing:0.05em; line-height:1.1'>
      TDS LIVE MONITOR
    </div>
    <div style='font-family:Share Tech Mono,monospace; font-size:0.7rem;
                color:#4d7ea8; letter-spacing:0.2em'>
      TOTAL DISSOLVED SOLIDS · REAL-TIME ANALYSIS · PREDICTIVE INTELLIGENCE
    </div>
  </div>
</div>
<hr style='border-color:#1e3a5f; margin:0.6rem 0 1rem'>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def quality_badge(tds_val):
    if tds_val < 50:
        return '<span class="badge badge-idle">◈ TOO PURE</span>'
    elif tds_val < 150:
        return '<span class="badge badge-safe">✔ EXCELLENT</span>'
    elif tds_val < 300:
        return '<span class="badge badge-safe">✔ GOOD</span>'
    elif tds_val < 500:
        return '<span class="badge badge-warn">⚠ ACCEPTABLE</span>'
    elif tds_val < 1000:
        return '<span class="badge badge-warn">⚠ POOR</span>'
    else:
        return '<span class="badge badge-danger">✘ UNSAFE</span>'


def quality_color(tds_val):
    if tds_val < 300:
        return "#39d97a"
    elif tds_val < 500:
        return "#f4c430"
    else:
        return "#ff4d4d"


# ─────────────────────────────────────────────
#  MAIN PLACEHOLDER
# ─────────────────────────────────────────────
placeholder = st.empty()


# ─────────────────────────────────────────────
#  RENDER DASHBOARD
# ─────────────────────────────────────────────
def render(latest_tds=None, latest_status="--"):
    df = pd.DataFrame(st.session_state.data)

    with placeholder.container():
        # ── KPI Row ──────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        if df.empty:
            k1.metric("Latest TDS",  "-- ppm")
            k2.metric("Average TDS", "-- ppm")
            k3.metric("Min TDS",     "-- ppm")
            k4.metric("Max TDS",     "-- ppm")
            k5.metric("Readings",    "0")
        else:
            cur   = df["tds"].iloc[-1]
            prev  = df["tds"].iloc[-2] if len(df) > 1 else cur
            delta = cur - prev
            k1.metric("Latest TDS",   f"{cur:.1f} ppm",  f"{delta:+.1f}")
            k2.metric("Average TDS",  f"{df['tds'].mean():.1f} ppm")
            k3.metric("Min TDS",      f"{df['tds'].min():.1f} ppm")
            k4.metric("Max TDS",      f"{df['tds'].max():.1f} ppm")
            k5.metric("Readings",     str(len(df)))

        # ── Status Row ───────────────────────────────
        st.markdown('<div class="section-header">▸ WATER QUALITY STATUS</div>',
                    unsafe_allow_html=True)
        s1, s2, s3 = st.columns([1, 1, 2])
        with s1:
            badge = quality_badge(latest_tds) if latest_tds is not None \
                    else '<span class="badge badge-idle">── IDLE</span>'
            st.markdown(f"**Quality Index** &nbsp; {badge}", unsafe_allow_html=True)
        with s2:
            pause_txt = "⏸ PAUSED" if st.session_state.paused else "▶ LIVE"
            color = "#f4c430" if st.session_state.paused else "#39d97a"
            st.markdown(
                f'<span style="font-family:Share Tech Mono,monospace;'
                f'font-size:0.85rem;color:{color}">{pause_txt}</span>',
                unsafe_allow_html=True,
            )
        with s3:
            if st.session_state.alert_triggered and latest_tds is not None:
                st.markdown(
                    f'<div class="alert-box">🚨 THRESHOLD EXCEEDED — '
                    f'{latest_tds:.1f} ppm &gt; {threshold} ppm limit</div>',
                    unsafe_allow_html=True,
                )

        # ── Chart + Gauge ─────────────────────────────
        st.markdown('<div class="section-header">▸ LIVE READINGS</div>',
                    unsafe_allow_html=True)
        chart_col, gauge_col = st.columns([3, 1])

        with chart_col:
            if not df.empty:
                chart_df = df[["tds"]].tail(window).rename(columns={"tds": "TDS (ppm)"})
                st.line_chart(chart_df, height=260, use_container_width=True)
            else:
                st.info("Waiting for data…")

        with gauge_col:
            st.markdown('<div class="gauge-wrap">', unsafe_allow_html=True)
            if latest_tds is not None:
                pct   = min(latest_tds / 1500, 1.0)
                color = quality_color(latest_tds)
                arc   = pct * 180
                gauge_svg = f"""
<svg viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg" style="width:100%">
  <defs>
    <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#39d97a"/>
      <stop offset="50%"  stop-color="#f4c430"/>
      <stop offset="100%" stop-color="#ff4d4d"/>
    </linearGradient>
  </defs>
  <path d="M 20 100 A 80 80 0 0 1 180 100"
        fill="none" stroke="#1e3a5f" stroke-width="14" stroke-linecap="round"/>
  <path d="M 20 100 A 80 80 0 0 1 180 100"
        fill="none" stroke="url(#arcGrad)" stroke-width="14"
        stroke-linecap="round"
        stroke-dasharray="{arc / 180 * 251:.1f} 251"
        opacity="0.9"/>
  <line x1="100" y1="100"
        x2="{100 + 65 * (-0.98 + pct * 1.96):.1f}"
        y2="{100 - 65 * (1 - abs(-1 + pct * 2)):.1f}"
        stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="100" cy="100" r="5" fill="{color}"/>
  <text x="100" y="80" text-anchor="middle"
        font-family="Share Tech Mono, monospace" font-size="18"
        fill="{color}">{latest_tds:.0f}</text>
  <text x="100" y="95" text-anchor="middle"
        font-family="Share Tech Mono, monospace" font-size="9"
        fill="#4d7ea8">ppm</text>
  <text x="20"  y="115" font-family="Share Tech Mono,monospace"
        font-size="8" fill="#4d7ea8">0</text>
  <text x="170" y="115" font-family="Share Tech Mono,monospace"
        font-size="8" fill="#4d7ea8">1500</text>
</svg>"""
                st.markdown(gauge_svg, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="font-family:Share Tech Mono,monospace;'
                    'color:#4d7ea8;font-size:0.8rem">NO SIGNAL</p>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Stats + Table ─────────────────────────────
        if not df.empty:
            st.markdown('<div class="section-header">▸ STATISTICS & LOG</div>',
                        unsafe_allow_html=True)
            stat_col, table_col = st.columns([1, 2])
            with stat_col:
                tds   = df["tds"]
                std   = tds.std() if len(tds) > 1 else 0.0
                p95   = np.percentile(tds, 95)
                above = (tds > threshold).sum()
                st.markdown(f"""
<div style='font-family:Share Tech Mono,monospace; font-size:0.82rem;
            line-height:2; color:#8aa3c1'>
  <div>STD DEV &nbsp;&nbsp;&nbsp; <span style='color:#48cae4'>{std:.2f} ppm</span></div>
  <div>95th PCT &nbsp;&nbsp; <span style='color:#48cae4'>{p95:.1f} ppm</span></div>
  <div>OVER LIMIT &nbsp;<span style='color:#ff4d4d'>{above} readings</span></div>
  <div>THRESHOLD &nbsp;<span style='color:#f4c430'>{threshold} ppm</span></div>
</div>""", unsafe_allow_html=True)
            with table_col:
                display_df = df.tail(8).copy().rename(columns={
                    "time": "Elapsed (s)", "tds": "TDS (ppm)", "status": "Status"
                })
                st.dataframe(
                    display_df.style.background_gradient(
                        subset=["TDS (ppm)"], cmap="RdYlGn_r", vmin=0, vmax=1000,
                    ),
                    use_container_width=True, height=240,
                )

        # ── NEW: Prediction Section ───────────────────
        if not df.empty:
            render_prediction_section(df, threshold)

        # ── NEW: Health Impact Section ────────────────
        render_health_section(latest_tds)


# ─────────────────────────────────────────────
#  INITIAL RENDER
# ─────────────────────────────────────────────
if not st.session_state.connected:
    render()
    st.markdown("""
<div style='text-align:center; padding:2rem;
            font-family:Share Tech Mono,monospace;
            color:#4d7ea8; font-size:0.85rem; letter-spacing:0.1em'>
  ── AWAITING CONNECTION · SELECT PORT AND CLICK CONNECT ──
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LIVE READING LOOP
# ─────────────────────────────────────────────
if st.session_state.connected:
    ser = st.session_state.ser
    latest_tds, latest_status = None, "--"

    for _ in range(100_000):
        if not st.session_state.paused:
            try:
                line   = ser.readline().decode().strip()
                parts  = line.split(",")
                if len(parts) == 3:
                    t, tds_raw, status = parts
                    tds_val = float(tds_raw)
                    latest_tds, latest_status = tds_val, status
                    st.session_state.data.append({
                        "time":      int(t),
                        "tds":       tds_val,
                        "status":    status,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
                    if tds_val > threshold:
                        st.session_state.alert_triggered = True
                    else:
                        st.session_state.alert_triggered = False
            except Exception:
                pass

        render(latest_tds, latest_status)
        time.sleep(1)