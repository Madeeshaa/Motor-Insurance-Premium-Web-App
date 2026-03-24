import streamlit as st
import numpy as np
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="Motor Insurance Premium Calculator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #112240 100%);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* Card glass effect */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,179,237,0.12);
}

/* Section headers */
.section-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,179,237,0.4), transparent);
}

/* Feature badge */
.feature-badge {
    background: rgba(99,179,237,0.10);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.feature-badge .value {
    font-size: 26px;
    font-weight: 700;
    color: #63b3ed;
}
.feature-badge .label {
    font-size: 11px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* Premium output */
.premium-box {
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.premium-pure {
    background: linear-gradient(135deg, rgba(49,130,206,0.18), rgba(99,179,237,0.10));
    border: 1px solid rgba(99,179,237,0.35);
}
.premium-final {
    background: linear-gradient(135deg, rgba(72,187,120,0.18), rgba(104,211,145,0.10));
    border: 1px solid rgba(72,187,120,0.35);
}
.premium-box .amount {
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 4px;
}
.premium-box .sub {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.pure-amount { color: #63b3ed; }
.final-amount { color: #68d391; }

/* Risk gauge */
.risk-indicator {
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.5px;
    animation: glow 2s ease-in-out infinite alternate;
}
.risk-low    { background: rgba(72,187,120,0.15); border: 1px solid rgba(72,187,120,0.4); color: #68d391; }
.risk-medium { background: rgba(237,187,77,0.15); border: 1px solid rgba(237,187,77,0.4); color: #f6d860; }
.risk-high   { background: rgba(245,101,101,0.15); border: 1px solid rgba(245,101,101,0.4); color: #fc8181; }

@keyframes glow {
    from { box-shadow: 0 0 6px rgba(99,179,237,0.1); }
    to   { box-shadow: 0 0 14px rgba(99,179,237,0.25); }
}

/* Model breakdown */
.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 14px;
}
.model-row:last-child { border-bottom: none; }
.model-name { color: #94a3b8; }
.model-value { color: #e2e8f0; font-weight: 500; }

/* Streamlit widget overrides */
.stDateInput>label, .stNumberInput>label, .stSlider>label,
.stSelectbox>label, .stTextInput>label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}
.stDateInput input, .stNumberInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #4299e1, #3182ce) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(49,130,206,0.4) !important;
}
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 12px;
}

/* Divider */
hr { border-color: rgba(99,179,237,0.12) !important; }

.app-title {
    font-size: 30px;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #90cdf4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.app-subtitle {
    color: #718096;
    font-size: 14px;
    margin-top: -4px;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Training models (this takes ~2 minutes on first run)...")
def load_models():
    import premium_app_script as pas
    return pas


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚙️ Settings</div>', unsafe_allow_html=True)

    current_date = st.date_input(
        "Calculation Reference Date",
        value=date(2026, 3, 23),
        min_value=date(2000, 1, 1),
        max_value=date(2035, 12, 31),
        help="Change this date to simulate calculations at any point in time."
    )

    st.markdown("---")
    st.markdown('<div class="section-title">💼 Premium Loading</div>', unsafe_allow_html=True)
    loading = st.slider(
        "Premium Loading (%)",
        min_value=0, max_value=100, value=20, step=1,
        help="Adjust this per company policy. Final Premium = Pure Premium × (1 + Loading%)"
    )
    st.info(
        f"**Loading: {loading}%**\n\n"
        "This covers expenses, profit margin, and catastrophe reserves. "
        "Typical range: 15%–35%."
    )

    st.markdown("---")
    st.markdown(
        "**Model:** GLM (Neg-Binomial + Gamma) + ANFIS (Neuro-Fuzzy) + MLP Neural Stacker",
        unsafe_allow_html=False
    )


# ─── Header ──────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="app-title">🛡️ Motor Insurance Premium Calculator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Hybrid Neuro-Fuzzy (ANFIS) + GLM model for actuarial pure premium estimation</div>',
        unsafe_allow_html=True
    )
with col_h2:
    st.markdown(f"""
    <div style="text-align:right; padding-top:8px;">
        <div style="font-size:11px; color:#718096; text-transform:uppercase; letter-spacing:1px;">Reference Date</div>
        <div style="font-size:20px; font-weight:700; color:#63b3ed;">{current_date.strftime('%d %b %Y')}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── Load Models ─────────────────────────────────────────────────────────────
pas = load_models()

# ─── Input Fields ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Policyholder & Vehicle Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**🧑 Policyholder**")
    birth_date = st.date_input(
        "Date of Birth",
        value=date(1988, 6, 15),
        min_value=date(1920, 1, 1),
        max_value=current_date,
        key="birth_date"
    )
    licence_date = st.date_input(
        "Licence Issued Date",
        value=date(2009, 3, 1),
        min_value=date(1940, 1, 1),
        max_value=current_date,
        key="licence_date"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**🚗 Vehicle Details**")
    matric_year = st.number_input(
        "Matriculation Year",
        min_value=1970, max_value=current_date.year,
        value=2017, step=1, key="matric_year"
    )
    vehicle_value = st.number_input(
        "Current Market Value (€)",
        min_value=500.0, max_value=200000.0,
        value=18000.0, step=500.0, format="%.0f",
        key="vehicle_value"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**⚙️ Engine**")
    cc = st.number_input(
        "Cylinder Capacity (CC)",
        min_value=600, max_value=4000,
        value=1600, step=50, key="cc"
    )
    st.markdown(f"""
    <br>
    <div style="font-size:12px; color:#718096; margin-top:8px;">
        <b>CC Range Guide:</b><br>
        600–1400 cc → Small<br>
        1100–2100 cc → Medium<br>
        1800–4000 cc → High
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Compute Derived Features ─────────────────────────────────────────────────
def exact_years(from_date, to_date):
    years = to_date.year - from_date.year
    if (to_date.month, to_date.day) < (from_date.month, from_date.day):
        years -= 1
    return max(0, years)

age   = max(18, exact_years(birth_date, current_date))
exp   = max(0,  exact_years(licence_date, current_date))
v_age = max(0,  current_date.year - matric_year)

# ─── Computed Features Display ───────────────────────────────────────────────
st.markdown('<div class="section-title">📐 Computed Risk Features</div>', unsafe_allow_html=True)
fc1, fc2, fc3, fc4, fc5 = st.columns(5)

def badge(col, label, value, unit=""):
    col.markdown(f"""
    <div class="feature-badge">
        <div class="value">{value}<span style="font-size:16px;color:#94a3b8;"> {unit}</span></div>
        <div class="label">{label}</div>
    </div>""", unsafe_allow_html=True)

badge(fc1, "Policyholder Age", age, "yrs")
badge(fc2, "Driving Experience", exp, "yrs")
badge(fc3, "Vehicle Age", v_age, "yrs")
badge(fc4, "Market Value", f"€{vehicle_value:,.0f}", "")
badge(fc5, "Engine CC", f"{cc:,}", "cc")

# Warnings
st.markdown("")
if exp > age - 16:
    st.warning("⚠️ Driving experience exceeds maximum possible for given age.")
if v_age > 30:
    st.warning("⚠️ Vehicle age is very high (>30 years). Verify matriculation year.")

# ─── Calculate Button ─────────────────────────────────────────────────────────
st.markdown("---")
calc_clicked = st.button("🔢 Calculate Premium", type="primary", use_container_width=True)

if calc_clicked:
    with st.spinner("Computing pure premium using Neuro-Fuzzy + GLM ensemble..."):
        try:
            result = pas.predict_pure_premium(age, exp, v_age, vehicle_value, cc)
            pure_premium, glm_pp, nf_pp, glm_freq, nf_freq = result
            final_premium = pure_premium * (1 + loading / 100.0)

            # Risk tier
            obs_pp = pas.obs_pure_prem
            ratio = pure_premium / (obs_pp + 1e-8)
            if ratio < 0.75:    risk_cls, risk_label, risk_icon = "risk-low",    "LOW RISK",    "🟢"
            elif ratio < 1.5:   risk_cls, risk_label, risk_icon = "risk-medium", "MEDIUM RISK", "🟡"
            else:               risk_cls, risk_label, risk_icon = "risk-high",   "HIGH RISK",   "🔴"

            st.markdown("---")
            st.markdown('<div class="section-title">💰 Premium Results</div>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns([2, 2, 1])

            with r1:
                st.markdown(f"""
                <div class="premium-box premium-pure">
                    <div class="sub">Pure Premium (Net Risk Cost)</div>
                    <div class="amount pure-amount">€{pure_premium:,.2f}</div>
                    <div style="font-size:12px;color:#718096;margin-top:6px;">
                        Model-estimated expected loss cost
                    </div>
                </div>""", unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="premium-box premium-final">
                    <div class="sub">Final Premium (+{loading}% loading)</div>
                    <div class="amount final-amount">€{final_premium:,.2f}</div>
                    <div style="font-size:12px;color:#718096;margin-top:6px;">
                        Pure Premium × {1 + loading/100:.2f}
                    </div>
                </div>""", unsafe_allow_html=True)

            with r3:
                st.markdown(f"""
                <div class="risk-indicator {risk_cls}" style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;">
                    <div style="font-size:28px;">{risk_icon}</div>
                    <div>{risk_label}</div>
                    <div style="font-size:11px;font-weight:400;color:#94a3b8;">
                        {ratio:.2f}× avg risk
                    </div>
                </div>""", unsafe_allow_html=True)

            # ─── Model Breakdown ──────────────────────────────────────────
            st.markdown("")
            b1, b2 = st.columns(2)

            with b1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**📊 Model Component Breakdown**")
                st.markdown(f"""
                <div class="model-row">
                    <span class="model-name">GLM Pure Premium</span>
                    <span class="model-value">€{glm_pp:,.2f}</span>
                </div>
                <div class="model-row">
                    <span class="model-name">Neuro-Fuzzy Pure Premium</span>
                    <span class="model-value">€{nf_pp:,.2f}</span>
                </div>
                <div class="model-row">
                    <span class="model-name">MLP-Stack (Final) Pure Premium</span>
                    <span class="model-value" style="color:#63b3ed;font-weight:700;">€{pure_premium:,.2f}</span>
                </div>
                <div class="model-row">
                    <span class="model-name">Portfolio Average PP</span>
                    <span class="model-value">€{obs_pp:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with b2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**📡 Frequency Signals**")
                st.markdown(f"""
                <div class="model-row">
                    <span class="model-name">GLM Claim Frequency</span>
                    <span class="model-value">{glm_freq:.4f} claims/yr</span>
                </div>
                <div class="model-row">
                    <span class="model-name">NF Claim Frequency</span>
                    <span class="model-value">{nf_freq:.4f} claims/yr</span>
                </div>
                <div class="model-row">
                    <span class="model-name">Premium Loading</span>
                    <span class="model-value">{loading}%</span>
                </div>
                <div class="model-row">
                    <span class="model-name">Loading Amount</span>
                    <span class="model-value" style="color:#68d391;">€{final_premium - pure_premium:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.success(f"✅ Calculation complete. Final Premium: **€{final_premium:,.2f}** (Loading: {loading}%)")

        except Exception as e:
            st.error(f"❌ Error calculating premium: {e}")
            import traceback
            st.code(traceback.format_exc())

else:
    # Placeholder state
    st.markdown("""
    <div style="text-align:center; padding:40px; color:#4a5568;">
        <div style="font-size:48px; margin-bottom:12px;">📋</div>
        <div style="font-size:16px; font-weight:600; color:#63b3ed;">Enter policyholder details above</div>
        <div style="font-size:13px; color:#718096; margin-top:6px;">
            Adjust the inputs and click <b>Calculate Premium</b> to get the actuarial estimate
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:11px; color:#4a5568; padding:8px 0;">
    Motor Insurance Premium Calculator &mdash; Hybrid GLM + ANFIS + MLP Neural Stacker Model<br>
    <i>For actuarial reference only. Final premiums subject to underwriting review.</i>
</div>
""", unsafe_allow_html=True)
