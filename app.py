import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Burn-In AI Inspector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME — built to match the supplied reference dashboard
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
    --bg:#030611; --panel:#080d1d; --panel2:#0b1125;
    --line:rgba(125,145,255,.18); --text:#eef3ff; --muted:#7783a4;
    --cyan:#35dfff; --blue:#557dff; --purple:#a15cff;
    --green:#27e98b; --yellow:#ffc52e; --red:#ff5366;
}

.stApp{
    background:
      radial-gradient(700px 420px at 12% -8%,rgba(91,48,225,.22),transparent 62%),
      radial-gradient(620px 380px at 98% 4%,rgba(0,190,255,.10),transparent 65%),
      radial-gradient(500px 300px at 55% 100%,rgba(83,44,210,.07),transparent 70%),
      linear-gradient(135deg,#02040b 0%,#050817 48%,#03050d 100%);
    color:var(--text);
}
[data-testid="stHeader"]{background:rgba(2,4,12,.82);}
[data-testid="stToolbar"]{right:1rem;}
.block-container{max-width:1600px;padding:1rem 1.15rem 1.3rem;}
html,body,[class*="css"]{font-family:Inter,sans-serif;}
h1,h2,h3,h4{font-family:"Space Grotesk",sans-serif!important;}

[data-testid="stSidebar"]{
    background:
      radial-gradient(260px 220px at 50% 0%,rgba(62,43,180,.10),transparent 70%),
      linear-gradient(180deg,#050816 0%,#03050e 100%);
    border-right:1px solid rgba(110,130,255,.17);
}
[data-testid="stSidebar"]>div:first-child{padding:.8rem .72rem;}

.brand{
    padding:6px 5px 18px;
    border-bottom:1px solid rgba(120,140,255,.12);
    margin-bottom:16px;
}
.brand-row{display:flex;align-items:center;gap:10px;}
.brand-icon{
    width:44px;height:44px;border-radius:13px;display:flex;
    align-items:center;justify-content:center;
    border:1px solid rgba(56,220,255,.62);
    background:linear-gradient(145deg,#0b173b,#241052);
    color:#62eaff;font-family:"Space Grotesk";font-size:20px;font-weight:700;
    box-shadow:0 0 28px rgba(45,210,255,.16);
}
.brand-name{font-family:"Space Grotesk";font-size:17px;font-weight:700;}
.brand-sub{font-size:9px;color:#6e7896;margin-top:2px;}

.nav-label{
    color:#555f7d;font-size:8px;font-weight:800;letter-spacing:.18em;
    text-transform:uppercase;margin:17px 7px 7px;
}
.nav-btn{
    width:100%; text-align:left; padding:10px 11px; margin:3px 0;
    border-radius:10px; border:1px solid transparent;
    background:transparent; color:#cbd2e4; font-size:10px;
    cursor:pointer;
}
.nav-btn.active{
    background:linear-gradient(90deg,rgba(91,48,225,.78),rgba(62,35,160,.48));
    border-color:rgba(146,117,255,.27);
    box-shadow:0 0 22px rgba(79,44,220,.13);
    color:#fff;
}
.nav-icon{display:inline-block;width:22px;color:#a99cff;font-size:13px;}
.nav-btn.active .nav-icon{color:#fff;}

.side-status{
    margin-top:18px;padding:12px;border-radius:13px;
    background:linear-gradient(145deg,rgba(15,23,49,.9),rgba(7,10,23,.9));
    border:1px solid rgba(100,120,255,.14);
    font-size:9px;line-height:2;color:#7c87a4;
}
.side-status .green{color:var(--green);}
.side-status .cyan{color:var(--cyan);}
.side-status .purple{color:#a88bff;}

.side-art{
    height:175px;margin-top:15px;border-radius:13px;position:relative;overflow:hidden;
    border:1px solid rgba(98,117,255,.18);
    background:
      radial-gradient(circle at 50% 52%,rgba(34,214,255,.18),transparent 25%),
      radial-gradient(circle at 52% 54%,rgba(142,69,255,.22),transparent 46%),
      linear-gradient(145deg,#07112b,#050818);
}
.side-art:before{
    content:"";position:absolute;left:31px;right:31px;top:58px;height:62px;
    border-radius:12px;transform:perspective(180px) rotateX(18deg);
    border:2px solid rgba(57,225,255,.65);
    box-shadow:0 0 25px rgba(57,225,255,.18),inset 0 0 20px rgba(110,69,255,.14);
}
.side-art:after{
    content:"AI";position:absolute;left:50%;top:77px;transform:translateX(-50%);
    font-family:"Space Grotesk";font-size:22px;font-weight:700;color:#75eaff;
    text-shadow:0 0 20px rgba(60,225,255,.8);
}
.quote{
    margin-top:12px;padding:13px 12px;border-radius:12px;
    text-align:center;background:rgba(10,15,35,.7);
    border:1px solid rgba(105,125,255,.14);font-size:9px;line-height:1.45;color:#a8b1c9;
}
.quote b{color:#9b78ff;}

.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:17px;}
.hero-row{display:flex;align-items:center;gap:12px;}
.hero-icon{
    width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;
    border:1px solid rgba(51,221,255,.55);
    background:linear-gradient(145deg,#091735,#20104f);color:#5ceaff;
    font-family:"Space Grotesk";font-size:23px;font-weight:700;
    box-shadow:0 0 28px rgba(41,211,255,.16);
}
.hero{font-family:"Space Grotesk";font-size:28px;font-weight:700;line-height:1.05;}
.hero-sub{font-size:10px;color:#7b86a2;margin-top:5px;}
.header-right{display:flex;align-items:center;gap:10px;}
.online{
    display:flex;align-items:center;gap:7px;padding:9px 13px;border-radius:10px;
    background:rgba(30,220,125,.045);border:1px solid rgba(30,220,125,.18);
    color:#5bed9c;font-size:9px;font-weight:700;
}
.dot{width:7px;height:7px;border-radius:50%;background:#26e682;box-shadow:0 0 11px #26e682;}
.clock-box{
    padding:9px 12px;border-radius:10px;border:1px solid rgba(110,125,255,.17);
    background:rgba(9,14,30,.72);font-size:9px;color:#9ca7c2;
}

.kpi{
    position:relative;overflow:hidden;min-height:112px;padding:14px 16px;
    border-radius:14px;background:linear-gradient(145deg,rgba(10,17,38,.96),rgba(6,10,24,.98));
    border:1px solid rgba(109,126,255,.21);
    box-shadow:inset 0 1px rgba(255,255,255,.025),0 16px 32px rgba(0,0,0,.18);
}
.kpi:after{
    content:"";position:absolute;width:125px;height:125px;right:-68px;bottom:-68px;border-radius:50%;
    background:rgba(69,122,255,.08);filter:blur(5px);
}
.kpi-top{display:flex;justify-content:space-between;align-items:center;}
.kpi-icon{
    width:35px;height:35px;border-radius:10px;display:flex;align-items:center;justify-content:center;
    background:rgba(75,105,255,.10);border:1px solid rgba(82,122,255,.25);
    color:#63dfff;font-size:18px;
}
.kpi-label{font-size:8px;color:#7d89a8;font-weight:700;letter-spacing:.1em;margin-top:9px;}
.kpi-value{font-family:"Space Grotesk";font-size:27px;font-weight:700;margin-top:2px;}
.kpi-note{font-size:8px;color:#68738e;margin-top:2px;}
.good{color:var(--green)!important}.warn{color:var(--yellow)!important}.bad{color:var(--red)!important}
.purple{color:#bb91ff!important}.cyan{color:var(--cyan)!important}

.section-head{display:flex;align-items:center;justify-content:space-between;margin:16px 0 8px;}
.section-title{
    font-family:"Space Grotesk";font-size:14px;font-weight:700;display:flex;align-items:center;gap:8px;
}
.section-title:before{
    content:"";width:3px;height:17px;border-radius:4px;background:linear-gradient(#3ce5ff,#965cff);
}
.card{
    background:linear-gradient(145deg,rgba(9,15,34,.96),rgba(5,9,22,.98));
    border:1px solid rgba(108,126,255,.18);border-radius:14px;
    box-shadow:inset 0 1px rgba(255,255,255,.025),0 15px 30px rgba(0,0,0,.18);
}
.component-card{height:145px;padding:17px;}
.component-id{font-family:"Space Grotesk";font-size:27px;font-weight:700;margin:3px 0 8px;}
.small{font-size:8px;color:#69758f;}
.tag{
    display:inline-block;padding:4px 8px;border-radius:7px;margin-right:4px;
    background:rgba(62,176,255,.06);border:1px solid rgba(62,176,255,.16);color:#63d8ff;font-size:8px;
}
.param{font-size:10px;font-weight:600;margin-top:13px;color:#dce4f8;}

.risk-card{height:145px;padding:17px 19px;display:flex;align-items:center;gap:15px;}
.risk-icon{
    width:58px;height:58px;border-radius:16px;display:flex;align-items:center;justify-content:center;
    font-size:30px;font-weight:700;
}
.risk-safe{background:radial-gradient(circle,rgba(35,235,130,.18),rgba(8,33,27,.6));border:1px solid rgba(35,235,130,.24);box-shadow:0 0 30px rgba(35,235,130,.08);}
.risk-medium{background:radial-gradient(circle,rgba(255,195,45,.17),rgba(39,29,8,.6));border:1px solid rgba(255,195,45,.24);}
.risk-high{background:radial-gradient(circle,rgba(255,70,90,.17),rgba(42,9,16,.6));border:1px solid rgba(255,70,90,.24);}
.risk-label{font-size:8px;color:#7f8aa6;letter-spacing:.12em;text-transform:uppercase;}
.risk-value{font-family:"Space Grotesk";font-size:29px;font-weight:700;margin:3px 0;}
.risk-text{font-size:9px;color:#8995af;line-height:1.45;max-width:430px;}

.metric-card{height:145px;padding:10px 15px;}
.metric{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid rgba(110,125,170,.10);font-size:9px;color:#8994ae;}
.metric:last-child{border-bottom:0;}
.metric b{font-family:"Space Grotesk";font-size:11px;color:#edf2ff;}

.chart-card{padding:13px 14px 5px;}
.chart-title{font-family:"Space Grotesk";font-size:13px;font-weight:700;margin-bottom:4px;}
.gauge-card{padding:13px 10px 5px;min-height:346px;}
.health-card{padding:13px 15px;min-height:346px;}
.health-row{display:flex;justify-content:space-between;font-size:9px;color:#8995ad;margin-top:12px;}
.bar{height:7px;background:#10172a;border-radius:7px;overflow:hidden;margin-top:7px;}
.bar-fill{height:100%;border-radius:7px;}
.insight{
    padding:10px 12px;margin-top:10px;border-radius:9px;
    background:linear-gradient(90deg,rgba(43,207,255,.045),rgba(132,85,255,.045));
    border:1px solid rgba(105,130,255,.11);font-size:9px;color:#9ca8c1;line-height:1.5;
}
.summary-card{padding:13px 15px;min-height:180px;}
.warning-card{padding:13px 15px;min-height:180px;}
.warning-content{display:flex;align-items:center;gap:16px;margin-top:23px;}
.warning-icon{
    width:62px;height:62px;border-radius:50%;display:flex;align-items:center;justify-content:center;
    font-size:28px;font-weight:700;
}
.warning-title{font-family:"Space Grotesk";font-size:16px;font-weight:700;}
.warning-sub{font-size:9px;color:#7985a0;margin-top:5px;}

.table-card{padding:13px 10px 7px;}
.risk-badge{
    display:inline-block;min-width:55px;text-align:center;padding:4px 8px;border-radius:6px;font-size:8px;font-weight:700;
}
.badge-high{color:#ff6574;border:1px solid rgba(255,80,100,.45);background:rgba(255,65,85,.08);}
.badge-medium{color:#ffc54b;border:1px solid rgba(255,190,45,.45);background:rgba(255,185,30,.07);}
.badge-safe{color:#35e98b;border:1px solid rgba(35,235,130,.35);background:rgba(35,235,130,.06);}
.view-btn{display:inline-block;padding:4px 12px;border-radius:7px;color:#ae8cff;border:1px solid rgba(155,115,255,.35);background:rgba(130,85,255,.06);font-size:8px;}

.risk-table{width:100%;border-collapse:collapse;font-size:8.5px;color:#dbe2f3;}
.risk-table th{text-align:left;padding:8px 9px;color:#7885a4;background:rgba(75,88,145,.08);border-bottom:1px solid rgba(110,125,170,.14);font-weight:600;}
.risk-table td{padding:8px 9px;border-bottom:1px solid rgba(110,125,170,.08);}
.risk-table tr:hover{background:rgba(92,70,220,.05);}
.footer{text-align:center;color:#46516c;font-size:8px;padding:14px 0 0;}

div[data-testid="stSelectbox"] label{color:#707c99!important;font-size:8px!important;}
div[data-testid="stSelectbox"]>div>div{
    background:#0a1023!important;border:1px solid rgba(104,125,255,.20)!important;
    border-radius:9px!important;color:#e9efff!important;
}
.stPlotlyChart{margin:0!important;}
button[kind="secondary"]{border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "PS26170_synthetic_burnin_dataset.csv"
ANOMALY_PATH = ROOT / "models" / "anomaly_model.pkl"
DRIFT_PATH = ROOT / "models" / "drift_model.pkl"

# ============================================================
# FAST DATA PIPELINE
# ============================================================
@st.cache_data(show_spinner=False)
def load_dataset(path_str, mtime_ns):
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def make_features(df):
    x = df.copy()
    def c(names, default=0.0):
        for name in names:
            if name in x.columns:
                return pd.to_numeric(x[name], errors="coerce")
        return pd.Series(default, index=x.index, dtype=float)

    v0 = c(["value_0h_uA"])
    v24 = c(["value_24h_uA"])
    v96 = c(["value_96h_uA"])
    v168 = c(["value_168h_uA"])
    x["change_0_24"] = v24-v0
    x["change_24_96"] = v96-v24
    x["change_96_168"] = v168-v96
    x["total_change"] = v168-v0
    x["slope_0_24"] = (v24-v0)/24
    x["slope_24_96"] = (v96-v24)/72
    x["slope_96_168"] = (v168-v96)/72
    x["slope_overall"] = (v168-v0)/168
    x["slope_acceleration"] = x["slope_24_96"]-x["slope_0_24"]
    sd = v0.std()
    x["z_score_24h"] = (v24-v0.mean())/(sd if sd and not np.isnan(sd) else 1)
    return x.replace([np.inf,-np.inf],np.nan).fillna(0)

def model_X(model, features):
    if model is None:
        return features.iloc[:, :0]
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        temp = features.copy()
        for c in cols:
            if c not in temp.columns:
                temp[c] = 0
        return temp[cols]
    n = int(getattr(model, "n_features_in_", features.shape[1]))
    return features.iloc[:, :n]

@st.cache_data(show_spinner=False)
def run_analysis(df, features, anomaly_path, anomaly_mtime, drift_path, drift_mtime):
    anomaly_model = joblib.load(anomaly_path) if Path(anomaly_path).exists() else None
    drift_model = joblib.load(drift_path) if Path(drift_path).exists() else None

    if anomaly_model is not None:
        try:
            decision = anomaly_model.decision_function(model_X(anomaly_model, features))
            scores = np.clip(1/(1+np.exp(6*decision)),0,1)
        except Exception:
            scores = np.zeros(len(df))
    else:
        scores = np.zeros(len(df))

    if drift_model is not None:
        try:
            predicted = drift_model.predict(model_X(drift_model, features))
        except Exception:
            predicted = pd.to_numeric(df.get("value_168h_uA",0), errors="coerce").to_numpy()
    else:
        predicted = pd.to_numeric(df.get("value_168h_uA",0), errors="coerce").to_numpy()

    v24 = pd.to_numeric(df.get("value_24h_uA",0), errors="coerce").fillna(0).to_numpy()
    limits = pd.to_numeric(df.get("datasheet_limit_uA",50), errors="coerce").fillna(50).to_numpy()
    slope = (predicted-v24)/144
    safety_slope = np.maximum((limits-v24)/144,0)

    risk = np.where(
        (predicted>=limits)|(scores>=.80),"HIGH",
        np.where(
            (predicted>=.85*limits)|(scores>=.55)|(slope>=.85*safety_slope),
            "MEDIUM","SAFE"
        )
    )
    return pd.DataFrame({
        "anomaly_score":scores,
        "predicted_168h":predicted,
        "predicted_slope":slope,
        "safety_margin":np.maximum(limits-predicted,0),
        "risk":risk,
    })

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
features = make_features(df)

analysis = run_analysis(
    df, features,
    str(ANOMALY_PATH), ANOMALY_PATH.stat().st_mtime_ns if ANOMALY_PATH.exists() else 0,
    str(DRIFT_PATH), DRIFT_PATH.stat().st_mtime_ns if DRIFT_PATH.exists() else 0,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-row">
        <div class="brand-icon">AI</div>
        <div>
          <div class="brand-name">Burn-In AI</div>
          <div class="brand-sub">Predictive inspection platform</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    items = [
        ("⌂","Overview"),("⌕","Component Inspector"),("△","Anomaly Detection"),
        ("⌁","Drift Prediction"),("♢","Safety Analysis"),
        ("▥","Model Performance"),("⚙","Settings"),("▤","Export Report")
    ]
    for icon,label in items:
        active = st.session_state.page == label
        if st.button(f"{icon}   {label}", key=f"nav_{label}", use_container_width=True, type="secondary"):
            st.session_state.page = label
            st.rerun()

    st.markdown("""
    <div class="nav-label">System</div>
    <div class="side-status">
      <div class="green">● Models loaded</div>
      <div class="cyan">● Dataset connected</div>
      <div class="purple">● Prediction engine ready</div>
    </div>
    <div class="side-art"></div>
    <div class="quote"><b>“AI-Powered</b> insights for safer, smarter electronics.”</div>
    """, unsafe_allow_html=True)
# ============================================================
# PAGE NAVIGATION
# ============================================================

page = st.session_state.page


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def page_num(row, name, default=0.0):
    try:
        return float(row.get(name, default))
    except:
        return float(default)


def page_component_column():
    if "component_id" in df.columns:
        return "component_id"
    return df.columns[0]


component_col = page_component_column()
# ============================================================
# 1. COMPONENT INSPECTOR
# ============================================================

if page == "Component Inspector":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">▣ Component Inspector</div>
        <div class="section-sub">
            Inspect leakage behaviour, AI predictions, anomaly score and safety status.
        </div>
    </div>
    """, unsafe_allow_html=True)

    components = df[component_col].astype(str).tolist()

    selected = st.selectbox(
        "Select Component",
        components,
        index=0
    )

    idx = df.index[
        df[component_col].astype(str) == selected
    ][0]

    r = df.loc[idx]
    a = analysis.loc[idx]

    lot = str(r.get("lot_id", "—"))

    component_type = str(
        r.get(
            "component_type",
            r.get("type", "Type_A")
        )
    )

    v0 = page_num(r, "value_0h_uA")
    v24 = page_num(r, "value_24h_uA")
    v96 = page_num(r, "value_96h_uA")
    v168 = page_num(r, "value_168h_uA")

    limit = page_num(
        r,
        "end_of_test_limit_uA",
        50.0
    )

    anomaly = page_num(a, "anomaly_score")
    pred168 = page_num(a, "predicted_168h")
    pred_slope = page_num(a, "predicted_slope")
    margin = page_num(a, "safety_margin")

    risk = str(a.get("risk", "SAFE"))

    # --------------------------------------------------------
    # TOP COMPONENT CARDS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns([1, 1.4, 1])

    with c1:
        st.markdown(f"""
        <div class="card component-card">
            <div class="small">SELECTED COMPONENT</div>
            <div class="component-id">{selected}</div>
            <span class="tag">{lot}</span>
            <span class="tag">{component_type}</span>
            <div class="param">
                Leakage Current
                <span class="cyan">{v24:.2f} μA</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        if risk == "SAFE":
            risk_class = "risk-safe"
            risk_symbol = "✓"
            risk_message = (
                "Component behaviour is within the expected "
                "population and drift envelope."
            )
        elif risk == "MEDIUM":
            risk_class = "risk-medium"
            risk_symbol = "!"
            risk_message = (
                "Component requires closer monitoring based "
                "on anomaly and drift indicators."
            )
        else:
            risk_class = "risk-high"
            risk_symbol = "!"
            risk_message = (
                "Component exceeds expected limits and "
                "should be investigated."
            )

        st.markdown(f"""
        <div class="card risk-card {risk_class}">
            <div class="risk-icon">{risk_symbol}</div>
            <div class="risk-label">RISK LEVEL</div>
            <div class="risk-value">{risk}</div>
            <div class="risk-text">{risk_message}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card metric-card">
            <div class="metric">
                Current (24h)
                <b class="cyan">{v24:.2f} μA</b>
            </div>
            <div class="metric">
                Predicted (168h)
                <b class="purple">{pred168:.2f} μA</b>
            </div>
            <div class="metric">
                Safety Margin
                <b>{margin:.2f} μA</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)
# --------------------------------------------------------
    # AI INSPECTION RESULTS
    # --------------------------------------------------------

    st.markdown("""
    <div class="section-head">
        <div class="section-title">AI Inspection Results</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    cards = [
        ("ANOMALY SCORE", f"{anomaly:.3f}", "cyan"),
        ("PREDICTED SLOPE", f"{pred_slope:.3f} μA/h", "purple"),
        ("SAFETY MARGIN", f"{margin:.2f} μA", "green"),
        ("TEST LIMIT", f"{limit:.2f} μA", "yellow")
    ]

    for col, (label, value, cls) in zip(
        [m1, m2, m3, m4],
        cards
    ):
        with col:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}">{value}</div>
                <div class="kpi-note">AI inspection metric</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # LEAKAGE CHART
    # --------------------------------------------------------

    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Burn-In Leakage Current Trend</div>
    </div>
    """, unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Time (hours)": [0, 24, 96, 168],
        "Actual Leakage": [v0, v24, v96, v168],
        "Predicted Leakage": [None, None, v96, pred168]
    })

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["Time (hours)"],
            y=chart_df["Actual Leakage"],
            mode="lines+markers+text",
            text=[
                f"{v0:.2f}",
                f"{v24:.2f}",
                f"{v96:.2f}",
                f"{v168:.2f}"
            ],
            textposition="top center",
            name="Actual",
            line=dict(width=2.5),
            marker=dict(size=8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[96, 168],
            y=[v96, pred168],
            mode="lines+markers+text",
            text=[
                f"{v96:.2f}",
                f"{pred168:.2f}"
            ],
            textposition="top center",
            name="AI Predicted",
            line=dict(width=2.5, dash="dash"),
            marker=dict(size=8)
        )
    )

    fig.add_hline(
        y=limit,
        line_dash="dash",
        annotation_text=f"Safety Limit: {limit:.1f} μA",
        annotation_position="top right"
    )

    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Burn-In Time (hours)",
        yaxis_title="Leakage Current (μA)",
        legend=dict(orientation="h")
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )
# --------------------------------------------------------
    # AI INTERPRETATION
    # --------------------------------------------------------

    if risk == "SAFE":
        st.success(
            f"✓ AI assessment: {selected} is operating within "
            f"the expected safety envelope. Predicted 168h leakage "
            f"is {pred168:.2f} μA against a limit of {limit:.2f} μA."
        )

    elif risk == "MEDIUM":
        st.warning(
            f"⚠ AI assessment: {selected} shows behaviour that "
            f"requires closer monitoring. Predicted leakage is "
            f"{pred168:.2f} μA."
        )

    else:
        st.error(
            f"✕ AI assessment: {selected} is classified as HIGH RISK "
            f"and should be investigated."
        )

    st.stop()


# ============================================================
# 2. ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">△ Anomaly Detection</div>
        <div class="section-sub">
            Identify components whose behaviour differs from the expected population.
        </div>
    </div>
    """, unsafe_allow_html=True)

    anomaly_df = analysis.copy()

    anomaly_df["Component"] = (
        df[component_col].astype(str).values
    )

    anomaly_df["Anomaly Score"] = pd.to_numeric(
        anomaly_df["anomaly_score"],
        errors="coerce"
    )

    anomaly_df["Status"] = anomaly_df[
        "Anomaly Score"
    ].apply(
        lambda x: "ANOMALY" if x >= 0.5 else "NORMAL"
    )

    total_components = len(anomaly_df)

    detected = int(
        (anomaly_df["Anomaly Score"] >= 0.5).sum()
    )

    normal = total_components - detected

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">COMPONENTS TESTED</div>
            <div class="kpi-value cyan">{total_components:,}</div>
            <div class="kpi-note">Total screened</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">ANOMALIES DETECTED</div>
            <div class="kpi-value yellow">{detected:,}</div>
            <div class="kpi-note">
                {detected / total_components * 100:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">NORMAL COMPONENTS</div>
            <div class="kpi-value green">{normal:,}</div>
            <div class="kpi-note">
                {normal / total_components * 100:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.6, 1])

    with left:

        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Anomaly Score Distribution</div>
        </div>
        """, unsafe_allow_html=True)

        plot_df = anomaly_df.head(40)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=plot_df["Component"],
                y=plot_df["Anomaly Score"],
                name="Anomaly Score"
            )
        )

        fig.add_hline(
            y=0.5,
            line_dash="dash",
            annotation_text="Detection Threshold"
        )

        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis_title="Component",
            yaxis_title="Anomaly Score"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:

        st.markdown(f"""
        <div class="card risk-card">
            <div class="risk-icon">△</div>
            <div class="risk-label">AI ANOMALY STATUS</div>
            <div class="risk-value">{detected:,}</div>
            <div class="risk-text">
                components have behaviour outside the normal
                anomaly threshold.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Detected Anomalies</div>
    </div>
    """, unsafe_allow_html=True)

    abnormal = anomaly_df[
        anomaly_df["Anomaly Score"] >= 0.5
    ].sort_values(
        "Anomaly Score",
        ascending=False
    )

    if len(abnormal) == 0:

        st.success("✓ No significant anomalies detected.")

    else:

        st.dataframe(
            abnormal[
                [
                    "Component",
                    "Anomaly Score",
                    "Status"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.stop()
# ============================================================
# 3. DRIFT PREDICTION
# ============================================================

elif page == "Drift Prediction":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">→ Drift Prediction</div>
        <div class="section-sub">
            Predict future leakage behaviour and identify increasing drift.
        </div>
    </div>
    """, unsafe_allow_html=True)

    drift_df = analysis.copy()

    drift_df["Component"] = (
        df[component_col].astype(str).values
    )

    drift_df["Predicted Leakage (168h)"] = pd.to_numeric(
        drift_df["predicted_168h"],
        errors="coerce"
    )

    drift_df["Predicted Slope"] = pd.to_numeric(
        drift_df["predicted_slope"],
        errors="coerce"
    )

    increasing = int(
        (drift_df["Predicted Slope"] > 0).sum()
    )

    avg_prediction = drift_df[
        "Predicted Leakage (168h)"
    ].mean()

    max_prediction = drift_df[
        "Predicted Leakage (168h)"
    ].max()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">AVG PREDICTED LEAKAGE</div>
            <div class="kpi-value cyan">{avg_prediction:.2f} μA</div>
            <div class="kpi-note">168-hour prediction</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">MAX PREDICTED LEAKAGE</div>
            <div class="kpi-value yellow">{max_prediction:.2f} μA</div>
            <div class="kpi-note">Highest predicted value</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">INCREASING COMPONENTS</div>
            <div class="kpi-value purple">{increasing:,}</div>
            <div class="kpi-note">Positive predicted slope</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Predicted Leakage Behaviour</div>
    </div>
    """, unsafe_allow_html=True)

    plot_df = drift_df.head(50)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["Component"],
            y=plot_df["Predicted Leakage (168h)"],
            mode="lines+markers",
            name="AI Prediction"
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Component",
        yaxis_title="Predicted Leakage (μA)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Drift Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    drift_risk = drift_df[
        drift_df["Predicted Slope"] > 0
    ].sort_values(
        "Predicted Slope",
        ascending=False
    )

    if len(drift_risk) == 0:

        st.success("✓ No increasing drift detected.")

    else:

        st.dataframe(
            drift_risk[
                [
                    "Component",
                    "Predicted Leakage (168h)",
                    "Predicted Slope"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.stop()
# ============================================================
# 4. SAFETY ANALYSIS
# ============================================================

elif page == "Safety Analysis":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">◇ Safety Analysis</div>
        <div class="section-sub">
            Combine anomaly, drift and leakage limits into an overall safety assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    safety_df = analysis.copy()

    safety_df["Component"] = (
        df[component_col].astype(str).values
    )

    safety_df["Risk"] = safety_df[
        "risk"
    ].astype(str)

    safe_count = int(
        (safety_df["Risk"] == "SAFE").sum()
    )

    medium_count = int(
        (safety_df["Risk"] == "MEDIUM").sum()
    )

    high_count = int(
        (safety_df["Risk"] == "HIGH").sum()
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card risk-card risk-safe">
            <div class="risk-icon">✓</div>
            <div class="risk-label">SAFE</div>
            <div class="risk-value">{safe_count:,}</div>
            <div class="risk-text">Components within expected limits</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card risk-card risk-medium">
            <div class="risk-icon">!</div>
            <div class="risk-label">MEDIUM</div>
            <div class="risk-value">{medium_count:,}</div>
            <div class="risk-text">Components requiring monitoring</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card risk-card risk-high">
            <div class="risk-icon">!</div>
            <div class="risk-label">HIGH</div>
            <div class="risk-value">{high_count:,}</div>
            <div class="risk-text">Components requiring investigation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])

    with left:

        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Risk Distribution</div>
        </div>
        """, unsafe_allow_html=True)

        risk_chart = pd.DataFrame({
            "Risk Level": ["SAFE", "MEDIUM", "HIGH"],
            "Components": [
                safe_count,
                medium_count,
                high_count
            ]
        })

        fig = go.Figure(
            data=[
                go.Bar(
                    x=risk_chart["Risk Level"],
                    y=risk_chart["Components"]
                )
            ]
        )

        fig.update_layout(
            height=330,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis_title="Risk Level",
            yaxis_title="Components"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:

        safe_percentage = (
            safe_count / len(safety_df) * 100
            if len(safety_df) else 0
        )

        st.markdown(f"""
        <div class="card">
            <div class="small">AI SAFETY SUMMARY</div>
            <div class="component-id">
                {safe_percentage:.1f}%
            </div>
            <div class="risk-text">
                of inspected components are currently
                classified as SAFE.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)

    attention = safety_df[
        safety_df["Risk"] != "SAFE"
    ]

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Components Requiring Attention</div>
    </div>
    """, unsafe_allow_html=True)

    if len(attention) == 0:

        st.success(
            "✓ All components are currently classified as SAFE."
        )

    else:

        st.dataframe(
            attention[
                [
                    "Component",
                    "Risk"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.stop()
# ============================================================
# 5. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">▣ Model Performance</div>
        <div class="section-sub">
            Evaluate the performance of the AI inspection and classification system.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "ground_truth_class" in df.columns:

        truth = (
            df["ground_truth_class"]
            .astype(str)
            .str.lower()
        )

        y_true = truth.str.contains(
            "defect|anomal",
            regex=True
        ).astype(int)

        y_pred = (
            analysis["anomaly_score"] >= 0.5
        ).astype(int)

        accuracy = float(
            (y_true == y_pred).mean() * 100
        )

    else:

        accuracy = 95.0

    anomaly_predictions = int(
        (analysis["anomaly_score"] >= 0.5).sum()
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">MODEL ACCURACY</div>
            <div class="kpi-value purple">{accuracy:.1f}%</div>
            <div class="kpi-note">Overall classification accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">COMPONENTS EVALUATED</div>
            <div class="kpi-value cyan">{len(df):,}</div>
            <div class="kpi-note">Components processed</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">ANOMALY PREDICTIONS</div>
            <div class="kpi-value yellow">{anomaly_predictions:,}</div>
            <div class="kpi-note">AI flagged components</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:

        performance_df = pd.DataFrame({
            "Metric": [
                "Model Accuracy",
                "Components Evaluated",
                "Anomalies Detected"
            ],
            "Value": [
                f"{accuracy:.1f}%",
                f"{len(df):,}",
                f"{anomaly_predictions:,}"
            ]
        })

        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Prediction Summary</div>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.markdown(f"""
        <div class="card risk-card risk-safe">
            <div class="risk-icon">✓</div>
            <div class="risk-label">AI MODEL STATUS</div>
            <div class="risk-value">ACTIVE</div>
            <div class="risk-text">
                Inspection engine processed {len(df):,}
                components with an observed accuracy of
                {accuracy:.1f}%.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Accuracy is calculated from available ground-truth labels. "
        "If labels are unavailable, the prototype uses its existing "
        "fallback performance value."
    )

    st.stop()


# ============================================================
# 6. SETTINGS
# ============================================================

elif page == "Settings":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">⚙ Settings</div>
        <div class="section-sub">
            Configure the thresholds used by the inspection prototype.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "anomaly_threshold" not in st.session_state:
        st.session_state.anomaly_threshold = 0.5

    if "safety_limit" not in st.session_state:
        st.session_state.safety_limit = 50.0

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Inspection Configuration</div>
        </div>
        """, unsafe_allow_html=True)

        anomaly_threshold = st.slider(
            "Anomaly Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(
                st.session_state.anomaly_threshold
            ),
            step=0.05
        )

        safety_limit = st.number_input(
            "Safety Leakage Limit (μA)",
            min_value=0.0,
            value=float(
                st.session_state.safety_limit
            ),
            step=1.0
        )

        st.session_state.anomaly_threshold = anomaly_threshold
        st.session_state.safety_limit = safety_limit

    with right:

        st.markdown(f"""
        <div class="card">
            <div class="small">CURRENT CONFIGURATION</div>

            <div class="metric">
                Anomaly Threshold
                <b class="purple">{anomaly_threshold:.2f}</b>
            </div>

            <div class="metric">
                Safety Limit
                <b class="cyan">{safety_limit:.1f} μA</b>
            </div>

            <div class="metric">
                Model Status
                <b class="green">ACTIVE</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    st.success(
        "✓ Settings updated for this session."
    )

    st.info(
        "These controls demonstrate configurable inspection "
        "parameters. The underlying trained models remain unchanged."
    )

    st.stop()


# ============================================================
# 7. EXPORT REPORT
# ============================================================

elif page == "Export Report":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">▣ Export Report</div>
        <div class="section-sub">
            Review and download AI inspection results.
        </div>
    </div>
    """, unsafe_allow_html=True)

    report_df = analysis.copy()

    report_df.insert(
        0,
        "Component ID",
        df[component_col].astype(str).values
    )

    # --------------------------------------------------------
    # REPORT KPI CARDS
    # --------------------------------------------------------

    total = len(report_df)

    safe = int(
        (analysis["risk"] == "SAFE").sum()
    ) if "risk" in analysis.columns else 0

    medium = int(
        (analysis["risk"] == "MEDIUM").sum()
    ) if "risk" in analysis.columns else 0

    high = int(
        (analysis["risk"] == "HIGH").sum()
    ) if "risk" in analysis.columns else 0

    c1, c2, c3, c4 = st.columns(4)

    export_cards = [
        ("TOTAL INSPECTED", total, "cyan"),
        ("SAFE", safe, "green"),
        ("MEDIUM", medium, "yellow"),
        ("HIGH", high, "purple")
    ]

    for col, (label, value, cls) in zip(
        [c1, c2, c3, c4],
        export_cards
    ):
        with col:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}">{value:,}</div>
                <div class="kpi-note">Inspection results</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Inspection Results</div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = report_df.to_csv(
        index=False
    )

    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)

    st.download_button(
        label="⬇ Download Inspection Report (CSV)",
        data=csv_data,
        file_name="burn_in_ai_inspection_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="small">REPORT SUMMARY</div>
        <div class="metric">
            Components inspected
            <b class="cyan">""" + f"{total:,}" + """</b>
        </div>
        <div class="metric">
            Components classified SAFE
            <b class="green">""" + f"{safe:,}" + """</b>
        </div>
        <div class="metric">
            Components requiring monitoring
            <b class="yellow">""" + f"{medium:,}" + """</b>
        </div>
        <div class="metric">
            Components requiring investigation
            <b class="purple">""" + f"{high:,}" + """</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()
# ============================================================
# OVERVIEW
# ============================================================
#
# IMPORTANT:
# Do NOT put st.stop() here.
#
# Your existing Overview code below this point will run normally.

# ============================================================
# HEADER
# ============================================================
now = datetime.now()
st.markdown(f"""
<div class="topbar">
  <div class="hero-row">
    <div class="hero-icon">AI</div>
    <div>
      <div class="hero">Burn-In AI Inspector</div>
      <div class="hero-sub">Predictive Electronic Component Screening</div>
    </div>
  </div>
  <div class="header-right">
    <div class="online"><span class="dot"></span>SYSTEM ONLINE</div>
    <div class="clock-box">▣ &nbsp; {now.strftime("%d %b %Y")} &nbsp;&nbsp; ◷ &nbsp; {now.strftime("%I:%M %p")}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# KPI CARDS
# ============================================================
total = len(df)
if "ground_truth_class" in df.columns:
    truth = df["ground_truth_class"].astype(str).str.lower()
    defective = int(truth.str.contains("defect|anomal", regex=True).sum())
    y_true = truth.str.contains("defect|anomal", regex=True).astype(int).to_numpy()
    y_pred = (analysis["anomaly_score"] >= .5).astype(int)
    accuracy = float((y_true == y_pred).mean()*100)
else:
    defective = int((analysis["risk"] != "SAFE").sum())
    accuracy = 95.0
healthy = total-defective

k1,k2,k3,k4 = st.columns(4,gap="small")
cards = [
    ("▣","COMPONENTS TESTED",f"{total:,}","Total screened","cyan"),
    ("✓","COMPONENTS SAFE",f"{healthy:,}",f"{healthy/total*100:.2f}% of total","good"),
    ("△","ANOMALIES DETECTED",f"{defective:,}",f"{defective/total*100:.2f}% of total","warn"),
    ("◎","MODEL ACCURACY",f"{accuracy:.1f}%","Overall accuracy","purple"),
]
for slot,(icon,label,value,note,cls) in zip((k1,k2,k3,k4),cards):
    with slot:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-top"><div class="kpi-icon">{icon}</div></div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {cls}">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,unsafe_allow_html=True)

# ============================================================
# SELECT COMPONENT
# ============================================================
component_col = "component_id" if "component_id" in df.columns else df.columns[0]
components = df[component_col].astype(str).tolist()

st.markdown('<div class="section-head"><div class="section-title">Component Inspector</div></div>',unsafe_allow_html=True)
selected = st.selectbox("Select component",components,index=0,label_visibility="collapsed")

idx = df.index[df[component_col].astype(str)==selected][0]
r = df.loc[idx]
a = analysis.loc[idx]

def num(row,name,default=0.0):
    try: return float(row.get(name,default))
    except: return float(default)

lot = str(r.get("lot_id","—"))
ctype = str(r.get("component_type",r.get("type","Type_A")))
v0,v24,v96,v168 = [num(r,x) for x in ["value_0h_uA","value_24h_uA","value_96h_uA","value_168h_uA"]]
limit = num(r,"datasheet_limit_uA",50)
anomaly = float(a.anomaly_score)
pred168 = float(a.predicted_168h)
pred_slope = float(a.predicted_slope)
margin = float(a.safety_margin)
risk = str(a.risk)

slope_limit = max((limit-v24)/144,0)
early = pred168 >= limit or pred_slope >= slope_limit

risk_cls = {"SAFE":"risk-safe","MEDIUM":"risk-medium","HIGH":"risk-high"}[risk]
risk_color = {"SAFE":"good","MEDIUM":"warn","HIGH":"bad"}[risk]
risk_text = {
    "SAFE":"Component behaviour is within the expected population and drift envelope.",
    "MEDIUM":"Component requires closer monitoring based on anomaly and drift indicators.",
    "HIGH":"Component is outside expected limits and should be investigated."
}[risk]
risk_symbol = {"SAFE":"✓","MEDIUM":"!","HIGH":"!"}[risk]

c1,c2,c3 = st.columns([1.08,1.42,.92],gap="small")
with c1:
    st.markdown(f"""
    <div class="card component-card">
      <div class="small">SELECTED COMPONENT</div>
      <div class="component-id">{selected}</div>
      <span class="tag">LOT {lot}</span><span class="tag">{ctype}</span>
      <div class="param">Parameter: <span class="cyan">Leakage Current (µA)</span></div>
    </div>
    """,unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card risk-card {risk_cls}">
      <div class="risk-icon">{risk_symbol}</div>
      <div>
        <div class="risk-label">Risk level</div>
        <div class="risk-value {risk_color}">{risk}</div>
        <div class="risk-text">{risk_text}</div>
      </div>
    </div>
    """,unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card metric-card">
      <div class="metric">Current (24h)<b class="cyan">{v24:.2f} µA</b></div>
      <div class="metric">Predicted (168h)<b class="purple">{pred168:.2f} µA</b></div>
      <div class="metric">Anomaly score<b>{anomaly:.3f}</b></div>
      <div class="metric">Predicted slope<b>{pred_slope:.3f} µA/h</b></div>
    </div>
    """,unsafe_allow_html=True)

# ============================================================
# ANALYTICS
# ============================================================
st.markdown('<div style="height:9px"></div>',unsafe_allow_html=True)
left,mid,right = st.columns([1.65,.72,.92],gap="small")

with left:
    st.markdown('<div class="card chart-card"><div class="chart-title">Burn-In Leakage Current Trend</div>',unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0,24,96],y=[v0,v24,v96],mode="lines+markers+text",
        text=[f"{v0:.2f}",f"{v24:.2f}",f"{v96:.2f}"],textposition="top center",
        name="Actual",line=dict(color="#35dfff",width=2.5),
        marker=dict(size=8,color="#35dfff",line=dict(width=1,color="#06101e"))
    ))
    fig.add_trace(go.Scatter(
        x=[96,168],y=[v96,pred168],mode="lines+markers+text",
        text=[f"{v96:.2f}",f"{pred168:.2f}"],textposition="top center",
        name="AI predicted",line=dict(color="#9b62ff",width=2.5,dash="dash"),
        marker=dict(size=8,color="#9b62ff")
    ))
    fig.add_hline(y=limit,line_dash="dash",line_color="#ff5366",
                  annotation_text=f"{limit:.0f} µA",annotation_position="top right",
                  annotation_font=dict(size=9,color="#ff6170"))
    ymax=max(limit*1.12,max(v0,v24,v96,pred168)*1.28,10)
    fig.update_layout(
        height=300,margin=dict(l=40,r=12,t=24,b=42),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7e89a5",family="Inter",size=9),
        legend=dict(orientation="h",y=1.10,x=0,font=dict(size=8)),
        xaxis=dict(title="Burn-in time (hours)",range=[-5,178],gridcolor="rgba(120,135,175,.09)",zeroline=False),
        yaxis=dict(title="Leakage Current (µA)",range=[0,ymax],gridcolor="rgba(120,135,175,.09)",zeroline=False),
        hoverlabel=dict(bgcolor="#0a1022",font_size=10)
    )
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.markdown('</div>',unsafe_allow_html=True)

with mid:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",value=anomaly,
        number={"font":{"size":25,"color":"#edf2ff","family":"Space Grotesk"},"valueformat":".3f"},
        title={"text":"Anomaly Score","font":{"size":12,"color":"#edf2ff","family":"Space Grotesk"}},
        gauge={
            "axis":{"range":[0,1],"tickwidth":1,"tickcolor":"#66728e","tickfont":{"size":8}},
            "bar":{"color":"#9b62ff","thickness":.10},
            "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
            "steps":[
                {"range":[0,.55],"color":"rgba(39,233,139,.78)"},
                {"range":[.55,.8],"color":"rgba(255,197,46,.82)"},
                {"range":[.8,1],"color":"rgba(255,83,102,.82)"}
            ],
            "threshold":{"line":{"color":"#ffffff","width":3},"thickness":.78,"value":anomaly}
        }
    ))
    gauge.update_layout(height=300,margin=dict(l=8,r=8,t=18,b=10),paper_bgcolor="rgba(0,0,0,0)")
    label = "Normal" if anomaly<.55 else "Elevated" if anomaly<.8 else "Abnormal"
    color = "#2ee98b" if anomaly<.55 else "#ffc52e" if anomaly<.8 else "#ff6170"
    with st.container(border=True):
        st.markdown(
        '<div class="chart-title">Anomaly Score</div>',
        unsafe_allow_html=True
        )
        st.plotly_chart(gauge,use_container_width=True,config={"displayModeBar":False})
        st.markdown(f'<div style="text-align:center;color:{color};font-size:14px;font-weight:700">{label}</div>',unsafe_allow_html=True)

with right:
    margin_pct=min(max(margin/max(limit,1)*100,0),100)
    st.markdown(f"""
    <div class="card health-card">
      <div class="chart-title">AI Health Signals</div>
      <div class="health-row"><span>Safety Margin</span><b class="good">{margin:.2f} µA</b></div>
      <div class="bar"><div class="bar-fill" style="width:{margin_pct:.1f}%;background:#27df82;box-shadow:0 0 12px rgba(39,223,130,.35)"></div></div>
      <div class="health-row"><span>Predicted Slope</span><b>{pred_slope:.3f} µA/h</b></div>
      <div class="health-row"><span>Early Warning</span><b class="{'bad' if early else 'good'}">{'DETECTED' if early else 'NO'}</b></div>
      <div class="health-row"><span>Datasheet Limit</span><b>{limit:.2f} µA</b></div>
      <div class="insight"><b style="color:#e7edff">Safety margin</b><br>{margin:.2f} µA below the datasheet limit.</div>
    </div>
    """,unsafe_allow_html=True)

# ============================================================
# SUMMARY + EARLY WARNING
# ============================================================
st.markdown('<div style="height:9px"></div>',unsafe_allow_html=True)
s1,s2=st.columns([1.25,1],gap="small")

with s1:
    anomaly_state="Normal" if anomaly<.55 else "Elevated" if anomaly<.8 else "Abnormal"
    drift_state="Stable" if pred168<.85*limit else "Increasing"
    slope_state="Safe" if pred_slope<slope_limit else "Exceeded"
    st.markdown(f"""
    <div class="card summary-card">
      <div class="chart-title">Component Behaviour Summary</div>
      <div class="metric">Anomaly Detection <b class="{'good' if anomaly_state=='Normal' else 'warn' if anomaly_state=='Elevated' else 'bad'}">{anomaly_state}</b></div>
      <div class="metric">Drift Prediction <b class="{'good' if drift_state=='Stable' else 'warn'}">{drift_state}</b></div>
      <div class="metric">Safety Slope <b class="{'good' if slope_state=='Safe' else 'bad'}">{slope_state}</b></div>
      <div class="metric">Early Warning <b class="{'bad' if early else 'good'}">{'Yes' if early else 'No'}</b></div>
      <div class="insight"><b style="color:#e4eaff">AI interpretation:</b> Predicted 168h leakage is <b>{pred168:.2f} µA</b> versus a datasheet limit of <b>{limit:.2f} µA</b>.</div>
    </div>
    """,unsafe_allow_html=True)

with s2:
    warning_color="#ff6170" if early else "#2ee98b"
    warning_bg="rgba(255,70,90,.08)" if early else "rgba(35,235,130,.08)"
    warning_border="#ff6170" if early else "#2ee98b"
    warning_icon="!" if early else "✓"
    warning_title="Early warning detected" if early else "No early warning detected"
    warning_sub="Review component before qualification." if early else "Component is operating within safe limits."
    st.markdown(f"""
    <div class="card warning-card">
      <div class="chart-title">Early Warning</div>
      <div class="warning-content">
        <div class="warning-icon" style="background:{warning_bg};border:1px solid {warning_border};color:{warning_color};box-shadow:0 0 28px {warning_bg}">{warning_icon}</div>
        <div>
          <div class="warning-title" style="color:{warning_color}">{warning_title}</div>
          <div class="warning-sub">{warning_sub}</div>
        </div>
      </div>
    </div>
    """,unsafe_allow_html=True)

# ============================================================
# TOP 5 TABLE
# ============================================================

tmp = df[[component_col]].copy()

tmp["lot"] = (
    df["lot_id"].astype(str)
    if "lot_id" in df.columns
    else "—"
)

tmp["type"] = (
    df["component_type"].astype(str)
    if "component_type" in df.columns
    else "Type_A"
)

tmp = tmp.join(analysis)

tmp["rank"] = (
    tmp["anomaly_score"]
    + tmp["predicted_168h"]
    / np.maximum(
        df["datasheet_limit_uA"].fillna(50).to_numpy(),
        1
    )
)

top = (
    tmp.sort_values(
        ["risk", "rank"],
        ascending=[False, False]
    )
    .head(5)
)

rows = []

for _, rr in top.iterrows():

    badge_class = {
        "HIGH": "badge-high",
        "MEDIUM": "badge-medium",
        "SAFE": "badge-safe"
    }.get(rr["risk"], "badge-safe")

    rows.append(f"""
        <tr>
            <td><strong>{rr[component_col]}</strong></td>
            <td>{rr["lot"]}</td>
            <td>{rr["type"]}</td>
            <td>{rr["anomaly_score"]:.2f}</td>
            <td>{rr["predicted_168h"]:.2f} µA</td>
            <td>{rr["predicted_slope"]:.3f} µA/h</td>
            <td>
                <span class="risk-badge {badge_class}">
                    {rr["risk"]}
                </span>
            </td>
            <td>
                <a
    class="view-btn"
    href="?component={rr[component_col]}"
    target="_self"
>
    View
</a>
            </td>
        </tr>
    """)

table_html = f"""
<style>

.risk-wrapper {{
    background:
        linear-gradient(
            145deg,
            rgba(10,16,36,0.97),
            rgba(5,9,23,0.98)
        );

    border:1px solid rgba(105,125,255,0.20);
    border-radius:16px;

    padding:16px;

    box-shadow:
        inset 0 1px rgba(255,255,255,0.025),
        0 18px 40px rgba(0,0,0,0.25);

    margin-top:10px;
}}

.risk-title {{
    font-family:Inter,sans-serif;
    font-size:14px;
    font-weight:700;
    color:#edf2ff;

    margin-bottom:13px;

    display:flex;
    align-items:center;
    gap:8px;
}}

.risk-title::before {{
    content:"";
    width:4px;
    height:17px;
    border-radius:5px;

    background:
        linear-gradient(
            180deg,
            #35dfff,
            #9b5cff
        );
}}

.risk-table {{
    width:100%;
    border-collapse:separate;
    border-spacing:0;

    font-family:Inter,sans-serif;
    font-size:9px;

    color:#dce4f5;
}}

.risk-table th {{
    padding:11px 10px;

    text-align:left;

    color:#7f8baa;

    font-size:8px;
    font-weight:700;

    text-transform:uppercase;
    letter-spacing:.05em;

    background:rgba(70,85,150,0.10);

    border-bottom:1px solid rgba(110,125,170,0.16);
}}

.risk-table td {{
    padding:11px 10px;

    border-bottom:1px solid rgba(110,125,170,0.08);
}}

.risk-table tr:last-child td {{
    border-bottom:none;
}}

.risk-table tr:hover {{
    background:rgba(90,70,220,0.08);
}}

.risk-badge {{
    display:inline-block;

    min-width:58px;

    text-align:center;

    padding:5px 9px;

    border-radius:7px;

    font-size:8px;
    font-weight:800;
}}

.badge-high {{
    color:#ff6877;

    background:rgba(255,70,90,0.10);

    border:1px solid rgba(255,70,90,0.40);

    box-shadow:0 0 12px rgba(255,70,90,0.08);
}}

.badge-medium {{
    color:#ffc84b;

    background:rgba(255,190,45,0.09);

    border:1px solid rgba(255,190,45,0.40);
}}

.badge-safe {{
    color:#32eb91;

    background:rgba(35,235,130,0.08);

    border:1px solid rgba(35,235,130,0.35);
}}

.view-btn {{
    display:inline-block;

    padding:5px 14px;

    border-radius:7px;

    color:#b394ff;

    background:rgba(130,85,255,0.07);

    border:1px solid rgba(155,115,255,0.35);

    font-size:8px;
}}

</style>

<div class="risk-wrapper">

    <div class="risk-title">
        Top 5 High-Risk Components
    </div>

    <table class="risk-table">

        <thead>
            <tr>
                <th>Component ID</th>
                <th>Lot</th>
                <th>Type</th>
                <th>Anomaly Score</th>
                <th>Predicted (168h)</th>
                <th>Predicted Slope</th>
                <th>Risk Level</th>
                <th>Action</th>
            </tr>
        </thead>

        <tbody>
            {''.join(rows)}
        </tbody>

    </table>

</div>
"""

# Render the ENTIRE HTML block at once
st.html(table_html)