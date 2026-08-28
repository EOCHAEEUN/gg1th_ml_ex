# -*- coding: utf-8 -*-
"""
전기차 에너지 최적 경로 데모 — Streamlit 앱
노트북(ev_energy_analysis.ipynb)이 저장한 ev_energy_model.pkl 을 로드해
① 주행 조건이 전비에 미치는 영향 ② 경로 A/B 에너지 비교를 보여준다.

설계 원칙
- 모델 artifact가 단일 진실 공급원: 피처·파생변수 정의·입력 범위를 앱에 하드코딩하지 않는다.
- 거리는 전비 예측 입력이 아니라 총에너지 변환 계수로만 사용한다.
  (모델에 거리를 넣으면 energy = rate(d) * d/100 에서 d가 제곱으로 들어가,
   동일 경로도 링크를 몇 개로 쪼개느냐에 따라 총에너지가 달라진다 — 경로탐색의
   링크 비용함수로 쓸 수 없다. 실측·트레이드오프 근거: ev_energy_analysis.ipynb 3-2/4-2.)
- 화면은 Streamlit 기본 컴포넌트로 구성해 라이트/다크 테마 모두에서 동일하게 보이게 한다.

실행: streamlit run app.py
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="EV 에너지 경로 추천", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

# 경로 추천에 집중한 저채도 단일 컬러 시스템.
st.markdown("""
<style>
:root {
    --brand:#567F86; --brand-dark:#3F6268; --brand-soft:#E8F0F1;
    --accent:#D6AA2B; --accent-dark:#71570D; --accent-soft:#FFF6D8;
    --neutral:#87969A; --bg:#F7F9F9; --surface:#FFFFFF; --border:#DCE3E4;
    --text:#26363C; --muted:#6E7D81;
    --warning:#B87535; --warning-soft:#F7E9DB;
    --danger:#A6525C; --danger-dark:#7C3D42; --danger-soft:#F6E8EA;
}
.stApp {background:var(--bg); color:var(--text);}
[data-testid="stHeader"] {background:rgba(247,249,249,.92);}
[data-testid="stSidebar"] > div:first-child {background:#F1F4F4;}
.block-container {padding-top:1.7rem; padding-bottom:3rem; max-width:1120px;}
h1 {font-size:1.9rem !important; margin-bottom:.15rem; letter-spacing:-.025em;}
h2 {font-size:1.24rem !important; margin:1.3rem 0 .35rem; letter-spacing:-.015em;}
h3 {font-size:1.04rem !important; margin:0 0 .35rem;}
[data-testid="stMetricValue"] {font-size:1.8rem; line-height:1.15; color:var(--text); font-variant-numeric:tabular-nums;}
[data-testid="stMetricLabel"] p {font-size:.85rem; color:var(--muted);}
[data-testid="stVerticalBlock"] {gap:.78rem;}
[data-testid="stHorizontalBlock"] {gap:.85rem;}
[data-testid="stVerticalBlockBorderWrapper"] {background:var(--surface); border-color:var(--border); border-radius:10px; box-shadow:none;}
[data-testid="stExpander"] {background:var(--surface); border-color:var(--border); border-radius:8px;}
[data-testid="stExpander"] summary p {font-size:.92rem;}
[data-testid="stCaptionContainer"] p {font-size:.8rem; margin-bottom:.1rem; color:var(--muted);}
hr {margin:.8rem 0 !important; border-color:var(--border);}
.scenario-line {color:var(--muted); font-size:.88rem; margin:.15rem 0 .85rem;}
.scenario-line strong {color:var(--text); font-weight:700;}
.section-lead {color:var(--muted); font-size:.89rem; margin-top:-.15rem; margin-bottom:.4rem;}
.decision {
    padding:.15rem 0 .15rem 1rem; border-left:3px solid var(--border); margin:1.1rem 0 1.3rem;
}
.decision:not(.warning):not(.danger) {border-left-color:var(--accent);}
.decision.warning {border-left-color:var(--warning);}
.decision.danger {border-left-color:var(--danger);}
.decision-title {font-size:1.14rem; font-weight:800; margin:0 0 .25rem; display:flex; align-items:center; gap:.42rem;}
.decision:not(.warning):not(.danger) .decision-title {color:var(--accent-dark);}
.decision.warning .decision-title {color:var(--warning);}
.decision.danger .decision-title {color:var(--danger);}
.decision-icon {font-size:1rem; line-height:1;}
.decision-detail {font-size:.88rem; color:var(--muted); margin:0;}
.decision-detail strong {color:var(--text); font-weight:750;}
.compare-wrap {background:var(--surface); border:1px solid var(--border); border-radius:9px; overflow:hidden; margin-bottom:.6rem;}
.compare-table {width:100%; border-collapse:collapse; table-layout:fixed; font-variant-numeric:tabular-nums;}
.compare-table th,.compare-table td {padding:.78rem 1rem; border-bottom:1px solid #E8EDEE; text-align:right;}
.compare-table tr:last-child td {border-bottom:0;}
.compare-table th {background:#F4F6F6; color:var(--muted); font-size:.78rem; font-weight:700;}
.compare-table th:first-child,.compare-table td:first-child {text-align:left; width:28%;}
.compare-table td:first-child {color:var(--muted); font-size:.84rem; font-weight:650;}
.compare-table td {color:var(--text); font-size:.94rem;}
.compare-table th.recommended {color:var(--accent-dark); border-bottom:2px solid var(--accent);}
.compare-table .best {color:var(--brand-dark); font-weight:850;}
.compare-table .danger-cell {color:var(--danger); font-weight:800;}
.rec-tag {font-weight:700; color:var(--accent-dark); font-size:.76rem; margin-left:.3rem;}
.table-note {font-size:.8rem; color:var(--muted); margin:.55rem 0 1.3rem;}
.tradeoff {padding:.15rem 0 .15rem 1rem; border-left:3px solid var(--neutral); color:var(--muted); font-size:.88rem; margin-top:.2rem;}
.tradeoff strong {color:var(--text); font-weight:750;}
.detail-summary {padding:.75rem .85rem; background:#F3F6F6; color:#586B70; border-radius:7px; font-size:.86rem;}
.flow-box {background:var(--surface); border:1px solid var(--border); border-radius:9px; padding:.2rem 1.05rem;}
.flow-row {display:flex; justify-content:space-between; align-items:baseline; gap:.75rem; padding:.65rem 0;}
.flow-row + .flow-row {border-top:1px solid #EEF2F2;}
.flow-label {font-size:.82rem; color:var(--muted); font-weight:650; flex:0 0 128px;}
.flow-value {font-size:1.05rem; font-weight:800; color:var(--text); font-variant-numeric:tabular-nums; flex:1;}
.flow-delta {font-size:.84rem; font-weight:750; text-align:right; white-space:nowrap;}
.flow-delta.bad {color:var(--danger);}
.flow-delta.good {color:var(--brand-dark);}
@media (max-width:700px) {
    .flow-row {flex-wrap:wrap;}
    .flow-delta {width:100%; text-align:left;}
    .compare-table th,.compare-table td {padding:.65rem .55rem;}
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path(__file__).resolve().parent / "ev_energy_model.pkl"
EXPECTED_ARTIFACT_VERSION = "2.0.0"
SAFE_SOC = 20.0

# 조건 비교의 기준점 — "표준 조건"
STANDARD = {
    "payload_kg": 100.0, "ambient_temp_C": 20.0, "hvac_power_kw": 0.5,
    "battery_temp_C": 25.0, "driving_style_index": 0.4, "tire_pressure_bar": 2.4,
}
LABELS = {
    "payload_kg": "적재중량", "ambient_temp_C": "외기온도", "hvac_power_kw": "공조 전력",
    "battery_temp_C": "배터리 온도", "driving_style_index": "운전 성향",
    "tire_pressure_bar": "타이어 공기압",
}
UNITS = {
    "payload_kg": "kg", "ambient_temp_C": "°C", "hvac_power_kw": "kW",
    "battery_temp_C": "°C", "driving_style_index": "", "tire_pressure_bar": "bar",
}
PRESETS = {
    "표준": STANDARD,
    "겨울철": {"payload_kg": 300.0, "ambient_temp_C": -5.0, "hvac_power_kw": 3.5,
             "battery_temp_C": 20.0, "driving_style_index": 0.5, "tire_pressure_bar": 2.4},
    "한파": {"payload_kg": 300.0, "ambient_temp_C": -10.0, "hvac_power_kw": 4.5,
            "battery_temp_C": 15.0, "driving_style_index": 0.6, "tire_pressure_bar": 2.2},
    "폭염": {"payload_kg": 100.0, "ambient_temp_C": 38.0, "hvac_power_kw": 4.0,
            "battery_temp_C": 35.0, "driving_style_index": 0.4, "tire_pressure_bar": 2.4},
    "고적재": {"payload_kg": 500.0, "ambient_temp_C": 20.0, "hvac_power_kw": 0.5,
             "battery_temp_C": 25.0, "driving_style_index": 0.5, "tire_pressure_bar": 2.4},
}


# ── 모델 로드 및 정합성 검증 ────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    art = joblib.load(MODEL_PATH)
    required = {"artifact_version", "pipeline", "features", "raw_features",
                "derived_features", "distance_col", "feature_bounds", "metrics"}
    missing = required - set(art)
    if missing:
        raise ValueError(f"모델 메타데이터 누락: {sorted(missing)}")
    if art["artifact_version"] != EXPECTED_ARTIFACT_VERSION:
        raise ValueError(f"모델 버전 불일치 — 앱={EXPECTED_ARTIFACT_VERSION}, "
                         f"모델={art['artifact_version']}. 노트북을 다시 실행하세요.")
    pipe_feats = list(getattr(art["pipeline"], "feature_names_in_", art["features"]))
    if pipe_feats != list(art["features"]):
        raise ValueError("모델 내부 피처와 artifact 메타데이터가 일치하지 않습니다.")
    return art


try:
    ART = load_artifact()
except FileNotFoundError:
    st.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH.name}\n\n"
             f"노트북을 실행해 모델을 저장한 뒤 app.py와 같은 폴더에 두세요.")
    st.stop()
except Exception as exc:
    st.error(f"모델 로드 실패: {exc}")
    st.stop()

PIPE, FEATS = ART["pipeline"], list(ART["features"])
RAW, DERIVED = list(ART["raw_features"]), ART["derived_features"]
DIST, BOUNDS, METRICS = ART["distance_col"], ART["feature_bounds"], ART["metrics"]
DERIVE_FN = {"(x - 20) ** 2": lambda x: (x - 20) ** 2,
             "abs(x - 20)": lambda x: (x - 20).abs()}


def predict_rate(rows: pd.DataFrame) -> np.ndarray:
    """조건 → 에너지 소비율(kWh/100km). 파생변수는 artifact 정의대로 생성."""
    rows = rows.copy()
    for name, spec in DERIVED.items():
        if name not in rows.columns:
            fn = DERIVE_FN.get(spec["formula"])
            if fn is None:
                raise ValueError(f"알 수 없는 파생변수: {name} = {spec['formula']}")
            rows[name] = fn(rows[spec["source"]])
    missing = [f for f in FEATS if f not in rows.columns]
    if missing:
        raise ValueError(f"모델 입력 누락: {missing}")
    return PIPE.predict(rows[FEATS])


def bound(name, fallback=(0.0, 1.0)):
    lo, hi = BOUNDS.get(name, fallback)
    return float(lo), float(hi)


def fmt(key, value):
    unit = UNITS[key]
    text = f"{value:.2f}" if key == "driving_style_index" else f"{value:g}"
    return f"{text}{(' ' + unit) if unit else ''}"


# ── 사이드바 ────────────────────────────────────────────────────────────
UI = {k: f"ui_{k}" for k in STANDARD}
for k, v in PRESETS["겨울철"].items():
    st.session_state.setdefault(UI[k], v)
st.session_state.setdefault("ui_soc", 80)
st.session_state.setdefault("ui_batt", 60)

with st.sidebar:
    st.subheader("주행 환경")
    preset = st.selectbox("환경 프리셋", list(PRESETS), index=1)
    if st.button("프리셋 적용", width="stretch"):
        for k, v in PRESETS[preset].items():
            st.session_state[UI[k]] = v
        st.rerun()

    st.divider()
    st.markdown("**배터리**")
    soc_now = st.slider("현재 SOC (%)", 10, 100, step=1, key="ui_soc")
    batt_kwh = st.number_input("배터리 용량 (kWh)", 40, 120, step=5, key="ui_batt")

    st.divider()
    st.markdown("**차량·환경 조건**")
    lo, hi = bound("ambient_temp_C")
    st.slider("외기온도 (°C)", lo, hi, step=1.0, key=UI["ambient_temp_C"])
    lo, hi = bound("payload_kg")
    st.slider("적재중량 (kg)", lo, hi, step=10.0, key=UI["payload_kg"])
    lo, hi = bound("hvac_power_kw")
    st.slider("공조 소비전력 (kW)", lo, hi, step=0.1, key=UI["hvac_power_kw"])
    lo, hi = bound("driving_style_index")
    st.slider("운전 성향 (0=온화, 1=급가감속)", lo, hi, step=0.05,
              key=UI["driving_style_index"])
    with st.expander("상세 조건"):
        lo, hi = bound("battery_temp_C")
        st.slider("배터리 온도 (°C)", lo, hi, step=1.0, key=UI["battery_temp_C"])
        lo, hi = bound("tire_pressure_bar")
        st.slider("타이어 공기압 (bar)", lo, hi, step=0.05, key=UI["tire_pressure_bar"])

    if st.button("표준 조건으로 초기화", width="stretch"):
        for k, v in STANDARD.items():
            st.session_state[UI[k]] = v
        st.rerun()

VEHICLE = {k: st.session_state[UI[k]] for k in STANDARD}


# ── 조건 기여도 분해 ────────────────────────────────────────────────────
# 선형 모델이므로 조건을 하나씩 표준값에서 현재값으로 바꿔 얻은 차이의 합이
# 전체 차이와 정확히 일치한다(가법성). 링크 조건과도 무관하다.
PROBE_LINK = {"speed_kmh": 80.0, "road_grade_pct": 0.0}


@st.cache_data(show_spinner=False)
def decompose(vehicle: dict) -> tuple[pd.DataFrame, float, float]:
    base_row = pd.DataFrame([{**STANDARD, **PROBE_LINK}])
    base_rate = float(predict_rate(base_row)[0])

    records = []
    for key, cur in vehicle.items():
        swapped = {**STANDARD, key: cur}
        rate = float(predict_rate(pd.DataFrame([{**swapped, **PROBE_LINK}]))[0])
        records.append({
            "조건": LABELS[key],
            "표준": fmt(key, STANDARD[key]),
            "현재": fmt(key, cur),
            "전비 영향": rate - base_rate,
        })
    cur_rate = float(predict_rate(pd.DataFrame([{**vehicle, **PROBE_LINK}]))[0])
    return pd.DataFrame(records), base_rate, cur_rate


contrib, base_rate, cur_rate = decompose(VEHICLE)
total_delta = cur_rate - base_rate


# ── 경로 계산 ───────────────────────────────────────────────────────────
DEFAULT_A = pd.DataFrame({"speed_kmh": [115.0, 120.0], "road_grade_pct": [1.0, 0.0],
                          DIST: [50.0, 50.0]})
DEFAULT_B = pd.DataFrame({"speed_kmh": [60.0, 65.0], "road_grade_pct": [1.0, 0.0],
                          DIST: [53.0, 53.0]})
LINK_KEYS = [c for c in RAW if c not in VEHICLE]


def validate(segs: pd.DataFrame) -> pd.DataFrame:
    need = LINK_KEYS + [DIST]
    missing = set(need) - set(segs.columns)
    if missing:
        raise ValueError(f"필수 열 누락: {sorted(missing)}")
    clean = segs[need].apply(pd.to_numeric, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("각 경로에 최소 1개의 유효한 링크가 필요합니다.")
    for col in LINK_KEYS:
        lo, hi = bound(col)
        if not clean[col].between(lo, hi).all():
            raise ValueError(f"{LABELS.get(col, col)}은 모델 학습 범위 {lo:g}~{hi:g} 안이어야 합니다.")
    if not clean[DIST].gt(0).all():
        raise ValueError("링크 거리는 0보다 커야 합니다.")
    return clean


def summarize(segs: pd.DataFrame, vehicle: dict) -> dict:
    clean = validate(segs)
    rows = clean.assign(**vehicle)
    rates = predict_rate(rows)                       # 거리는 입력 아님 — 이유는 모듈 docstring 참고
    d = rows[DIST].to_numpy()
    energy = rates * d / 100                         # 거리는 여기, 변환 계수로만 사용
    total, dist = float(energy.sum()), float(d.sum())
    minutes = float((d / rows["speed_kmh"].to_numpy()).sum() * 60)
    return {"dist": dist, "total": total, "rate": total / dist * 100,
            "efficiency": dist / total,
            "soc": float(soc_now - total / batt_kwh * 100),
            "minutes": minutes, "avg_speed": dist / (minutes / 60),
            "seg": energy}


def hours(minutes: float) -> str:
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}시간 {m}분" if h else f"{m}분"


# ── 본문 ────────────────────────────────────────────────────────────────
st.title("EV 에너지 경로 추천")
st.markdown(
    f'<div class="scenario-line"><strong>현재 조건</strong> · SOC {soc_now}% · {batt_kwh}kWh · '
    f'{VEHICLE["ambient_temp_C"]:g}°C · 적재 {VEHICLE["payload_kg"]:g}kg · '
    f'HVAC {VEHICLE["hvac_power_kw"]:g}kW</div>',
    unsafe_allow_html=True,
)

# 조건 설명용 가상 링크 결과는 메인 의사결정과 분리해 상세 영역에서만 사용한다.
r_std = batt_kwh / base_rate * 100
r_cur = batt_kwh / cur_rate * 100
changed = contrib[contrib["표준"] != contrib["현재"]].sort_values(
    "전비 영향", key=abs, ascending=False)

# 1) 경로 편집과 계산
with st.expander("경로 조건 수정 (속도·경사·거리)", expanded=False):
    sp_lo, sp_hi = bound("speed_kmh")
    gr_lo, gr_hi = bound("road_grade_pct")
    di_lo, di_hi = bound(DIST)
    cfg = {
        "speed_kmh": st.column_config.NumberColumn("평균속도 (km/h)", min_value=sp_lo,
                                                   max_value=sp_hi, step=5.0, required=True),
        "road_grade_pct": st.column_config.NumberColumn("도로 경사 (%)", min_value=gr_lo,
                                                        max_value=gr_hi, step=0.5, required=True),
        DIST: st.column_config.NumberColumn("링크 거리 (km)", min_value=di_lo,
                                            max_value=di_hi, step=1.0, required=True),
    }
    ea, eb = st.columns(2)
    with ea:
        st.markdown("**경로 A · 고속도로**")
        segs_a = st.data_editor(DEFAULT_A, num_rows="dynamic", column_config=cfg,
                                hide_index=True, width="stretch", key="route_a")
    with eb:
        st.markdown("**경로 B · 국도**")
        segs_b = st.data_editor(DEFAULT_B, num_rows="dynamic", column_config=cfg,
                                hide_index=True, width="stretch", key="route_b")
    st.caption("거리는 전비 예측 입력이 아니므로 링크를 몇 개로 나누든 총 에너지는 같습니다.")

try:
    A, B = summarize(segs_a, VEHICLE), summarize(segs_b, VEHICLE)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

ok_a, ok_b = A["soc"] >= 0, B["soc"] >= 0
low = "A" if A["total"] <= B["total"] else "B"

if not ok_a and not ok_b:
    need = abs((A if low == "A" else B)["soc"]) / 100 * batt_kwh
    pick = None
    decision_class = "danger"
    decision_title = "현재 배터리로 두 경로 모두 도착할 수 없습니다"
    decision_detail = f"경로 {low} 기준 최소 <strong>{need:.1f}kWh</strong>를 추가 충전해야 합니다. 충전 후 저소비 경로는 <strong>{low}</strong>입니다."
elif ok_a != ok_b:
    pick = "A" if ok_a else "B"
    blocked = "B" if ok_a else "A"
    decision_class = "warning"
    decision_title = f"경로 {pick}만 현재 배터리로 도착할 수 있습니다"
    decision_detail = f"경로 <strong>{blocked}</strong>는 주행 중 배터리가 소진되므로 충전 경유가 필요합니다."
else:
    pick = low
    win = A if pick == "A" else B
    save, gap = abs(A["total"] - B["total"]), abs(A["soc"] - B["soc"])
    if win["soc"] < SAFE_SOC:
        decision_class = "warning"
        decision_title = f"경로 {pick}가 에너지는 적게 쓰지만 잔량이 부족합니다"
        decision_detail = f"<strong>{save:.1f}kWh</strong> 절약 · 도착 SOC <strong>+{gap:.1f}%p</strong> · 예상 잔량 {win['soc']:.1f}%. 충전 계획을 함께 검토하세요."
    else:
        decision_class = ""
        decision_title = f"경로 {pick}를 추천합니다"
        decision_detail = f"<strong>{save:.1f}kWh</strong> 절약 · 도착 SOC <strong>+{gap:.1f}%p</strong> · 상대 경로 대비 에너지 효율이 높습니다."

DECISION_ICON = {"": "✓", "warning": "⚠", "danger": "✕"}
st.markdown(
    f'<div class="decision {decision_class}">'
    f'<div class="decision-title"><span class="decision-icon">{DECISION_ICON[decision_class]}</span>{decision_title}</div>'
    f'<div class="decision-detail">{decision_detail}</div></div>',
    unsafe_allow_html=True,
)


def cell_class(best=False, danger=False):
    if danger:
        return "danger-cell"
    return "best" if best else ""


status_a = f"가능 · {soc_now}→{A['soc']:.1f}%" if ok_a else f"불가 · {abs(A['soc']):.1f}%p 부족"
status_b = f"가능 · {soc_now}→{B['soc']:.1f}%" if ok_b else f"불가 · {abs(B['soc']):.1f}%p 부족"
if ok_a and ok_b:
    status_best_a, status_best_b = A["soc"] > B["soc"], B["soc"] > A["soc"]
else:
    status_best_a, status_best_b = ok_a, ok_b

head_a = "recommended" if pick == "A" else ""
head_b = "recommended" if pick == "B" else ""
tag_a = '<span class="rec-tag">✓ 추천</span>' if pick == "A" else ""
tag_b = '<span class="rec-tag">✓ 추천</span>' if pick == "B" else ""
st.markdown(
    f"""
    <div class="compare-wrap">
      <table class="compare-table">
        <thead><tr><th>비교 항목</th><th class="{head_a}">경로 A · 고속도로{tag_a}</th><th class="{head_b}">경로 B · 국도{tag_b}</th></tr></thead>
        <tbody>
          <tr><td>도착 가능 (SOC)</td><td class="{cell_class(status_best_a, not ok_a)}">{status_a}</td><td class="{cell_class(status_best_b, not ok_b)}">{status_b}</td></tr>
          <tr><td>소비전력</td><td class="{cell_class(A['total'] < B['total'])}">{A['total']:.1f} kWh</td><td class="{cell_class(B['total'] < A['total'])}">{B['total']:.1f} kWh</td></tr>
          <tr><td>전비</td><td class="{cell_class(A['efficiency'] > B['efficiency'])}">{A['efficiency']:.2f} km/kWh</td><td class="{cell_class(B['efficiency'] > A['efficiency'])}">{B['efficiency']:.2f} km/kWh</td></tr>
          <tr><td>거리</td><td class="{cell_class(A['dist'] < B['dist'])}">{A['dist']:.0f} km</td><td class="{cell_class(B['dist'] < A['dist'])}">{B['dist']:.0f} km</td></tr>
          <tr><td>예상 시간</td><td class="{cell_class(A['minutes'] < B['minutes'])}">{hours(A['minutes'])}</td><td class="{cell_class(B['minutes'] < A['minutes'])}">{hours(B['minutes'])}</td></tr>
        </tbody>
      </table>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<div class="table-note">권장 안전 잔량 {SAFE_SOC:.0f}% 기준입니다.</div>', unsafe_allow_html=True)

win, oth = (A, B) if low == "A" else (B, A)
st.markdown(
    f'<div class="tradeoff"><strong>비교 요약</strong> · 경로 {low}는 에너지를 '
    f'<strong>{abs(A["total"] - B["total"]):.1f}kWh</strong> 적게 사용합니다. '
    f'거리는 <strong>{abs(win["dist"] - oth["dist"]):.0f}km</strong> '
    f'{"길고" if win["dist"] > oth["dist"] else "짧고"} 예상 시간은 '
    f'<strong>{abs(win["minutes"] - oth["minutes"]):.0f}분</strong> '
    f'{"더 걸립니다" if win["minutes"] > oth["minutes"] else "짧습니다"}.</div>',
    unsafe_allow_html=True,
)


# 2) 주행 환경 영향은 의사결정 이후의 상세 설명으로 제공한다.
#    읽는 순서: 비교 전제 → 전체 결과 → 원인 순위 → 계산 상세.

with st.expander("현재 조건에서 에너지 소비가 왜 늘었나요?"):
    st.markdown(
        f'<div class="tradeoff"><strong>분석 조건</strong> · 80km/h · 평지 · {batt_kwh:g}kWh 배터리 가정'
        f'<br>실제 경로 결과가 아닌 차량·환경 조건 비교용 분석입니다.</div>',
        unsafe_allow_html=True,
    )

    if changed.empty:
        st.info("왼쪽에서 조건을 변경하면 요인별 영향이 표시됩니다.")
    else:
        pct = total_delta / base_rate * 100
        range_diff = r_cur - r_std
        rate_word = "증가" if total_delta > 0 else "감소"
        range_word = "감소" if range_diff < 0 else "증가"
        st.markdown(
            f"""
            <div class="flow-box">
              <div class="flow-row">
                <div class="flow-label">에너지 소비율</div>
                <div class="flow-value">{base_rate:.1f} → {cur_rate:.1f} kWh/100km</div>
                <div class="flow-delta {'bad' if total_delta > 0 else 'good'}">{total_delta:+.2f}kWh · {abs(pct):.0f}% {rate_word}</div>
              </div>
              <div class="flow-row">
                <div class="flow-label">가정상 주행가능거리</div>
                <div class="flow-value">{r_std:.0f} → {r_cur:.0f} km</div>
                <div class="flow-delta {'bad' if range_diff < 0 else 'good'}">{abs(range_diff):.0f}km {range_word}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f'<div class="section-lead" style="margin-top:1rem;">'
                    f'소비가 {rate_word}한 주요 원인</div>', unsafe_allow_html=True)

        has_share = abs(total_delta) > 1e-9
        chart_df = changed.copy()
        chart_df["표시"] = chart_df["조건"] + "  " + chart_df["표준"] + " → " + chart_df["현재"]
        chart_df["소비율 변화"] = chart_df["전비 영향"]
        chart_df["비중"] = chart_df["전비 영향"] / total_delta * 100 if has_share else 0.0
        order = chart_df["표시"].tolist()

        top_idx = chart_df["소비율 변화"].abs().idxmax()
        chart_df["색상"] = [
            ("#7C3D42" if v > 0 else "#3F6268") if i == top_idx else
            ("#A6525C" if v > 0 else "#567F86")
            for i, v in chart_df["소비율 변화"].items()
        ]
        chart_df["강조"] = (chart_df.index == top_idx).astype(float)
        chart_df["레이블"] = (
            chart_df["소비율 변화"].map(lambda v: f"{v:+.2f}")
            + (chart_df["비중"].map(lambda v: f" · {v:.0f}%") if has_share else "")
        )

        has_pos = (chart_df["소비율 변화"] > 0).any()
        has_neg = (chart_df["소비율 변화"] < 0).any()
        if has_pos and has_neg:
            extent = max(float(chart_df["소비율 변화"].abs().max()) * 1.28, .5)
            x_domain = [-extent, extent]
        elif has_pos:
            x_domain = [0, float(chart_df["소비율 변화"].max()) * 1.18]
        else:
            x_domain = [float(chart_df["소비율 변화"].min()) * 1.18, 0]

        bars = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusEnd=4, size=21)
            .encode(
                x=alt.X("소비율 변화:Q", title="에너지 소비율 변화 (kWh/100km)",
                        scale=alt.Scale(domain=x_domain)),
                y=alt.Y("표시:N", title=None, sort=order),
                color=alt.Color("색상:N", scale=None, legend=None),
                opacity=alt.Opacity("강조:Q", scale=alt.Scale(domain=[0, 1], range=[0.55, 1]), legend=None),
                tooltip=["조건:N", "표준:N", "현재:N",
                         alt.Tooltip("소비율 변화:Q", format="+.2f"),
                         alt.Tooltip("비중:Q", format=".0f", title="비중(%)")],
            )
        )
        pos_labels = bars.transform_filter(alt.datum["소비율 변화"] >= 0).mark_text(
            align="left", baseline="middle", dx=6, color="#5D6E74"
        ).encode(text="레이블:N")
        neg_labels = bars.transform_filter(alt.datum["소비율 변화"] < 0).mark_text(
            align="right", baseline="middle", dx=-6, color="#5D6E74"
        ).encode(text="레이블:N")
        zero_rule = alt.Chart(pd.DataFrame({"기준": [0]})).mark_rule(
            color="#AAB5B8", strokeWidth=1
        ).encode(x="기준:Q")
        st.altair_chart(
            (bars + pos_labels + neg_labels + zero_rule)
            .properties(height=max(190, 40 * len(chart_df)))
            .configure_view(strokeWidth=0),
            width="stretch",
        )

        top_row = chart_df.loc[top_idx]
        if has_share:
            st.markdown(
                f'<div class="detail-summary"><strong>가장 큰 영향</strong> · '
                f'{top_row["조건"]}이 전체 소비 {rate_word}분의 약 {abs(top_row["비중"]):.0f}%를 차지합니다.</div>',
                unsafe_allow_html=True,
            )
            verb = "증가량" if total_delta > 0 else "감소량"
            st.caption(f"각 요인의 영향을 합하면 현재 환경의 전체 {verb}인 "
                       f"{abs(contrib['전비 영향'].sum()):.2f}kWh/100km가 됩니다.")
        else:
            st.markdown(
                f'<div class="detail-summary"><strong>가장 큰 영향</strong> · '
                f'{top_row["조건"]}의 영향이 가장 크지만, 증가·감소 요인이 서로 상쇄돼 '
                f'전체 소비율은 표준과 거의 같습니다.</div>',
                unsafe_allow_html=True,
            )

    with st.popover("표준 조건 전체 보기"):
        baseline_df = pd.DataFrame({
            "조건": [LABELS[k] for k in STANDARD],
            "표준값": [fmt(k, v) for k, v in STANDARD.items()],
        })
        st.dataframe(baseline_df, hide_index=True, width="stretch")


# 4) 링크 단위 분석
with st.expander("링크 단위 조건 분석"):
    st.caption("한 링크의 속도·경사·거리를 바꿔가며 전비와 배터리 변화를 확인합니다.")
    k1, k2, k3 = st.columns(3)
    lo, hi = bound("speed_kmh")
    ls = k1.slider("링크 평균속도 (km/h)", lo, hi, 90.0, 5.0, key="lk_speed")
    lo, hi = bound("road_grade_pct")
    lg = k2.slider("도로 경사 (%)", lo, hi, 1.0, 0.5, key="lk_grade")
    lo, hi = bound(DIST)
    ld = k3.slider("링크 거리 (km)", lo, hi, 50.0, 5.0, key="lk_dist")

    rate = float(predict_rate(pd.DataFrame([{**VEHICLE, "speed_kmh": ls,
                                             "road_grade_pct": lg}]))[0])
    flat = float(predict_rate(pd.DataFrame([{**VEHICLE, "speed_kmh": ls,
                                             "road_grade_pct": 0.0}]))[0])
    kwh = rate * ld / 100
    after = soc_now - kwh / batt_kwh * 100

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("에너지 소비율", f"{rate:.1f} kWh/100km",
              delta=f"{rate - cur_rate:+.1f} vs 평지 80km/h", delta_color="inverse",
              icon=":material/electric_bolt:")
    n2.metric("경사 영향", f"{rate - flat:+.2f} kWh/100km", delta_color="off",
              icon=":material/landscape:")
    n3.metric("링크 소비량", f"{kwh:.1f} kWh", icon=":material/route:")
    n4.metric("통과 후 SOC", f"{after:.1f} %", delta=f"{after - soc_now:.1f} %p",
              icon=":material/battery_5_bar:")

    if after < 0:
        st.error("이 링크를 통과하기 전에 배터리가 소진됩니다. 충전 경유를 검토하세요.")
    elif after < SAFE_SOC:
        st.warning(f"통과 후 잔량이 {after:.1f}%로 낮습니다.")

# 5) 모델 정보
with st.expander("모델 정보 및 예측 오차"):
    st.markdown(f"""
- 모델 `{ART.get('model_name', 'linear')}` · artifact v{ART['artifact_version']} · 입력 {len(FEATS)}개
- 교차검증 R² **{METRICS['cv_r2']:.4f}** · MAE **{METRICS['cv_mae']:.3f} kWh/100km**
- 60kWh 기준 주행가능거리 오차: 평균 **{METRICS['range_mae_km']:.1f} km**,
  95% 사례 **{METRICS['range_p95_km']:.1f} km** 이내
- 거리는 모델 입력이 아니라 `예측 소비율 × 거리 ÷ 100` 변환에만 사용합니다.
""")
    st.caption("잔량이 빠듯할 때는 평균 오차가 아니라 95% 값을 안전 마진으로 두는 편이 안전합니다.")
    saved = ART.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        st.warning(f"모델은 scikit-learn {saved}에서 저장됐고 현재 환경은 {sklearn.__version__}입니다. "
                   f"노트북을 이 환경에서 한 번 실행하면 경고가 사라집니다.", icon="⚠️")
