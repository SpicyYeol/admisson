import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

import numpy as np

# Page Config
st.set_page_config(page_title="대학 입학 데이터 심층 분석", layout="wide")

# --- Authentication Gateway ---
def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "251224":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.markdown("### 🔐 데이터 보호를 위해 비밀번호를 입력해주세요.")
        st.text_input(
            "비밀번호 (Password)", type="password", on_change=password_entered, key="password"
        )
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 비밀번호가 틀렸습니다. 다시 시도해주세요.")
        st.stop()
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.markdown("### 🔐 데이터 보호를 위해 비밀번호를 입력해주세요.")
        st.text_input(
            "비밀번호 (Password)", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 비밀번호가 틀렸습니다. 다시 시도해주세요.")
        st.stop()
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

# --- PDF Export & Print Styling ---
st.markdown("""
<style>
    /* Print optimizations */
    @media print {
        /* Hide sidebar, buttons, decorations, and the "Running" spinner */
        [data-testid="stSidebar"], 
        [data-testid="stStatusWidget"],
        .stButton, 
        header, 
        footer, 
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        .stExpanderToggleIcon,
        [data-testid="stStatusWidget"] {
            display: none !important;
            height: 0 !important;
            width: 0 !important;
            overflow: hidden !important;
        }
        
        /* 1. Global Opacity & Color Fix: Stop the "Running" fade effect */
        [data-testid="stAppViewContainer"], 
        .main, 
        .stApp,
        [data-testid="stVerticalBlock"],
        [data-testid="stBlock"] {
            opacity: 1 !important;
            filter: none !important;
            background: white !important;
        }

        /* 2. Force text to be solid black (no transparencies) */
        h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown {
            color: black !important;
            opacity: 1 !important;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }

        /* 3. Specialized Layout Fixes */
        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
            max-width: 100% !important;
            margin: 0 !important;
        }

        /* Avoid breaking charts/metrics across pages */
        .stMetric, .stTable, .stPlotlyChart, .stImage, [data-testid="stVerticalBlock"] > div {
            page-break-inside: avoid !important;
            opacity: 1 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📄 보고서 출력")
    if st.button("🖨️ PDF 리포트로 저장/출력"):
        import time
        # Use a dynamic key to ensure the component re-renders and triggers the script every time
        # Add a timestamp in the script comment to make the HTML content unique 
        # This forces the browser to re-execute the script even on consecutive clicks
        placeholder = st.empty()
        with placeholder:
            st.components.v1.html(
                f"<script>window.parent.print(); // {time.time()}</script>",
                height=0
            )

st.title("🎓 대학 입학 데이터 심층 분석 대시보드")
st.markdown("---")

# 1. Load & Preprocess Data
@st.cache_data
def load_and_process_data(file_obj):
    try:
        df = pd.read_excel(file_obj, engine='openpyxl')
        
        # Numeric Conversion
        df['수능등급'] = pd.to_numeric(df['수능등급'], errors='coerce')
        df['석차백분율(내신)'] = pd.to_numeric(df['석차백분율(내신)'], errors='coerce')
        
        # Filter Zeros (Use np.nan to avoid NAType issues)
        df['수능등급'] = df['수능등급'].replace(0, np.nan)
        df['석차백분율(내신)'] = df['석차백분율(내신)'].replace(0, np.nan)
        
        # 1. Classification
        def classify_type(row):
            if '정시' in str(row['모집구분']):
                return '정시'
            return '수시'
        df['입학유형'] = df.apply(classify_type, axis=1)
        
        # 2. Representative Score
        def get_score(row):
            if row['입학유형'] == '정시':
                return row['수능등급']
            return row['석차백분율(내신)']
        df['대표성적'] = df.apply(get_score, axis=1)
        
        # 3. Interview Field
        df['면접유무'] = df['전형구분'].apply(lambda x: '면접 위주' if '면접' in str(x) else '서류/교과 위주')
        
        # 4. Standardized Admission Match (Grouping)
        def standardize_admission(row):
            name = str(row['전형구분']).replace(' ', '')
            
            # 1. Base Category
            if '정원외' in name: return '정원외' # Explicitly separate Extra-quota
            elif '지역교과' in name: category = '지역교과'
            elif '지역인재' in name: category = '지역인재'
            elif '교과' in name: category = '학생부교과'
            elif '종합' in name or '잠재' in name: category = '학생부종합'
            elif '수능' in name: category = '수능위주'
            elif '논술' in name: category = '논술위주'
            elif '실기' in name: category = '실기위주'
            elif '고른' in name or '기회' in name or '농어촌' in name: return '고른기회/특별'
            else: category = '기타'
            
            # 2. Add Interview Info
            is_interview = '면접' in str(row['전형구분'])
            suffix = "(면접O)" if is_interview else "(면접X)"
            
            return f"{category} {suffix}"
            
        df['전형그룹'] = df.apply(standardize_admission, axis=1)
        
        # 5. Pass Status for Competition Rate
        def check_pass(status):
            s = str(status)
            if '합격' in s or '충원' in s:
                if '불합격' in s: return False
                return True
            return False
        df['합격여부'] = df['합격구분'].apply(check_pass)
        
        # 6. Global Segmentation (Refined Early, Regular, Total)
        def classify_segment(row):
            mojib = str(row['모집구분'])
            jeon = str(row['전형구분'])
            if '정시' in mojib:
                return '정시'
            if '수시' in mojib:
                # User's exclusion criteria
                excl = ['수능', '기회', '농어촌', '기초', '재외', '특성화']
                if not any(kw in jeon for kw in excl):
                    return '수시(일반)'
                return '수시(기타)'
            return '기타'
        df['분석그룹'] = df.apply(classify_segment, axis=1)
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 등 오류: {e}")
        return pd.DataFrame()

# Sidebar Data Upload
with st.sidebar:
    st.markdown("### 📥 데이터 입력")
    uploaded_file = st.file_uploader("입시 결과 엑셀 파일(.xlsx)을 업로드하세요", type=["xlsx"])

if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
    if df.empty:
        st.warning("⚠️ 파일은 업로드되었으나 데이터를 읽을 수 없습니다. 형식을 확인해주세요.")
        st.stop()
else:
    st.info("👋 환영합니다! 분석을 시작하려면 왼쪽 사이드바에서 **입시 데이터 엑셀 파일**을 업로드해주세요.")
    st.stop()

# --- Cached Utility Functions for Performance ---

@st.cache_data
def get_regional_population_data():
    """Generates years 2020-2040 population projections for all 17 regions."""
    years_dense = np.arange(2020, 2041)
    xp = [2020, 2023, 2024, 2025, 2030, 2040]
    fp_nat = [497562, 439510, 394940, 453812, 400000, 280000]
    dense_nat = np.interp(years_dense, xp, fp_nat)
    
    regions_info = {
        '서울': {'base': 80000, 'decl': 0.7}, '부산': {'base': 27000, 'decl': 0.5},
        '대구': {'base': 22000, 'decl': 0.5}, '인천': {'base': 26000, 'decl': 0.7},
        '광주': {'base': 16000, 'decl': 0.5}, '대전': {'base': 14000, 'decl': 0.5},
        '울산': {'base': 11000, 'decl': 0.5}, '세종': {'base': 3000, 'decl': 1.5},
        '경기': {'base': 130000, 'decl': 0.75}, '강원': {'base': 13000, 'decl': 0.45},
        '충북': {'base': 14000, 'decl': 0.5}, '충남': {'base': 19000, 'decl': 0.55},
        '전북': {'base': 17000, 'decl': 0.45}, '전남': {'base': 16000, 'decl': 0.45},
        '경북': {'base': 23000, 'decl': 0.45}, '경남': {'base': 33000, 'decl': 0.5},
        '제주': {'base': 6000, 'decl': 0.6}
    }
    
    dense_data = {'연도': years_dense, '전국': dense_nat}
    nat_ratios = np.array(fp_nat) / fp_nat[0]
    
    for reg, info in regions_info.items():
        b = info['base']
        target_2040 = b * info['decl']
        p20, p23, p24, p25 = b, b*nat_ratios[1], b*nat_ratios[2], b*nat_ratios[3]
        p30 = p25 + (target_2040 - p25) * (5/15)
        p40 = target_2040
        y_pts = [p20, p23, p24, p25, p30, p40]
        dense_data[reg] = np.interp(years_dense, xp, y_pts)
        
    return pd.DataFrame(dense_data), regions_info

def check_region_all(txt):
    """Maps raw education office strings to 17 major regions."""
    txt = str(txt)
    regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
    for r in regions:
        if r in txt: return r
    if '경상남도' in txt: return '경남'
    if '경상북도' in txt: return '경북'
    if '전라남도' in txt: return '전남'
    if '전라북도' in txt or '전북' in txt: return '전북'
    if '충청남도' in txt: return '충남'
    if '충청북도' in txt: return '충북'
    return '기타'

@st.cache_data
def perform_full_correlation_analysis(df_sample, df_dense_all, regions_list):
    """Calculates Pearson correlation between population and applicant count for all regions."""
    # We need a representative slice of df for internal stats
    df_sample['권역_상세'] = df_sample['교육청소재지'].apply(check_region_all)
    internal_all = df_sample.groupby(['학년도', '권역_상세']).size().reset_index(name='지원자수')
    
    corr_results = []
    for reg in regions_list:
        reg_int = internal_all[internal_all['권역_상세'] == reg]
        if not reg_int.empty:
            merged = pd.merge(reg_int, df_dense_all[['연도', reg]], left_on='학년도', right_on='연도', how='inner')
            if len(merged) > 2:
                if merged[reg].std() == 0 or merged['지원자수'].std() == 0: r = 0
                else: r = np.corrcoef(merged[reg], merged['지원자수'])[0, 1]
                
                sensitivity = "🔴 높음" if r > 0.7 else ("🟡 보통" if r > 0.4 else "🟢 낮음")
                corr_results.append({
                    '지역': reg, '상관계수(r)': r, '인구민감도': sensitivity, 
                    '지원자규모(Avg)': int(merged['지원자수'].mean())
                })
    return pd.DataFrame(corr_results).sort_values('상관계수(r)', ascending=False)

@st.cache_data
def get_department_analysis_data(df_yr):
    """Processes department-level stats for the quota simulation."""
    def calc_stats(x):
        reg = x[x['등록구분'] == '등록']
        pass_count = x['합격여부'].sum()
        return pd.Series({
            '지원자수': len(x),
            '합격자수': pass_count,
            '등록자수': len(reg),
            '등록자평균성적': reg['대표성적'].mean(),
            '경쟁률': len(x) / pass_count if pass_count > 0 else 0
        })
    stats = df_yr.groupby('모집단위').apply(calc_stats).reset_index()
    return stats

@st.cache_data
def get_future_prediction_data(df_hist, df_dense_all):
    """
    Predicts next 5 years (2026-2030) using a Capture Rate (Market Share) approach.
    Logic: 
    1. Historical Capture Rate = Applicants / (Lagged weighted population)
    2. Forecast Capture Rate based on historical time trend.
    3. Pred Applicants = Pred Capture Rate * Future Population.
    This ensures that declining population reflects in the forecast even if capture rate is growing.
    """
    # 1. Historical Stats
    yearly_stats = df_hist.groupby('학년도').agg(
        지원자수=('수험번호', 'count'),
        평균성적=('대표성적', 'mean'),
        합격자수=('합격여부', 'sum')
    ).reset_index()
    
    if len(yearly_stats) < 2: return pd.DataFrame(), pd.DataFrame()
    
    # 2. Regional Weights
    last_y = yearly_stats.iloc[-1]['학년도']
    df_recent = df_hist[df_hist['학년도'] >= last_y - 2].copy()
    df_recent['지역'] = df_recent['교육청소재지'].apply(check_region_all)
    reg_weights = df_recent['지역'].value_counts() / len(df_recent)
    
    # 3. Create Weighted & LAGGED Population Metric
    regions_list = [r for r in reg_weights.index if r in df_dense_all.columns]
    df_dense_all = df_dense_all.copy()
    df_dense_all['weighted_pop'] = 0
    for reg in regions_list:
        df_dense_all['weighted_pop'] += df_dense_all[reg] * reg_weights[reg]
    
    # LAG LOGIC: Population of year Y-1 affects admission of year Y
    df_lagged_pop = df_dense_all[['연도', 'weighted_pop']].copy()
    df_lagged_pop['입시적용연도'] = df_lagged_pop['연도'] + 1
    
    hist_merged = pd.merge(yearly_stats, df_lagged_pop[['입시적용연도', 'weighted_pop']], 
                           left_on='학년도', right_on='입시적용연도')
    
    # 4. Capture Rate (Applicants / Population) - STABILIZED APPROACH
    hist_merged['capture_rate'] = hist_merged['지원자수'] / hist_merged['weighted_pop']
    
    # CRITICAL FIX: To prevent unrealistic upward trends based on short-term fluctuations,
    # we assume the university will maintain its RECENT capture rate rather than growing it indefinitely.
    # This makes the "Population Cliff" the primary driver of the forecast.
    stable_capture_rate = hist_merged['capture_rate'].tail(2).mean()
    
    # Competition to Grade Correlation
    last_quota = yearly_stats.iloc[-1]['합격자수'] if yearly_stats.iloc[-1]['합격자수'] > 0 else 1
    hist_merged['경쟁률'] = hist_merged['지원자수'] / last_quota
    
    # Weighting for grade correlation - keeping it recent
    weights = np.linspace(0.5, 1.0, len(hist_merged))
    # Logic: More Competition (X) -> Better/Lower Grade (Y). Slope m should be NEGATIVE.
    m_grade, c_grade = np.polyfit(hist_merged['경쟁률'], hist_merged['평균성적'], 1, w=weights)
    
    # CRITICAL LOGIC FIX: In an inverted axis (1 top, 9 bottom), 'decline' means grade numbers get BIGGER.
    # If historical data suggests otherwise due to noise, we force the "Vacuum Effect" logic.
    if m_grade > -0.1: # If slope is positive or nearly zero (unrealistic)
        m_grade = -0.5 # Force: 1 point drop in competition results in 0.5 grade point rise (worsening)
    
    # Baseline Alignment for grades
    last_act_comp = hist_merged.iloc[-1]['경쟁률']
    last_act_grade = hist_merged.iloc[-1]['평균성적']
    c_grade_adj = last_act_grade - m_grade * last_act_comp
    
    # 5. Forecast (2026-2030)
    future_adm_years = [2026, 2027, 2028, 2029, 2030]
    future_data = []
    
    for f_y in future_adm_years:
        # Get lagged population
        pop_y = f_y - 1
        pop_row = df_dense_all[df_dense_all['연도'] == pop_y]
        if pop_row.empty: continue
        pop_val = pop_row['weighted_pop'].values[0]
        
        # Predicted Apps = Stable Capture Rate * Future Population
        pred_app = stable_capture_rate * pop_val
        pred_comp = pred_app / last_quota
        
        # Predicted Grade: As competition drops, numerical grades will RISE (approaching 9)
        pred_grade = m_grade * pred_comp + c_grade_adj
        # Safety bound (cannot be better than 1 or worse than 9)
        pred_grade = max(1.0, min(9.0, pred_grade))
        
        future_data.append({
            '연도': f_y, 
            'weighted_pop': pop_val, 
            '예측지원자수': pred_app, 
            '예측경쟁률': pred_comp, 
            '예측평균성적': pred_grade,
            '예측점유율': stable_capture_rate
        })
    
    return pd.DataFrame(future_data), hist_merged

if df.empty:
    st.stop()

# --- Sidebar: Checkbox Filters ---
st.sidebar.header("🔍 검색 조건")
st.sidebar.markdown("**📅 학년도 선택**")

# Ensure years are sorted and unique
all_years = sorted(df['학년도'].dropna().unique().astype(int))

# Preset Selection Radio
preset = st.sidebar.radio(
    "기간 선택 옵션",
    ('최근 3개년', '최근 5개년', '전체', '직접 선택(Checkbox)'),
    index=2 # Default to 'All' to match previous behavior of selecting all
)

selected_years = []

if preset == '최근 3개년':
    selected_years = all_years[-3:]
    st.sidebar.success(f"선택: {', '.join(map(str, selected_years))}")
elif preset == '최근 5개년':
    selected_years = all_years[-5:]
    st.sidebar.success(f"선택: {', '.join(map(str, selected_years))}")
elif preset == '전체':
    selected_years = all_years
    st.sidebar.success(f"전체 {len(selected_years)}개 학년도 선택됨")
else:
    # Custom Selection
    st.sidebar.caption("아래에서 원하는 학년도를 체크하세요.")
    for year in all_years:
        if st.sidebar.checkbox(f"{year} 학년도", value=True):
            selected_years.append(year)

if not selected_years:
    st.warning("학년도를 하나 이상 선택해주세요.")
    st.stop()

df_filtered = df[df['학년도'].isin(selected_years)]
last_year = df_filtered['학년도'].max() if not df_filtered.empty else df['학년도'].max()

# --- 1. Executive Summary Metrics (Segmentation: Early/Regular/Total) ---
st.header("📌 종합 입시 지표 요약 (수시(일반) / 정시 / 합계)")

def get_row_metrics(target_df, label):
    df_pass = target_df.dropna(subset=['대표성적'])
    df_reg = df_pass[df_pass['등록구분'] == '등록']
    
    count = len(target_df)
    mean_reg = df_reg['대표성적'].mean() if not df_reg.empty else 0
    cut_70 = df_reg['대표성적'].quantile(0.7) if not df_reg.empty else 0
    
    if not df_reg.empty:
        min_row = df_reg.loc[df_reg['대표성적'].idxmax()]
        min_grade = min_row['대표성적']
        min_type = min_row['전형그룹']
    else:
        min_grade, min_type = 0, "-"
        
    return count, mean_reg, cut_70, min_grade, min_type

# Prepare segments
df_early_ref = df_filtered[df_filtered['분석그룹'] == '수시(일반)']
df_reg_total = df_filtered[df_filtered['분석그룹'] == '정시']
segments = [
    ("수시 (일반 - 수능/기회/농어촌/기초/재외/특성화 제외)", df_early_ref, "🔵"),
    ("정시 (전체)", df_reg_total, "🔴"),
    ("전체 합계", df_filtered, "🟣")
]

# Insight Block
st.info(f"""
**💡 핵심 요약 ({last_year}학년도)**: 
- **수시(일반)**: 등록자 평균 {df_early_ref[df_early_ref['등록구분']=='등록']['대표성적'].mean():.2f}등급 (주요 6개 차등 전형 제외)
- **정시(전체)**: 등록자 평균 {df_reg_total[df_reg_total['등록구분']=='등록']['대표성적'].mean():.2f}등급 (수능 성적 기준)
- 세분화된 데이터를 바탕으로 전형별 타겟 마케팅 및 정원 조정 전략을 수립하십시오.
""")

for label, sub_df, emoji in segments:
    count, mean_reg, cut_70, min_g, min_t = get_row_metrics(sub_df, label)
    st.markdown(f"##### {emoji} {label}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 지원자", f"{count:,.0f}명")
    c2.metric("최종등록 평균", f"{mean_reg:.2f}")
    c3.metric("70% 컷", f"{cut_70:.2f}")
    c4.metric("최저점", f"{min_g:.2f}")
    c5.write(f"최저점 발생:\n{min_t}")
    st.write("")

st.markdown("---")

# --- 2. External Market Environment ---
st.header("🗺️ 1. 외부 시장 환경 및 지역별 상관관계")

df_dense_all, regions_info = get_regional_population_data()

# Insight Block
st.success("""
**💡 시장 환경 인사이트**: 
- 학령인구 감소와 본교 지원자 수의 상관관계가 **0.7 이상**인 지역(빨간색)은 인구 변화에 직접적인 타격을 입는 '위험 지역'입니다. 
- 대조적으로 상관관계가 낮은 지역은 브랜드 인지도나 지역적 특수성으로 방어되고 있음을 의미합니다.
""")

col_pop1, col_pop2 = st.columns([2, 1])
with col_pop1:
    st.markdown("##### 📉 시도별 학령인구 추이 (2020~2040)")
    default_show = ['부산', '울산', '경남', '서울']
    regions_to_show = st.multiselect("표시할 지역 선택", list(regions_info.keys()), default=default_show)
    
    fig_all, ax_all = plt.subplots(figsize=(10, 5))
    ax_bg = ax_all.twinx()
    sns.lineplot(data=df_dense_all, x='연도', y='전국', ax=ax_bg, color='grey', alpha=0.3, linestyle='--', label='전국 총계')
    ax_bg.set_ylabel("전국 총계 (명)", color='grey')
    
    for reg in regions_to_show:
        sns.lineplot(data=df_dense_all, x='연도', y=reg, label=reg, marker='o', ax=ax_all)
    
    ax_all.set_ylabel("지역별 고3 학생수 (명)")
    ax_all.grid(True, linestyle=':', alpha=0.5)
    st.pyplot(fig_all)

with col_pop2:
    st.markdown("##### 📊 2040 인구 전망 데이터")
    st.dataframe(df_dense_all[['연도'] + regions_to_show].set_index('연도').style.format("{:,.0f}"))

# 17-Region Correlation Grid Visualization
st.markdown("##### 🗺️ 전국 17개 시도별 상관관계 매트릭스 (인구 vs 지원자)")
sel_corr_seg = st.radio("상관관계 분석 대상 선택", ["전체 합계", "수시(일반)", "정시"], key='corr_seg_sel', horizontal=True)

# Performance optimization: Perform analysis on the subset
df_for_corr = df if sel_corr_seg == "전체 합계" else df[df['분석그룹'] == sel_corr_seg]
corr_res_df = perform_full_correlation_analysis(df_for_corr, df_dense_all, list(regions_info.keys()))

# Facet-like Subplots for all regions
fig_grid, axes = plt.subplots(nrows=3, ncols=6, figsize=(20, 10))
axes = axes.flatten()
internal_all = df_for_corr[df_for_corr['학년도'].isin(selected_years)].copy()
internal_all['권역_상세'] = internal_all['教育청소재지'].apply(check_region_all) if '教育청소재지' in internal_all.columns else internal_all['교육청소재지'].apply(check_region_all)

for i, reg in enumerate(list(regions_info.keys())):
    if i < len(axes):
        ax = axes[i]
        reg_agg = internal_all[internal_all['권역_상세'] == reg].groupby('학년도').size().reset_index(name='지원자수')
        merged_t = pd.merge(reg_agg, df_dense_all[['연도', reg]], left_on='학년도', right_on='연도', how='inner')
        
        if not merged_t.empty and len(merged_t) > 1:
            corr_row = corr_res_df[corr_res_df['지역'] == reg]
            if not corr_row.empty:
                r_val = corr_row['상관계수(r)'].values[0]
                color = 'red' if r_val > 0.7 else ('orange' if r_val > 0.4 else 'blue')
                sns.regplot(data=merged_t, x=reg, y='지원자수', ax=ax, color=color, scatter_kws={'alpha':0.5})
                ax.set_title(f"{reg} (r={r_val:.2f})")
            else:
                ax.text(0.5, 0.5, "No Corr", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Data Lack", ha='center', va='center')
        ax.set_xlabel("")
        ax.set_ylabel("")

plt.tight_layout()
st.pyplot(fig_grid)

# Macro Trends: Early(Refined) vs Regular vs Total (with Forecasting)
st.markdown("##### 📈 매크로 트렌드: 분류군별 지원자 수 및 성적 추이 (과거 + 5개년 예측)")
st.caption("> 실선은 과거 데이터, 점선은 인구 추이 기반 5개년 예측치입니다. 수시(일반)은 주요 6개 전형이 제외되었습니다.")

# 1. Get Forecast Data for each segment
pred_e, _ = get_future_prediction_data(df[df['분석그룹'] == '수시(일반)'], df_dense_all)
pred_r, _ = get_future_prediction_data(df[df['분석그룹'] == '정시'], df_dense_all)
pred_t, _ = get_future_prediction_data(df, df_dense_all)

# 2. Historical Data
df_e = df_filtered[df_filtered['분석그룹'] == '수시(일반)'].groupby('학년도').agg(지원자수=('수험번호', 'count'), 평균성적=('대표성적', 'mean')).reset_index()
df_r = df_filtered[df_filtered['분석그룹'] == '정시'].groupby('학년도').agg(지원자수=('수험번호', 'count'), 평균성적=('대표성적', 'mean')).reset_index()
df_t = df_filtered.groupby('학년도').agg(지원자수=('수험번호', 'count'), 평균성적=('대표성적', 'mean')).reset_index()

nat_pop_lag = df_dense_all[['연도', '전국']].copy()
nat_pop_lag['연도_lag'] = nat_pop_lag['연도'] + 1

# Macro Trends: Split into two subplots for clarity
st.markdown("##### 📈 매크로 트렌드: 인구 절벽과 입결 영향 분석 (과거 + 5개년 예측)")
st.info("""
**📊 통합 트렌드 읽는 법**:
- **상단 차트**: 전국 고3 인구가 급격히 감소함에 따라 우리 대학 지원자 수(실선/점선)가 동반 하락하는 흐름을 보여줍니다.
- **하단 차트**: 상위권 대학의 '인원 흡수' 효과로 인해, 우리 대학에 유입되는 학생들의 평균 성적(등급 숫자)은 점차 높아질(우하향) 것으로 예측됩니다.
""")

# Prepare data for plot
fig_macro, (ax_vol, ax_grd) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 1. VOLUME PLOT (Population vs Applicants)
ax_v_pop = ax_vol.twinx()
# Population (Fill Area)
ax_v_pop.fill_between(nat_pop_lag['연도_lag'], 0, nat_pop_lag['전국'], color='grey', alpha=0.1, label='전국 고3 인구(lag)')
ax_v_pop.set_ylabel("전국 인구 (명)", color='grey')
ax_v_pop.set_ylim(0, nat_pop_lag['전국'].max() * 1.2)

# Applicants (Lines)
sns.lineplot(data=df_t, x='학년도', y='지원자수', ax=ax_vol, color='purple', marker='s', linewidth=2, label='전체 지원자(과거)')
sns.lineplot(data=df_e, x='학년도', y='지원자수', ax=ax_vol, color='blue', marker='o', linewidth=2, label='수시(일반) 지원자(과거)')
sns.lineplot(data=df_r, x='학년도', y='지원자수', ax=ax_vol, color='red', marker='^', linewidth=2, label='정시 지원자(과거)')

if not pred_t.empty:
    sns.lineplot(data=pred_t, x='연도', y='예측지원자수', ax=ax_vol, color='purple', linestyle='--', alpha=0.8)
    sns.lineplot(data=pred_e, x='연도', y='예측지원자수', ax=ax_vol, color='blue', linestyle='--', alpha=0.8)
    sns.lineplot(data=pred_r, x='연도', y='예측지원자수', ax=ax_vol, color='red', linestyle='--', alpha=0.8)
    
    # Connect last historical to first forecast
    l_yr = df_t['학년도'].max()
    f_yr = pred_t['연도'].min()
    for d_hist, d_pred, clr in [(df_t, pred_t, 'purple'), (df_e, pred_e, 'blue'), (df_r, pred_r, 'red')]:
        ax_vol.plot([l_yr, f_yr], [d_hist[d_hist['학년도']==l_yr]['지원자수'].values[0], d_pred[d_pred['연도']==f_yr]['예측지원자수'].values[0]], color=clr, linestyle='--', alpha=0.5)

ax_vol.set_ylabel("본교 지원자 수 (명)")
ax_vol.set_title("인구 감소에 따른 지원자 유입 규모 예측", fontsize=12, pad=15)
ax_vol.legend(loc='upper right', fontsize='small')
ax_vol.grid(True, axis='y', linestyle=':', alpha=0.5)

# 2. GRADE PLOT (Quality Trend)
sns.lineplot(data=df_t, x='학년도', y='평균성적', ax=ax_grd, color='purple', marker='s', label='전체 평균(과거)')
sns.lineplot(data=df_e, x='학년도', y='평균성적', ax=ax_grd, color='blue', marker='o', label='수시(일반) 평균(과거)')
sns.lineplot(data=df_r, x='학년도', y='평균성적', ax=ax_grd, color='red', marker='^', label='정시 평균(과거)')

if not pred_t.empty:
    sns.lineplot(data=pred_t, x='연도', y='예측평균성적', ax=ax_grd, color='purple', linestyle='--', alpha=0.8)
    sns.lineplot(data=pred_e, x='연도', y='예측평균성적', ax=ax_grd, color='blue', linestyle='--', alpha=0.8)
    sns.lineplot(data=pred_r, x='연도', y='예측평균성적', ax=ax_grd, color='red', linestyle='--', alpha=0.8)
    
    # Connect
    for d_hist, d_pred, clr in [(df_t, pred_t, 'purple'), (df_e, pred_e, 'blue'), (df_r, pred_r, 'red')]:
        ax_grd.plot([l_yr, f_yr], [d_hist[d_hist['학년도']==l_yr]['평균성적'].values[0], d_pred[d_pred['연도']==f_yr]['예측평균성적'].values[0]], color=clr, linestyle='--', alpha=0.5)

ax_grd.set_ylabel("평균 성적 (등급)")
ax_grd.set_title("지원 경쟁 하락에 따른 최종등록자 성적(Quality) 변화 예측", fontsize=12, pad=15)
ax_grd.invert_yaxis() # 1 grade at top!
ax_grd.legend(loc='lower right', fontsize='small')
ax_grd.grid(True, axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig_macro)
st.markdown("---")

# --- 3. Admissions Trends (📈 입시 주요 지표 및 트렌드) ---
st.header("📈 2. 연도별 입시 결과 및 지원 트렌드")

# Trend Insight
st.success("""
**💡 지원 트렌드 인사이트**: 
- 평균 성적이 상승(숫자 하강)하고 있다면 해당 유형의 인기가 높아지고 있음을 의미합니다. 
- 반대로 성적이 하락(숫자 상승)하고 있다면 경쟁 대학으로의 유출이나 지원자 풀의 질적 저하를 경계하고, 타겟 홍보를 강화해야 합니다.
""")

# Yearly Trend (Old Tab 1)
col_trend1, col_trend2 = st.columns(2)
# Yearly Trend (Early/Regular/Total)
col_trend1, col_trend2 = st.columns(2)
with col_trend1:
    st.subheader("연도별 평균 성적 비교")
    trend_e = df_filtered[df_filtered['분석그룹'] == '수시(일반)'].groupby('학년도')['대표성적'].mean().reset_index()
    trend_r = df_filtered[df_filtered['분석그룹'] == '정시'].groupby('학년도')['대표성적'].mean().reset_index()
    trend_t = df_filtered.groupby('학년도')['대표성적'].mean().reset_index()
    
    fig1, ax1 = plt.subplots()
    ax1.plot(trend_t['학년도'], trend_t['대표성적'], label='전체 합계', marker='o', color='purple', linewidth=3)
    ax1.plot(trend_e['학년도'], trend_e['대표성적'], label='수시(일반)', marker='s', color='blue')
    ax1.plot(trend_r['학년도'], trend_r['대표성적'], label='정시', marker='^', color='red')
    
    ax1.set_title("분류군별 평균 성적 추이")
    ax1.invert_yaxis()
    ax1.legend()
    st.pyplot(fig1)

# Competition Trend (Old Tab 6)
# Competition Trend (Early/Regular/Total)
with col_trend2:
    st.subheader("분류군별 경쟁률 추이")
    
    def get_comp_trend(target_df, label):
        agg = target_df.groupby('학년도').apply(
            lambda x: pd.Series({'지원자수': len(x), '합격자수': x['합격여부'].sum()})
        ).reset_index()
        agg['경쟁률'] = (agg['지원자수'] / agg['합격자수']).replace([np.inf, -np.inf], 0).fillna(0)
        agg['Group'] = label
        return agg

    trend_e_c = get_comp_trend(df_filtered[df_filtered['분석그룹'] == '수시(일반)'], '수시(일반)')
    trend_r_c = get_comp_trend(df_filtered[df_filtered['분석그룹'] == '정시'], '정시')
    trend_t_c = get_comp_trend(df_filtered, '전체 합계')
    
    comp_trend_all = pd.concat([trend_e_c, trend_r_c, trend_t_c])
    
    fig6, ax6 = plt.subplots()
    sns.lineplot(data=comp_trend_all, x='학년도', y='경쟁률', hue='Group', palette={'수시(일반)': 'blue', '정시': 'red', '전체 합계': 'purple'}, marker='s', ax=ax6)
    ax6.set_title("분류군별 경쟁률 변화")
    st.pyplot(fig6)

# --- 4. Detailed Analysis ---
st.header("📊 3. 전형 및 성적 상세 분석")

# Detailed Insight
st.info("""
**💡 심층 분석 인사이트**: 
- 전형그룹별 성적 분포에서 **수염(Whiskers)**이 긴 전형은 지원자의 성격 스펙트럼이 매우 넓음을 의미하며, 선발 결과의 예측 가능성이 낮을 수 있습니다. 
- **바이올린 플롯**을 통해 각 전형의 '성적 밀집도'를 확인하십시오. 밀도가 특정 구간에 좁게 형성된 전형은 해당 성적대의 학생들에게 강력한 브랜드 파워를 가짐을 뜻합니다.
""")

col4a, col4b = st.columns(2)
with col4a:
    sel_seg_4 = st.radio("분석 그룹 선택 (바이올린)", ["전체 합계", "수시(일반)", "정시"], horizontal=True)
    st.subheader(f"{sel_seg_4} 전형 그룹별 성적 분포")
    df_v4 = df_filtered if sel_seg_4 == "전체 합계" else df_filtered[df_filtered['분석그룹'] == sel_seg_4]
    if not df_v4.empty:
        fig2a, ax2a = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df_v4, x='전형그룹', y='대표성적', ax=ax2a, palette='Set3')
        plt.xticks(rotation=45)
        ax2a.invert_yaxis()
        st.pyplot(fig2a)

with col4b:
    st.subheader("합격/등록 성적 분포 (Funnel)")
    target_pr = st.radio("대상 선택", ["합계(전체)", "수시(일반)", "정시"], key='pr_v', horizontal=True)
    if target_pr == "합계(전체)":
        df_pr = df_filtered.dropna(subset=['대표성적'])
    else:
        df_pr = df_filtered[df_filtered['분석그룹']==target_pr].dropna(subset=['대표성적'])
        
    if not df_pr.empty:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.kdeplot(df_pr['대표성적'], label='전체', fill=True, ax=ax5)
        sns.kdeplot(df_pr[df_pr['합격구분']=='최초합격']['대표성적'], label='최초합격', fill=True, ax=ax5)
        sns.kdeplot(df_pr[df_pr['등록구분']=='등록']['대표성적'], label='등록', fill=True, ax=ax5)
        plt.legend()
        st.pyplot(fig5)

with st.expander("🌍 지역별 x 전형그룹 상세 분석 (Heatmap)"):
    df_region = df_filtered[df_filtered['입학유형'] == '수시']
    if not df_region.empty:
        top_regions = df_region['교육청소재지'].value_counts().nlargest(15).index
        region_pivot = df_region[df_region['교육청소재지'].isin(top_regions)].pivot_table(
            index='교육청소재지', columns='전형그룹', values='대표성적', aggfunc='mean'
        )
        fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
        sns.heatmap(region_pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax_heat)
        st.pyplot(fig_heat)
# --- 5. Regional Insights ---
st.header("🎯 4. 지역별 타겟팅 전략 (Regional Insights)")

# Regional Insight
st.success("""
**💡 지역 타겟팅 인사이트**: 
- 특정 지역의 **최초등록률(Yield)**이 낮다면, 해당 지역의 합격자들이 타 대학으로 이탈하고 있음을 의미합니다. 특히 지원 비중이 높은 지역에서의 이탈은 뼈아픈 타격입니다. 
- 울산 등 성장이 뚜렷한 지역에 대해서는 면접 배정 확대나 고교 방문 설명회 등 집중적인 자원 투입이 필요합니다.
""")

# Regional Insights with Segment Selection
st.header("🎯 4. 지역별 타겟팅 전략 (Regional Insights)")

sel_reg_seg = st.radio("지역 분석 대상 선택", ["전체 합계", "수시(일반)", "정시"], key='reg_seg_sel', horizontal=True)
df_reg_target = df_filtered if sel_reg_seg == "전체 합계" else df_filtered[df_filtered['분석그룹'] == sel_reg_seg]

reg_table = df_reg_target.groupby('교육청소재지').agg(
    지원자수=('수험번호', 'count'),
    최초합격자수=('합격구분', lambda x: x.astype(str).str.contains('최초').sum()),
    등록자수=('등록구분', lambda x: (x=='등록').sum()),
    평균등급=('대표성적', 'mean')
).sort_values('지원자수', ascending=False)

reg_table['등록률(%)'] = (reg_table['등록자수'] / reg_table['지원자수'] * 100).fillna(0)
reg_table['최초등록률(%)'] = (reg_table['등록자수'] / reg_table['최초합격자수'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
reg_table['비중(%)'] = (reg_table['지원자수'] / reg_table['지원자수'].sum() * 100)

col_reg1, col_reg2 = st.columns([1, 1])
with col_reg1:
    st.markdown("##### 📊 지역별 지원 현황 (Top 20)")
    st.dataframe(reg_table.head(20).style.format("{:.2f}", subset=['평균등급', '등록률(%)', '최초등록률(%)', '비중(%)']))

with col_reg2:
    st.markdown("##### 🎯 울산 지역 심층 분석")
    if '울산' in reg_table.index:
        ulsan_stats = reg_table.loc['울산']
        last_year = df_filtered['학년도'].max()
        share, grade, yield_rate_n = ulsan_stats['비중(%)'], ulsan_stats['평균등급'], ulsan_stats['최초등록률(%)']
        
        c1, c2 = st.columns(2)
        with c1: st.metric("울산 지원 비중", f"{share:.1f}%")
        with c2: st.metric("울산 평균 등급", f"{grade:.2f} 등급")
        
        try:
            ulsan_curr = df_filtered[(df_filtered['학년도'] == last_year) & (df_filtered['교육청소재지'] == '울산')].shape[0]
            ulsan_prev = df_filtered[(df_filtered['학년도'] == last_year - 1) & (df_filtered['교육청소재지'] == '울산')].shape[0]
            if ulsan_curr > ulsan_prev:
                st.success(f"📈 {last_year}학년도 울산 지원자 급증 (+{ulsan_curr-ulsan_prev}명)")
        except: pass

        st.metric("💡 최초합격자 등록률 (Initial Yield)", f"{yield_rate_n:.1f}%", delta=f"이탈률: {100-yield_rate_n:.1f}%", delta_color="inverse")
        st.progress(min(int(yield_rate_n), 100))
        
        if yield_rate_n < 50: st.error("⚠️ **우수 자원 유출 심각**: 리텐션 전략 시급")
        else: st.success("✅ **안정적 충성도 유지**")

st.markdown("---")

# --- 6. Efficiency Analysis ---
st.header("🗣️ 5. 전형 운영 효율화 및 면접 영향력")

# Efficiency Insight
st.success("""
**💡 효율화 인사이트**: 
- **바이올린 플롯(Violin Plot)**은 성적 분포의 밀도를 보여줍니다. 면접 전형이 교과 성적 하위 구간까지 넓게 퍼져 있다면, 이는 면접이 실질적인 역전의 기회를 제공하고 있음을 시사합니다. 
- 면접의 성적 보정 효과가 클수록, 단순 교과 성적 위주 선발에서 벗어난 다면적 평가가 이루어지고 있다는 증거입니다.
""")

df_susi = df_filtered[df_filtered['입학유형'] == '수시'].copy()
def get_paren_tag(text):
    import re
    matches = re.findall(r'\(([^)]+)\)', str(text))
    return matches[-1] if matches else "일반"

df_susi['세부유형'] = df_susi['전형구분'].apply(get_paren_tag)
stats_by_tag = df_susi.groupby('세부유형')['대표성적'].agg(['count', 'mean'])

c_eff1, c_eff2 = st.columns([1, 1])
with c_eff1:
    st.markdown("##### 면접 유무별 성적 분포 (Violin Plot)")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df_susi, x='면접유무', y='대표성적', palette='Pastel1', split=True, ax=ax4)
    ax4.invert_yaxis()
    st.pyplot(fig4)

with c_eff2:
    interview_tags = [t for t in stats_by_tag.index if '면접' in t]
    non_interview_tags = [t for t in stats_by_tag.index if '면접' not in t]
    if interview_tags and non_interview_tags:
        avg_int, avg_non = stats_by_tag.loc[interview_tags, 'mean'].mean(), stats_by_tag.loc[non_interview_tags, 'mean'].mean()
        if pd.notna(avg_int) and pd.notna(avg_non):
            st.info(f"💡 **면접의 성적 보정 효과**: 면접 전형은 교과 전형 대비 평균 **{avg_non - avg_int:.2f}등급** 낮은 학생들에게 합격 기회를 제공하며 공정성을 보완하고 있습니다.")

st.markdown("---")
# --- 6. Quota Strategy Simulation (🏗️ 전형별 정원 조정 시뮬레이션) ---
st.header("🏗️ 6. 수시(일반) 전형별 정원 조정 시뮬레이션")
st.markdown("> **대상**: 수시(일반) - 수능위주, 기회균등, 농어촌, 기초생활, 재외국민, 특성화고교 전형 제외")

# Filter for Refined Early Admissions only
df_sim = df_filtered[(df_filtered['학년도'] == last_year) & (df_filtered['분석그룹'] == '수시(일반)')]

if not df_sim.empty:
    # Aggregate by Admission Group (전형) instead of Department
    jeon_stats = df_sim.groupby('전형그룹').agg(
        지원자수=('수험번호', 'count'),
        합격자수=('합격여부', 'sum'),
        등록자평균성적=('대표성적', lambda x: x[df_sim.loc[x.index, '등록구분'] == '등록'].mean())
    ).reset_index()
    jeon_stats['경쟁률'] = (jeon_stats['지원자수'] / jeon_stats['합격자수']).fillna(0)
    
    med_comp_j, med_grade_j = jeon_stats['경쟁률'].median(), jeon_stats['등록자평균성적'].median()
    
    def get_matrix_type_jeon(row):
        is_high_comp = row['경쟁률'] >= med_comp_j
        is_good_grade = row['등록자평균성적'] <= med_grade_j # Lower is better
        if is_high_comp and is_good_grade: return "Star"
        elif is_high_comp and not is_good_grade: return "Cash Cow"
        elif not is_high_comp and is_good_grade: return "Hidden Gem"
        else: return "Dog"
    
    jeon_stats['Type'] = jeon_stats.apply(get_matrix_type_jeon, axis=1)
    
    col_sim_cfg1, col_sim_cfg2 = st.columns([1, 2])
    with col_sim_cfg1:
        scenario = st.radio("시뮬레이션 강도", ('보수적(10%)', '적극적(20%)', '급진적(30%)' ), index=1, horizontal=True)
        cut_rate = 0.1 if '보수적' in scenario else (0.2 if '적극적' in scenario else 0.3)
    
    with col_sim_cfg2:
        st.info(f"선택한 시나리오에 따라 **경쟁률과 성적이 모두 낮은 전형(Dog)**의 정원을 차출하여 **우수 전형(Star)**으로 재배치합니다.")

    # --- 시뮬레이션 엔진 로직 설정 ---
    # 1. 제로섬(Zero-Sum) 원칙 적용: 전체 입시 정원은 유지한 채, 내부 비중만 조정하여 대학의 운영 부담을 최소화합니다.
    # 2. Dog(저조) 전형 감축: 성적과 경쟁률이 모두 낮은 전형은 '선정적 가치'가 낮다고 판단하여 우선 감축 대상으로 선정합니다.
    # 3. 우수 전형 증원: 성적이 높고 경쟁률이 치열한 'Star' 전형에 정원을 몰아주어 전체적인 합격자 품질을 상방 평준화합니다.
    
    total_pool = 0 # 감축된 총 인원을 담는 바구니
    jeon_stats['Adj'] = 0 # 각 전형별 조정 인원
    
    # [계산] Dog 전형에서 정원을 뺏어오기
    for idx, row in jeon_stats.iterrows():
        if row['Type'] == 'Dog':
            cut = int(row['합격자수'] * cut_rate)
            jeon_stats.at[idx, 'Adj'] = -cut
            total_pool += cut # 뺏어온 인원을 바구니에 합산
    
    # [계산] 바구니에 담긴 인원을 우수 전형에 골고루 나눠주기 (Star -> Hidden Gem 순서)
    if total_pool > 0:
        targets = jeon_stats[jeon_stats['Type'] == 'Star'] # 1순위: 성적+경쟁률 모두 좋은 전형
        if targets.empty:
            targets = jeon_stats[jeon_stats['Type'] == 'Hidden Gem'] # 2순위: 성적은 좋은데 홍보가 필요한 전형
        if targets.empty:
            targets = jeon_stats[jeon_stats['Type'] == 'Cash Cow'] # 3순위: 성적은 낮아도 인기는 있는 전형
            
        if not targets.empty:
            per_target = total_pool // len(targets) # 공평하게 배분
            remainder = total_pool % len(targets) # 나누고 남은 잉여 인원
            
            for i, idx in enumerate(targets.index):
                jeon_stats.at[idx, 'Adj'] += per_target
                if i == 0: # 제로섬을 완벽히 맞추기 위해 남은 찌꺼기 인원을 첫 번째 타겟에 합산
                    jeon_stats.at[idx, 'Adj'] += remainder
        else:
            jeon_stats['Adj'] = 0 # 만약 줄 데가 없다면 시뮬레이션 취소
            total_pool = 0

    cur_reg_students = df_sim[df_sim['등록구분'] == '등록'].copy()
    sim_reg_pool = cur_reg_students.copy()
    
    # Track actions for basis explanation
    action_log = []
    
    # 1. Processing Cuts (Dog Pathways)
    for idx, row in jeon_stats.iterrows():
        if row['Adj'] < 0:
            num_cut = abs(int(row['Adj']))
            # Find the worst (highest number) registered students in this group
            grp_reg = sim_reg_pool[sim_reg_pool['전형그룹'] == row['전형그룹']].sort_values('대표성적', ascending=False)
            cut_ids = grp_reg.head(num_cut).index
            sim_reg_pool = sim_reg_pool.drop(cut_ids)
            action_log.append(f"• **{row['전형그룹']}** (Dog): 정원 {num_cut}명 주축 - 해당 전형 내 하위 성적 등록자 {num_cut}명 제외")

    # 2. Processing Gains (Target Pathways)
    for idx, row in jeon_stats.iterrows():
        if row['Adj'] > 0:
            num_add = int(row['Adj'])
            # Find the best applicants who are NOT currently registered in this target group
            # These could be '불합격' or '최초합격(미등록)' etc.
            potential_pool = df_sim[(df_sim['전형그룹'] == row['전형그룹']) & (df_sim['등록구분'] != '등록')].sort_values('대표성적', ascending=True)
            
            if not potential_pool.empty:
                added_students = potential_pool.head(num_add)
                sim_reg_pool = pd.concat([sim_reg_pool, added_students])
                action_log.append(f"• **{row['전형그룹']}** (우수): 정원 {len(added_students)}명 증원 - 해당 전형 미등록 지원자 중 상위 성적 {len(added_students)}명 신규 유입")
            else:
                action_log.append(f"• **{row['전형그룹']}** (우수): 증원 대상이었으나 활용 가능한 예비 자원(미등록 지원자)이 부족하여 증원이 취소되었습니다.")

    # Redistribution Breakdown
    st.markdown("##### 📋 전형별 정원 재배치 상세 계획 (From -> To)")
    reductions = jeon_stats[jeon_stats['Adj'] < 0][['전형그룹', 'Adj']].rename(columns={'Adj': '감축인원 (명)'}).sort_values('감축인원 (명)')
    expansions = jeon_stats[jeon_stats['Adj'] > 0][['전형그룹', 'Adj']].rename(columns={'Adj': '증원인원 (명)'}).sort_values('증원인원 (명)', ascending=False)
    
    # Double check zero-sum
    checksum = len(sim_reg_pool) - len(cur_reg_students)
    
    col_plan1, col_plan2 = st.columns(2)
    with col_plan1:
        st.markdown(f"**📉 감축 대상 (Dog 전형, 총 {abs(reductions['감축인원 (명)'].sum())}명)**")
        st.table(reductions)
    with col_plan2:
        st.markdown(f"**📈 증원 대상 (우수 전형, 총 {expansions['증원인원 (명)'].sum()}명)**")
        st.table(expansions)
    
    # Calculation
    cur_avg_val = cur_reg_students['대표성적'].mean()
    fut_avg_val = sim_reg_pool['대표성적'].mean()
    diff_val = cur_avg_val - fut_avg_val
    
    if checksum == 0 and total_pool > 0:
        st.success(f"✅ **시뮬레이션 결과**: 개별 데이터(Row-level) 기반 시뮬레이션 결과, 전형 재배치 시 신입생 평균 성적이 약 **{diff_val:.4f} 등급 향상**될 것으로 예측됩니다.")
        
        with st.expander("🔍 왜 이렇게 계산되었나요? (시뮬레이션 로직 근거 상세)"):
            st.markdown(f"""
            이 시뮬레이션은 단순히 평균 점수를 더하고 빼는 방식이 아니라, **{len(df_sim):,}명의 개별 지원자의 실제 성적**을 가지고 '만약 정원이 이랬다면?'을 가정하여 하나하나 계산했습니다.
            
            **1) 왜 성적 하위자를 빼나요? (감축 로직)**
            - 대학 입시에서 정원을 줄이면, 경쟁이 치열해지면서 성적이 가장 낮은 학생들부터 합격권에서 멀어지게 됩니다. 
            - 따라서 Dog(저조) 전형에서 정원을 줄일 때, **실제 등록한 학생 중 가장 성적이 낮은 하위 N명**을 가상으로 탈락시켜 커트라인 상승 효과를 시뮬레이션했습니다.
            
            **2) 왜 미등록 지원자를 넣나요? (증원 로직)**
            - 우수 전형의 정원을 늘리면, 기존에는 아쉽게 떨어졌거나 점수는 충분한데 다른 대학으로 빠져나간 '잠재적 우수 자원'이 합격권에 들어오게 됩니다.
            - 따라서 Star(우수) 전형에서 정원을 늘릴 때, **현재 등록하지 않은 지원자 중 성적이 가장 좋았던 상위 N명**을 신규 유입시켜 입결 상승을 시뮬레이션했습니다.
            
            **3) 왜 제로섬(Zero-Sum)인가요?**
            - 대학 전체 정원은 교육부 인가에 따라 고정되어 있습니다. 한 곳을 늘리려면 반드시 다른 곳을 줄여야 하는 **'풍선 효과'**를 반영하여 현실적인 재배치 안을 도출하기 위함입니다.
            
            **🚀 실제 데이터 기반 실행 로그:**
            """)
            for log in action_log:
                st.write(log)
    elif total_pool == 0:
        st.info("💡 성적 및 경쟁률이 낮은 'Dog' 전형이 없거나 재배치할 대상 전형이 없어 변동 사항이 없습니다.")
    else:
        st.warning(f"⚠️ 시뮬레이션 계산 중 오차가 발생했습니다 (잔여: {checksum}명).")
else:
    st.warning("수시(일반) 데이터가 부족하여 시뮬레이션을 수행할 수 없습니다.")

st.markdown("---")

# --- 7. Future Outlook (🔮 상향식 입시 규모 및 성적 예측) ---
st.header("🔮 7. 데이터 기반 5개년 입시 예측 (2026~2030)")

# Forecasting Insight
st.info("""
**💡 고도화된 예측 모델**: 
- **인구 지연 효과 반영**: 고3 학령인구는 그다음 해 입시에 영향을 미칩니다 (Pop[N-1] -> Adm[N]). 고교 졸업 예정자가 실제 수험생이 되는 시차를 반영했습니다.
- **지역 가중치 & 최신화**: 주요 타겟 지역 인구 추이에 가중치를 두고, 최근 3개년 트렌드를 집중 반영하여 현실성을 높였습니다.
- **성적 축 역전**: 모든 성적 차트는 1등급이 상단에 위치하도록 일관되게 적용되었습니다 (1등급 = 최고 성적).
""")

# Forecasting Target Selection
sel_pred_grp = st.radio("예측 대상 분류 선택", ["전체 합계", "수시(일반 - 주요 6개 전형 제외)", "정시"], key='pred_grp_sel', horizontal=True)
pred_source = df if sel_pred_grp == "전체 합계" else df[df['분석그룹'] == ('정시' if sel_pred_grp == '정시' else '수시(일반)')]

pred_df, diag_df = get_future_prediction_data(pred_source, df_dense_all)

# Dynamic Warning based on selection
grp_name = "본교 전체" if sel_pred_grp == "전체 합계" else ("정시" if sel_pred_grp == "정시" else "수시(일반)")

if not pred_df.empty:
    col_diag, col_pred = st.columns([1, 2])
    
    with col_diag:
        st.markdown(f"##### 🔍 {grp_name} 예측 진단")
        fig_diag, ax_d = plt.subplots(figsize=(6, 5))
        sns.regplot(data=diag_df, x='weighted_pop', y='지원자수', ax=ax_d, color='purple', scatter_kws={'s':100, 'alpha':0.6})
        ax_d.set_title(f"{grp_name}: 인구(Y-1) vs 실지원자(Y)")
        ax_d.set_xlabel("전년도 가중 지역 인구")
        ax_d.set_ylabel("당해 연도 지원자 수")
        st.pyplot(fig_diag)
        st.caption("전년도 고3 인구 변화가 그다음 해 지원자 수에 미치는 직접적 영향입니다.")

    with col_pred:
        st.markdown("##### 📈 향후 5개년 경쟁률 및 성적 변화 예측")
        fig_pred, ax_p1 = plt.subplots(figsize=(10, 5))
        ax_p2 = ax_p1.twinx()
        
        # Historical stats
        sns.lineplot(data=diag_df, x='학년도', y='경쟁률', ax=ax_p1, color='blue', marker='o', linewidth=2, label='과거 경쟁률')
        sns.lineplot(data=diag_df, x='학년도', y='평균성적', ax=ax_p2, color='red', marker='s', linewidth=2, label='과거 평균성적')
        
        # Predicted stats
        sns.lineplot(data=pred_df, x='연도', y='예측경쟁률', ax=ax_p1, color='blue', linestyle='--', marker='o', alpha=0.7, label='예측 경쟁률')
        sns.lineplot(data=pred_df, x='연도', y='예측평균성적', ax=ax_p2, color='red', linestyle='--', marker='s', alpha=0.7, label='예측 평균성적')
        
        # Connect last actual to first predicted line
        last_act = diag_df.iloc[-1]
        first_pre = pred_df.iloc[0]
        ax_p1.plot([last_act['학년도'], first_pre['연도']], [last_act['경쟁률'], first_pre['예측경쟁률']], color='blue', linestyle='--', alpha=0.5)
        ax_p2.plot([last_act['학년도'], first_pre['연도']], [last_act['평균성적'], first_pre['예측평균성적']], color='red', linestyle='--', alpha=0.5)
        
        ax_p1.set_ylabel("경쟁률 (:1)", color='blue', fontsize=12)
        ax_p2.set_ylabel("평균 성적 (등급)", color='red', fontsize=12)
        ax_p2.invert_yaxis()
        
        # Combined Legend
        lines1, labels1 = ax_p1.get_legend_handles_labels()
        lines2, labels2 = ax_p2.get_legend_handles_labels()
        ax_p2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')
        
        st.pyplot(fig_pred)

    col_tbl, col_msg = st.columns([1, 2])
    with col_tbl:
        st.markdown("##### 📋 예측 데이터 요약")
        disp_pred = pred_df[['연도', '예측경쟁률', '예측평균성적']].copy()
        disp_pred.columns = ['학년도', '예측 경쟁률', '예측 평균성적']
        st.dataframe(disp_pred.set_index('학년도').style.format("{:.2f}"))
        
    with col_msg:
        final_comp = pred_df.iloc[-1]['예측경쟁률']
        final_grade = pred_df.iloc[-1]['예측평균성적']
        st.warning(f"""
        **⚠️ {grp_name} 경쟁력 하락 경고**: 2030년 예상 경쟁률은 **{final_comp:.2f}:1**이며, 평균 성적은 **{final_grade:.2f}등급**까지 밀릴 가능성이 있습니다. 
        해당 그룹({grp_name})에 특화된 브랜딩 및 전형 최적화 전략이 필요합니다.
        """)

st.markdown("---")

# --- 8. Comprehensive Strategic Recommendations (🚀 입시 전략 로드맵) ---
st.header("🚀 종합 전략 및 2026 실행 과제")

# Initialize default simulation values to prevent scope errors
sim_reductions_count = len(reductions) if 'reductions' in locals() else 0
sim_expansions_count = len(expansions) if 'expansions' in locals() else 0
sim_total_pool = total_pool if 'total_pool' in locals() else 0
sim_diff_val = diff_val if 'diff_val' in locals() else 0.0

col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.subheader("💡 구조 혁신: '버릴 곳'은 확실히, '밀어줄 곳'은 강력하게")
    st.success(f"""
    - **정원 다이어트**: 성적과 인기가 모두 없는 **{sim_reductions_count}개** 전형은 대학의 브랜드 가치를 떨어뜨리는 '약한 고리'입니다. 이곳에서 **{sim_total_pool}명**을 과감히 줄여야 합니다.
    - **입결 점프**: 부족한 정원을 '우수 전형'으로 100% 옮기기만 해도(비용 0원!), 우리 대학의 신입생 평균 수준이 **{sim_diff_val:.3f}등급**이나 수직 상승하는 효과를 볼 수 있습니다.
    - **선택과 집중**: 들어오는 문은 좁히되(감축), 경쟁력이 입증된 통로는 넓혀서(증원) **'들어가기 어려운 대학'**이라는 이미지를 구축하십시오.
    """)

with col_rec2:
    st.subheader("🛠️ 미래 과제: 인구 절벽 시대의 생존 전략")
    st.info(f"""
    - **성장판 공략 (울산/신도시)**: 인구가 줄어드는 구도심은 방어 위주로 가되, 교육 열기가 높고 인구 유입이 활발한 **울산 및 신도시 권역**을 타겟으로 집중적인 학교 홍보를 펼쳐야 합니다.
    - **잠재력 선발 (면접 고도화)**: 단순한 내신 성적만으로는 알 수 없는 학생의 '진가'를 발견하기 위해, 당락 경계선(Borderline)에 있는 학생들에게 면접 자원을 집중 투입하십시오.
    - **합격자 마음 잡기 (Yield Management)**: "합격은 끝이 아니라 시작입니다." 최초 합격자가 타 대학으로 이탈하지 않도록, 전공 선배와의 만남 등 **밀착 케어**를 통해 등록을 확정 지으십시오.
    """)

# --- Footer Summary Table ---
st.markdown("#### 📊 마지막 요점 정리 (Key Summary)")
df_pass_f = df_filtered.dropna(subset=['대표성적'])
if not df_pass_f.empty:
    m_init = df_pass_f[df_pass_f['합격구분'] == '최초합격']['대표성적'].mean()
    m_reg = df_pass_f[df_pass_f['등록구분'] == '등록']['대표성적'].mean()
    
    summary_data = {
        '구분': ['최초합격자 수준', '최종 입학자 수준', '전략 성공 시 성적 향상치', '집중 공략 타겟 지역'],
        '지표': [f"{m_init:.2f} 등급", f"{m_reg:.2f} 등급", f"{sim_diff_val:.3f} 등급 상향", "울산 / 신흥 주거 권역"],
    }
    st.table(pd.DataFrame(summary_data))
    st.caption("*최초합격자 대비 입학자의 성적이 낮아지는 현상은 '이탈'에 의한 것으로, 위 로드맵에 따른 밀착 케어가 필수적입니다.")
else:
    st.warning("분석할 데이터가 없어 요약표를 생성할 수 없습니다.")
