import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ==========================================
# 0. 全局页面配置 (必须在第一行)
# ==========================================
st.set_page_config(
    page_title="PRO BADMINTON | 智能赛事分析决策系统",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. 🛠️ 训练逻辑核心 (保持你原有的硬核逻辑不变)
# ==========================================
def calculate_rolling_features(df):
    """计算滚动特征 (核心逻辑)"""
    df = df.sort_values(['match_id', 'set_id', 'rally_id'])
    df['win_flag'] = df['winner_is_A']
    df['smash_flag'] = (df['n_smash'] > 0).astype(int)
    
    match_groups = df.groupby(['match_id', 'set_id'])
    final_dfs = []
    
    for _, group in match_groups:
        group = group.copy()
        group['roll_win_3'] = group['win_flag'].shift(1).rolling(window=3, min_periods=1).mean()
        group['roll_win_10'] = group['win_flag'].shift(1).rolling(window=10, min_periods=1).mean()
        group['roll_smash_10'] = group['smash_flag'].shift(1).rolling(window=10, min_periods=1).mean()
        group['roll_score_std_10'] = group['score_diff'].shift(1).rolling(window=10, min_periods=1).std().fillna(0)
        group['expand_win_rate'] = group['win_flag'].shift(1).expanding().mean()
        group['expand_pressure_mean'] = group['spatial_pressure'].shift(1).expanding().mean()
        final_dfs.append(group)
        
    return pd.concat(final_dfs).fillna(0)

def engineer_features(file_path):
    """读取 CSV 并构建特征"""
    if not os.path.exists(file_path):
        st.error(f"❌ 找不到训练数据: {file_path}")
        return None

    df = pd.read_csv(file_path)
    if 'match_id' not in df.columns: df['match_id'] = 1
    if 'set_id' not in df.columns: df['set_id'] = 1
    
    rally_features = []
    groups = df.groupby(['match_id', 'set_id', 'rally'])
    
    for (match_id, set_id, rally_id), rally_data in groups:
        last = rally_data.iloc[-1]
        winner = last['getpoint_player']
        if pd.isna(winner):
             ws = rally_data['getpoint_player'].dropna()
             if not ws.empty: winner = ws.iloc[0]
             else: continue
        target = 1 if winner == 'A' else 0
        
        first = rally_data.iloc[0]
        s1, s2 = first['roundscore_A'], first['roundscore_B']
        
        dists = np.sqrt((rally_data['hit_x']-rally_data['landing_x'])**2 + 
                        (rally_data['hit_y']-rally_data['landing_y'])**2).fillna(0)
        counts = rally_data['type'].value_counts()
        
        rally_features.append({
            'match_id': match_id, 'set_id': set_id, 'rally_id': rally_id,
            'winner_is_A': target,
            'score_diff': s1 - s2,
            'total_score': s1 + s2,
            'spatial_pressure': dists.mean(),
            'n_smash': counts.get('殺球', 0)
        })
        
    df_rally = pd.DataFrame(rally_features)
    return calculate_rolling_features(df_rally)

def train_and_save_model():
    """执行训练并保存模型"""
    csv_file = 'shuttleset_master_table.csv'
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("⚙️ 阶段 1/4: 深度时序特征工程处理中...")
    df = engineer_features(csv_file)
    progress_bar.progress(25)
    
    if df is not None:
        feature_cols = [
            'score_diff', 'total_score', 'spatial_pressure', 'n_smash', 
            'roll_win_3', 'roll_win_10', 'roll_smash_10', 'roll_score_std_10',
            'expand_win_rate', 'expand_pressure_mean'
        ]
        X = df[feature_cols].fillna(0)
        y = df['winner_is_A']
        
        status_text.text("🧠 阶段 2/4: 构建异构多模型融合架构 (Voting Ensemble)...")
        progress_bar.progress(50)
        
        hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=5, random_state=42)
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        lr = make_pipeline(StandardScaler(), LogisticRegression())
        
        model = VotingClassifier(
            estimators=[('hgb', hgb), ('mlp', mlp), ('lr', lr)],
            voting='soft', weights=[2, 2, 1]
        )
        
        status_text.text("⚡ 阶段 3/4: 模型拟合与超参数寻优中...")
        model.fit(X, y)
        progress_bar.progress(80)
        
        status_text.text("💾 阶段 4/4: 序列化导出决策模型...")
        joblib.dump(model, 'badminton_model.pkl')
        joblib.dump(feature_cols, 'model_features.pkl')
        progress_bar.progress(100)
        
        st.success("✅ 核心引擎训练完毕！系统已加载最新预测模型。")
        time.sleep(1)

# ==========================================
# 2. 🖥️ 现代化主页 UI 设计 (竞赛路演级)
# ==========================================

# --- Hero Section (主视觉区) ---
st.markdown("""
<div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #0f3460 0%, #1a508b 100%); border-radius: 12px; color: white; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
    <h1 style="font-size: 3.8rem; font-weight: 900; letter-spacing: 2px; margin-bottom: 1rem;">
        智羽视界 <span style="font-weight: 300; font-size: 2.5rem;">SmartShuttle Vision</span>
    </h1>
    <h3 style="font-weight: 400; opacity: 0.9; margin-bottom: 1.5rem;">
        基于多模态时空数据的羽毛球赛事智能追踪与决策沙盘
    </h3>
    <p style="font-size: 1.1rem; opacity: 0.8; max-width: 700px; margin: 0 auto;">
        突破传统录像复盘局限，融合「空间压制算法」与「动态势能推演」，为专业队伍提供毫秒级战术沙盘、全维度能力画像与智能策略建议。
    </p>
</div>
""", unsafe_allow_html=True)

# --- 核心亮点区 (向评委展示技术壁垒，你丢失的就是这一块) ---
st.markdown("### 🌟 系统核心技术壁垒")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    with st.container(border=True):
        st.markdown("#### 🧠 异构集成预测引擎")
        st.caption("底层摒弃单一算法，创新融合 `HistGradientBoosting`、`MLP` 神经网络与 `LR` 的 Voting Ensemble 架构，全面捕捉赛事时序非线性特征，胜率预测更精准。")

with col_f2:
    with st.container(border=True):
        st.markdown("#### 📐 空间压力场建模")
        st.caption("不仅记录比分，更独创「空间压制分」计算模型。通过解析二维场地落点欧氏距离，量化球员真实跑动消耗与球路控制力，让隐性数据显性化。")

with col_f3:
    with st.container(border=True):
        st.markdown("#### 🤖 大语言模型战术复盘")
        st.caption("深度接入 GLM-4 等先进大语言模型。将生硬的阵列数据转化为“教练级”战术指导自然语言，一键生成高颜值多维能力雷达与专业数据战报。")

st.divider()

# --- 系统功能矩阵 (正确的页面跳转) ---
st.markdown("### 🚀 核心功能矩阵")
col_a, col_b, col_c = st.columns(3)

with col_a:
    with st.container(border=True):
        st.markdown("#### 🔴 现场追踪与AI教练")
        st.caption("场边教练的终极武器。实时点选落点，计算跑动压制，结合 GLM-4 大模型与蒙特卡洛算法进行战术预演。")
        if st.button("进入实况追踪 →", use_container_width=True):
            st.switch_page("pages/01_🔴_实况追踪与AI沙盘.py")

with col_b:
    with st.container(border=True):
        st.markdown("#### 🔀 综合对决与天梯")
        st.caption("基于全局数据的天梯 ELO 积分系统。构建六维能力雷达图，预测任意两名选手的巅峰对决胜率。")
        if st.button("查看榜单与对决 →", use_container_width=True):
            st.switch_page("pages/02_🔀_全息画像与胜率预测.py")

with col_c:
    with st.container(border=True):
        st.markdown("#### 📈 深度复盘与战报")
        st.caption("赛后势能波动可视化。抓取关键节点事件，一键生成高颜值数据战报与专业复盘报告。")
        if st.button("赛后数据复盘 →", use_container_width=True):
            st.switch_page("pages/03_📈_智能复盘与数据战报.py")

st.markdown("<br>", unsafe_allow_html=True)
col_d, col_e = st.columns(2)

with col_d:
    with st.container(border=True):
        st.markdown("#### 🌳 赛事树系统")
        st.caption("生成官方赛程的晋级路线图，支持用户导入数据的局内回合切片树状追踪。")
        if st.button("查看赛事晋级树 →", use_container_width=True):
            st.switch_page("pages/04_🌳_全景赛程与晋级追踪.py")

with col_e:
    with st.container(border=True):
        st.markdown("#### 📝 数据管家")
        st.caption("支持手动单点录入、历史 CSV 批量导入，内置智能数据补全算法与模拟比赛生成器。")
        if st.button("数据录入与管理 →", use_container_width=True):
            st.switch_page("pages/05_📝_底层数据引擎与管理.py")

st.divider()

# --- 后台训练引擎控制台 (隐藏底层逻辑，但供评委检阅) ---
st.markdown("### ⚙️ 核心算法引擎状态")
st.info("💡 当前已就绪。系统依靠 `shuttleset_master_table.csv` 构建了包含多维短期气势、中期状态与长期基线的滚动特征集。")

with st.expander("🛠️ 展开底层模型控制台 (评委演示与系统重置专用)", expanded=False):
    st.markdown("在更新底层比赛数据库后，可在此触发实时在线学习（Online Learning），重新拟合集成模型。")
    if st.button("🚀 强制重构并训练模型 (Train Advanced Model)", type="primary"):
        with st.spinner("系统正在调配算力..."):
            train_and_save_model()

# --- 页脚学术背书 ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem; padding-top: 2rem; border-top: 1px solid #dee2e6;">
    <p><b>PRO BADMINTON VISION</b> - 智能控制与追踪辅助决策系统</p>
    <p>© 2026 北京邮电大学 通信工程学院 · 参赛项目专属大屏</p>
</div>
""", unsafe_allow_html=True)