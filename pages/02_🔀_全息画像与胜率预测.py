import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import json
import uuid
import streamlit.components.v1 as components

# ==========================================
# 1. 页面配置与目录设定
# ==========================================
st.set_page_config(page_title="选手排行", page_icon="🔀", layout="wide")
DATA_DIR = 'match_database'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_all_match_files():
    """获取所有比赛文件列表"""
    return sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

# ==========================================
# 2. 核心算法：ELO 天梯积分
# ==========================================
def get_k_factor(games_played):
    if games_played < 10: return 40
    else: return 20

def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def process_elo_history(df):
    all_players = pd.concat([df['player_A'], df['player_B']]).unique()
    ratings = {player: 1500 for player in all_players if str(player) != 'nan' and player != "Unknown"}
    games_count = {player: 0 for player in ratings}
    
    match_results = df.sort_values(['match_id', 'rally']).groupby('match_id').tail(1).copy()
    
    for _, row in match_results.iterrows():
        pA, pB = row['player_A'], row['player_B']
        if pA not in ratings or pB not in ratings: continue
        
        old_rA, old_rB = ratings[pA], ratings[pB]
        expected_a = calculate_expected_score(old_rA, old_rB)
        actual_a = 1 if row['winner'] == 'A' else 0
        
        kA, kB = get_k_factor(games_count[pA]), get_k_factor(games_count[pB])
        
        ratings[pA] += round(kA * (actual_a - expected_a))
        ratings[pB] += round(kB * ((1 - actual_a) - (1 - expected_a)))
        games_count[pA] += 1
        games_count[pB] += 1
        
    return ratings

# ==========================================
# 3. 核心算法：个人能力六维统计与胜率基础
# ==========================================
def calculate_single_player_stats(df, player):
    part = df[(df['player_A'] == player) | (df['player_B'] == player)].copy()
    if len(part) < 3: return None

    part['is_A'] = part['player_A'] == player
    part['i_won'] = ((part['is_A']) & (part['winner'] == 'A')) | ((~part['is_A']) & (part['winner'] == 'B'))

    wins = part[part['i_won']]
    losses = part[~part['i_won']]
    if len(wins) == 0: return None

    offense = len(wins[wins['type'] == '杀球']) / len(wins) * 200
    tech = len(wins[wins['type'].isin(['网前球', '平抽', '吊球', '挑球'])]) / len(wins) * 200
    defense_situations = part[((part['is_A']) & (part['smasher'] == 'B')) | ((~part['is_A']) & (part['smasher'] == 'A'))]
    defense = defense_situations['i_won'].mean() * 200 if len(defense_situations) > 0 else 60
    
    clutch_part = part[(abs(part['score_A'] - part['score_B']) <= 3) | (part['score_A'] >= 18) | (part['score_B'] >= 18)]
    mental = clutch_part['i_won'].mean() * 100 if len(clutch_part) > 0 else 60
    
    high_pressure_part = part[part['pressure_score'] > 60]
    physical = high_pressure_part['i_won'].mean() * 100 + 20 if len(high_pressure_part) > 0 else part['pressure_score'].mean()
    
    stability = 100 - (len(losses[losses['type'] == '失误']) / max(len(losses), 1)) * 100

    def clip(x): return int(np.clip(x, 40, 95))

    return {
        "进攻": clip(offense), "防守": clip(defense),
        "技术": clip(tech), "心理": clip(mental),
        "身体": clip(physical), "稳定性": clip(stability)
    }

def get_player_base_stats(df, player):
    part = df[(df['player_A'] == player) | (df['player_B'] == player)].copy()
    if len(part) < 3: return None
    part['is_A'] = part['player_A'] == player
    part['i_won'] = ((part['is_A']) & (part['winner'] == 'A')) | ((~part['is_A']) & (part['winner'] == 'B'))
    
    wins, total = part['i_won'].sum(), len(part)
    return {"win_rate": (wins + 2) / (total + 4)}  # 贝叶斯平滑

def simulate_match_monte_carlo(p_start_mean, num_simulations=1000):
    a_wins = 0
    for _ in range(num_simulations):
        p0 = np.clip(p_start_mean, 0.35, 0.65)
        curr_p = np.random.beta(p0 * 20, (1 - p0) * 20)
        curr_a, curr_b = 0, 0
        while True:
            if curr_a >= 21 and curr_a >= curr_b + 2: a_wins += 1; break
            if curr_b >= 21 and curr_b >= curr_a + 2: break
            if curr_a == 30: a_wins += 1; break
            if curr_b == 30: break
            if curr_a >= 18 or curr_b >= 18:
                curr_p = 0.5 * curr_p + 0.5 * np.random.uniform(0.4, 0.6)
            if random.random() < curr_p: curr_a += 1
            else: curr_b += 1
            curr_p += 0.15 * (p0 - curr_p)
            curr_p = np.clip(curr_p, 0.35, 0.65)
    final_rate = np.random.beta(a_wins + 3, num_simulations - a_wins + 3)
    return float(np.clip(final_rate, 0.05, 0.95))

# ==========================================
# 4. ECharts 自动动画雷达图（支持1-2人，自动异色）
# ==========================================
def render_comparison_animated_radar(players_data):
    radar_id = f"radar_{uuid.uuid4().hex}"
    first_player = list(players_data.keys())[0]
    categories = list(players_data[first_player].keys())
    indicators = [{"name": k, "max": 100} for k in categories]

    series_data = [{
        "name": name,
        "value": [int(stats[c]) for c in categories]
    } for name, stats in players_data.items()]

    html = f"""
    <div id="{radar_id}" style="width:100%;height:460px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <script>
    const chart = echarts.init(document.getElementById("{radar_id}"));
    const targets = {json.dumps(series_data)};
    const indicators = {json.dumps(indicators)};

    let data = targets.map(t => ({{
        name: t.name,
        value: Array(t.value.length).fill(0)
    }}));

    chart.setOption({{
        title: {{ text: "能力维度对比", left: "center" }},
        legend: {{ bottom: 0 }},
        tooltip: {{}},
        radar: {{ 
            indicator: indicators,
            shape: 'polygon',
            splitArea: {{ areaStyle: {{ color: ['#fff', '#f8f9fa', '#e9ecef', '#dee2e6'] }} }}
        }},
        series: [{{ type: 'radar', data: data, areaStyle: {{opacity:0.25}} }}]
    }});

    let step = 0;
    const steps = 30;
    const timer = setInterval(() => {{
        step++;
        data = targets.map(t => ({{
            name: t.name,
            value: t.value.map(v => v * step / steps)
        }}));
        chart.setOption({{ series: [{{ data: data }}] }});
        if (step >= steps) clearInterval(timer);
    }}, 40);
    </script>
    """
    components.html(html, height=480)

# ==========================================
# 5. 页面主逻辑
# ==========================================
st.title("🔀 全息画像 | 选手全维能力与胜率预演")
st.markdown("综合【个人六维能力模型】与【天梯ELO积分】，全方位对比选手能力并推演巅峰对决。")
st.divider()
all_files = get_all_match_files()

if not all_files:
    st.warning("⚠️ 数据库文件夹为空。请先去【数据录入】页面录入或导入比赛文件。")
else:
    # --- 顶栏：数据池选择 ---
    with st.expander("📁 调整分析数据池 (默认选中全部)", expanded=False):
        selected_files = st.multiselect("请选择要纳入计算的比赛：", all_files, default=all_files)
    
    if not selected_files:
        st.info("请至少选择一场比赛来进行分析。")
        st.stop()
        
    # 读取并合并选中的比赛数据
    df = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f)) for f in selected_files], ignore_index=True)
    
    # 计算当前数据池的天梯积分
    final_ratings = process_elo_history(df)
    rank_df = pd.DataFrame([{"选手": k, "天梯积分": v} for k, v in final_ratings.items()])
    if rank_df.empty:
        st.warning("所选比赛中没有有效的胜负记录来计算排名。")
        st.stop()
        
    rank_df = rank_df.sort_values("天梯积分", ascending=False).reset_index(drop=True)
    # 为表格生成排名索引（作为单独一列展示，方便选中后不会乱序）
    rank_df.insert(0, "排名", range(1, len(rank_df) + 1))

    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📈 实时天梯榜单")
        st.caption("👇 **请直接在下方表格左侧的复选框中勾选 1~2 名选手**")
        
        # 使用 st.dataframe 的 on_select 捕捉用户的点击选择
        selection_event = st.dataframe(
            rank_df, 
            use_container_width=True, 
            height=500,
            hide_index=True,
            on_select="rerun",           # 用户点击勾选时自动刷新页面并提取数据
            selection_mode="multi-row"   # 允许多选
        )
        
        # 获取用户在表格中勾选的行索引
        selected_indices = selection_event.selection.rows
        
        # 根据行索引提取出对应的选手姓名
        selected_players = rank_df.iloc[selected_indices]['选手'].tolist() if selected_indices else []

        # 限制最多只能选2名
        if len(selected_players) > 2:
            st.warning("⚠️ 最多只能选择 2 名选手进行对比！系统已自动为您截取前两名。")
            selected_players = selected_players[:2]

    with col2:
        if not selected_players:
            st.info("👈 请在左侧榜单中**勾选选手所在行**（最多勾选2人），即可在此处查看雷达图和胜率。")
        else:
            # 1. 生成雷达图
            st.subheader("📊 能力雷达对比")
            radar_data = {}
            for p in selected_players:
                stats = calculate_single_player_stats(df, p)
                if stats: radar_data[p] = stats
            
            if radar_data:
                render_comparison_animated_radar(radar_data)
                
            # 2. 如果选择了两人，结合天梯和能力进行胜率预测
            if len(selected_players) == 2:
                pA, pB = selected_players
                st.subheader("⚔️ 巅峰对决：综合胜率预测")
                st.caption("综合【个人六维能力模型】与【天梯ELO积分】的蒙特卡洛多回合模拟结果：")
                
                sA = get_player_base_stats(df, pA)
                sB = get_player_base_stats(df, pB)
                
                if sA and sB:
                    # 个人能力维度的胜率预期
                    p_stats = sA['win_rate'] / (sA['win_rate'] + sB['win_rate'])
                    # 天梯 ELO 维度的胜率预期
                    p_elo = calculate_expected_score(final_ratings.get(pA, 1500), final_ratings.get(pB, 1500))
                    
                    # 综合计算：两项权重各占 50%
                    p_start_combined = (p_stats + p_elo) / 2
                    
                    final_rate = simulate_match_monte_carlo(p_start_combined)
                    
                    c1, c2, c3 = st.columns([2, 2, 1])
                    c1.metric(f"🟦 {pA} 胜率", f"{final_rate:.1%}", f"当前积分: {final_ratings.get(pA,1500)}")
                    c2.metric(f"🟩 {pB} 胜率", f"{1-final_rate:.1%}", f"当前积分: {final_ratings.get(pB,1500)}")
                    
                    if final_rate > 0.5:
                        c3.success(f"🏅 模型看好：\n**{pA}**")
                    else:
                        c3.success(f"🏅 模型看好：\n**{pB}**")
                        
                    st.progress(final_rate)
                else:
                    st.warning("⚠️ 其中一方在此数据池中的有效回合数据不足，无法完成科学的胜率模拟。")