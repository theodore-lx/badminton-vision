import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="数据录入", page_icon="📝", layout="wide")

# ==========================================
# 1. 核心配置与工具函数 (多文件管理系统)
# ==========================================
DATA_DIR = 'match_database'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 默认主数据库文件名（用于初始化）
MAIN_DB_FILE = 'match_data.csv' 

STANDARD_COLUMNS = [
    "match_id", "set_id", "rally",      
    "player_A", "player_B",             
    "score_A", "score_B",               
    "winner", "type",                             
    "pressure_score", "smasher",                          
    "hit_x", "hit_y", "landing_x", "landing_y"            
]

def get_all_match_files():
    """获取目录下所有的比赛CSV文件"""
    return sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

def load_and_fix_data(file_name):
    """读取并标准化特定文件的数据"""
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    try:
        df = pd.read_csv(path)
        for c in STANDARD_COLUMNS:
            if c not in df.columns:
                df[c] = "Unknown" if 'player' in c else 0
        return df[STANDARD_COLUMNS]
    except Exception as e:
        st.error(f"读取 {file_name} 失败: {e}")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

def save_to_db(new_data_df, file_name, mode='a'):
    """保存数据到指定文件"""
    # 确保文件名以 .csv 结尾
    if not file_name.endswith('.csv'):
        file_name += '.csv'
        
    path = os.path.join(DATA_DIR, file_name)
    for col in STANDARD_COLUMNS:
        if col not in new_data_df.columns:
            new_data_df[col] = 0 if 'player' not in col else "Unknown"
    
    new_data_df = new_data_df[STANDARD_COLUMNS]
    
    if mode == 'w':
        new_data_df.to_csv(path, index=False)
    else:
        header = not os.path.exists(path)
        new_data_df.to_csv(path, mode='a', header=header, index=False)

# --- 保留模拟数据生成功能 ---
# --- 智能生成模拟数据功能 ---
def generate_mock_db():
    """生成完美适配六维雷达公式的模拟数据"""
    
    player_profiles = {
        "Axelsen": "Attack",       # 进攻极高
        "Momota": "Defense",       # 防守、稳定性极高
        "Ginting": "Speed",        # 进攻、技术较高
        "Shi Yuqi": "Balanced",    # 均衡
        "Lee Zii Jia": "Attack",   # 进攻极高，稳定性差
        "Christie": "Technique",   # 技术极高
        "Antonsen": "Technique",   # 技术极高
        "Kunlavut": "Defense",     # 防守极高
        "Kodai": "Stability",      # 稳定性极高
        "Lakshya Sen": "Balanced"  # 均衡
    }
    
    players = list(player_profiles.keys())
    rows = []
    
    # 增加到 800 回合，让各项统计指标的数据池更稳定
    for i in range(800):
        pA, pB = np.random.choice(players, 2, replace=False)
        style_A = player_profiles[pA]
        style_B = player_profiles[pB]
        
        # 决定赢家 (各 50% 概率)
        winner = "A" if np.random.random() < 0.5 else "B"
        w_player = pA if winner == "A" else pB
        l_player = pB if winner == "A" else pA
        w_style = style_A if winner == "A" else style_B
        l_style = style_B if winner == "A" else style_A
        
        score_a = np.random.randint(10, 22)
        score_b = np.random.randint(10, 22)
        pressure = np.random.randint(40, 95)
        
        shot_type = "其他"
        smasher = "None"
        
        # --- 核心修正 1：输家有没有杀球？(这直接决定赢家的【防守】得分) ---
        if w_style == "Defense":
            l_smash_prob = 0.65  # 防守型选手赢球，大概率是防死了对面的杀球
        elif l_style == "Attack":
            l_smash_prob = 0.5   # 进攻型选手输球，大概率是自己一顿猛扣没打死
        else:
            l_smash_prob = 0.15  # 其他情况偶尔出现防守反击

        if np.random.random() < l_smash_prob:
            smasher = "B" if winner == "A" else "A" # 设定输家为杀球方
        
        # --- 核心修正 2：赢家是怎么得分的？(这决定赢家的【进攻】、【技术】和输家的【稳定】) ---
        error_prob = 0.12
        if l_style == "Attack": error_prob = 0.25      # 进攻型易失误
        if l_style == "Stability": error_prob = 0.03   # 稳定型极少失误
        
        if np.random.random() < error_prob:
            shot_type = "失误"
            smasher = "None" # 对手失误送分，抹除原有的杀球状态
        else:
            if w_style == "Attack":
                if np.random.random() < 0.7:
                    shot_type = "杀球"
                    smasher = winner # 赢家主动杀球
                else:
                    shot_type = np.random.choice(["网前球", "平抽", "挑球"])
                    
            elif w_style in ["Technique", "Stability"]:
                if np.random.random() < 0.75:
                    shot_type = np.random.choice(["网前球", "吊球", "平抽"]) # 技术流得分
                else:
                    shot_type = "杀球"
                    smasher = winner
                    
            elif w_style == "Defense":
                shot_type = np.random.choice(["挑球", "平抽", "网前球", "杀球"])
                if shot_type == "杀球": smasher = winner
                
            else: # Balanced / Speed
                shot_type = np.random.choice(["杀球", "网前球", "吊球", "平抽"])
                if shot_type == "杀球": smasher = winner

        rows.append({
            "match_id": f"Mock_Match_{i//40 + 1}",
            "set_id": 1,
            "rally": (i % 40) + 1,
            "player_A": pA, "player_B": pB,
            "score_A": score_a, "score_B": score_b,
            "winner": winner, "type": shot_type,
            "pressure_score": pressure, "smasher": smasher,
            "hit_x": 0, "hit_y": 0, "landing_x": 0, "landing_y": 0
        })

    df_mock = pd.DataFrame(rows)[STANDARD_COLUMNS]
    save_to_db(df_mock, file_name="模拟比赛数据_Sample.csv", mode='w')

# --- 保留智能补全算法 ---
def get_player_profile(df, player):
    part = df[(df['player_A'] == player) | (df['player_B'] == player)]
    if len(part) == 0: return np.array([0.5, 0.2, 0.5]) 
    is_a = part['player_A'] == player
    wins = ((is_a) & (part['winner'] == 'A')) | ((~is_a) & (part['winner'] == 'B'))
    win_rate = wins.mean()
    smash_rate = len(part[part['type'] == '杀球']) / len(part)
    avg_pressure = part['pressure_score'].mean() / 100.0
    return np.array([win_rate, smash_rate, avg_pressure])

def smart_augment_data(main_df, upload_df, min_records=5):
    combined_df = pd.concat([main_df, upload_df], ignore_index=True)
    all_players_upload = pd.concat([upload_df['player_A'], upload_df['player_B']]).unique()
    augmented_rows = []
    experienced_players = []
    all_combined_players = pd.concat([combined_df['player_A'], combined_df['player_B']]).unique()
    for p in all_combined_players:
        if len(combined_df[(combined_df['player_A'] == p) | (combined_df['player_B'] == p)]) >= min_records:
            experienced_players.append(p)
    if not experienced_players: return upload_df
    experienced_profiles = {p: get_player_profile(combined_df, p) for p in experienced_players}
    for p in all_players_upload:
        p_records = combined_df[(combined_df['player_A'] == p) | (combined_df['player_B'] == p)]
        if 0 < len(p_records) < min_records:
            p_profile = get_player_profile(p_records, p)
            best_match = min(experienced_profiles.keys(), key=lambda x: np.linalg.norm(p_profile - experienced_profiles[x]))
            needed_count = min_records - len(p_records)
            donor_records = combined_df[(combined_df['player_A'] == best_match) | (combined_df['player_B'] == best_match)]
            sampled = donor_records.sample(n=min(needed_count, len(donor_records)), replace=True).copy()
            sampled['match_id'] = sampled['match_id'].apply(lambda x: f"{x}_aug")
            sampled.loc[sampled['player_A'] == best_match, 'player_A'] = p
            sampled.loc[sampled['player_B'] == best_match, 'player_B'] = p
            augmented_rows.append(sampled)
    if augmented_rows: return pd.concat([upload_df, pd.concat(augmented_rows)], ignore_index=True)
    return upload_df

# ==========================================
# 2. 页面主逻辑
# ==========================================
st.title("📝 数据引擎 | 底层赛事数据与平台管理")
st.markdown("支持手动单点录入、历史 CSV 批量导入、智能数据补全算法与模拟比赛生成器。")
st.divider()

all_files = get_all_match_files()
st.sidebar.info(f"📁 数据库文件夹内共有 {len(all_files)} 个比赛文件")

tab1, tab2, tab3 = st.tabs(["✍️ 手动录入", "📂 批量导入", "🛠️ 数据库管理"])

# --- Tab 1: 手动录入 (新增：录入文件夹/文件名选择) ---
with tab1:
    st.subheader("✍️ 记录比赛数据")
    
    # 获取现有文件列表
    all_files = get_all_match_files()
    
    col_a, col_b = st.columns([2, 3])
    with col_a:
        is_new_file = st.radio("录入方式", ["存入已有比赛文件", "创建新比赛文件"], horizontal=True)
        
        if is_new_file == "创建新比赛文件":
            target_file = st.text_input("请输入新比赛名称 (例如: 2024全英赛决赛)", value="My_New_Match")
            if not target_file.endswith('.csv'): target_file += '.csv'
        else:
            if not all_files:
                st.warning("文件夹为空，请选择“创建新比赛文件”")
                target_file = "match_data.csv"
            else:
                target_file = st.selectbox("请选择要存入的比赛文件：", all_files)

    st.markdown("---")
    
    # 加载目标文件的数据以推断下一球信息
    current_df = load_and_fix_data(target_file)
    
    if not current_df.empty:
        last_row = current_df.iloc[-1]
        def_match, def_pA, def_pB = last_row['match_id'], last_row['player_A'], last_row['player_B']
        try:
            def_set, def_rally = int(last_row['set_id']), int(last_row['rally']) + 1
            def_sA = int(last_row['score_A']) + (1 if last_row['winner'] == 'A' else 0)
            def_sB = int(last_row['score_B']) + (1 if last_row['winner'] == 'B' else 0)
        except: def_set, def_rally, def_sA, def_sB = 1, 1, 0, 0
    else:
        def_match, def_set, def_rally, def_pA, def_pB, def_sA, def_sB = "Match_01", 1, 1, "选手A", "选手B", 0, 0

    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)
        match_id = c1.text_input("Match ID", value=str(def_match))
        player_a = c2.text_input("Player A", value=str(def_pA))
        player_b = c3.text_input("Player B", value=str(def_pB))
        
        c4, c5 = st.columns(2)
        set_id = c4.number_input("Set", min_value=1, value=def_set)
        rally_id = c5.number_input("Rally", min_value=1, value=def_rally)
        
        cc1, cc2, cc3 = st.columns(3)
        score_a = cc1.number_input("Score A", 0, 30, def_sA)
        score_b = cc2.number_input("Score B", 0, 30, def_sB)
        winner = cc3.radio("Winner", ["A", "B"], horizontal=True)
        
        t1, t2, t3 = st.columns(3)
        shot_type = t1.selectbox("技术动作", ["杀球", "网前球", "平抽", "挑球", "失误", "其他"])
        pressure = t2.slider("压力值", 0, 100, 50)
        smash = t3.selectbox("杀球方", ["None", "A", "B"])
        
        if st.form_submit_button("💾 确认录入"):
            new_row = pd.DataFrame([{
                "match_id": match_id, "set_id": set_id, "rally": rally_id,
                "player_A": player_a, "player_B": player_b, "score_A": score_a, "score_B": score_b,
                "winner": winner, "type": shot_type, "pressure_score": pressure, "smasher": smash,
                "hit_x":0, "hit_y":0, "landing_x":0, "landing_y":0
            }])
            save_to_db(new_row, file_name=target_file, mode='a')
            st.success(f"成功存入文件: {target_file}")
            st.rerun()

# --- Tab 2: 批量导入 ---
with tab2:
    st.subheader("📂 批量导入比赛数据")  # 👈 新增：与其他两个 Tab 保持一致的小标题
    st.caption("上传、拖入或粘贴多个 CSV 文件。每个文件将作为一个独立的“比赛”存储。")
    # 增加 accept_multiple_files=True 以支持多文件及拖入/粘贴多个文件
    uploaded_files = st.file_uploader("上传 CSV 比赛数据", type=["csv"], accept_multiple_files=True)
    enable_augmentation = st.checkbox("✨ 开启智能补全", value=True)
    
    if uploaded_files:
        if st.button(f"🚀 确认批量导入并创建 {len(uploaded_files)} 个文件"):
            # 如果开启补全，提取一次历史数据即可，避免循环里重复高耗时读取
            if enable_augmentation:
                current_files = get_all_match_files()
                all_history = pd.concat([load_and_fix_data(f) for f in current_files]) if current_files else pd.DataFrame()
                
            # 循环处理选中的所有文件
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                df_upload = pd.read_csv(uploaded_file)
                
                if enable_augmentation:
                    df_upload = smart_augment_data(all_history, df_upload)
                    
                save_to_db(df_upload, file_name=file_name, mode='w')
                st.success(f"成功！比赛“{file_name}”已存档。")
                
            st.rerun()

# --- Tab 3: 数据库管理 ---
with tab3:
    st.subheader("🛠️ 数据库分级管理")
    
    col_mock, col_clear_all = st.columns([1, 1])
    with col_mock:
        if st.button("🎲 生成模拟数据文件", use_container_width=True):
            generate_mock_db()
            st.success("✅ 模拟数据已生成！")
            st.rerun()
    with col_clear_all:
        if st.button("🚨 清空所有比赛文件", type="secondary", use_container_width=True):
            for f in get_all_match_files():
                os.remove(os.path.join(DATA_DIR, f))
            st.warning("所有数据已销毁！")
            st.rerun()

    st.markdown("---")
    
    all_files = get_all_match_files()
    if not all_files:
        st.info("暂无比赛数据文件。")
    else:
        selected_match = st.selectbox("📂 选择要管理的比赛名称：", all_files)
        
        if selected_match:
            match_df = load_and_fix_data(selected_match)
            c_save, c_delete_one = st.columns([1, 1])
            
            edited_df = st.data_editor(
                match_df,
                num_rows="dynamic",
                use_container_width=True,
                height=450,
                key=f"editor_{selected_match}"
            )
            
            with c_save:
                if st.button(f"💾 保存对 {selected_match} 的修改", type="primary", use_container_width=True):
                    save_to_db(edited_df, file_name=selected_match, mode='w')
                    st.success("修改已保存！")
            
            with c_delete_one:
                if st.button(f"🗑️ 仅清空/删除该比赛: {selected_match}", use_container_width=True):
                    os.remove(os.path.join(DATA_DIR, selected_match))
                    st.warning(f"文件 {selected_match} 已移除。")
                    st.rerun()