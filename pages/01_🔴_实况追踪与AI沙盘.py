import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import requests
import json
import math
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="现场直播追踪与实时AI教练", page_icon="🏸", layout="wide")

# ==========================================
# 0. 数据库与文件配置
# ==========================================
DATA_DIR = 'match_database'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

STANDARD_COLUMNS = [
    "match_id", "set_id", "rally",      
    "player_A", "player_B",             
    "score_A", "score_B",               
    "winner", "type",                                             
    "pressure_score", "smasher",                                          
    "hit_x", "hit_y", "landing_x", "landing_y"            
]

def get_all_match_files():
    return sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

def load_and_fix_data(file_name):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    try:
        df = pd.read_csv(path)
        for c in STANDARD_COLUMNS:
            if c not in df.columns:
                df[c] = "Unknown" if 'player' in c else 0
        return df[STANDARD_COLUMNS]
    except:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

def save_to_db(new_data_df, file_name):
    if not file_name.endswith('.csv'): file_name += '.csv'
    path = os.path.join(DATA_DIR, file_name)
    for col in STANDARD_COLUMNS:
        if col not in new_data_df.columns:
            new_data_df[col] = 0 if 'player' not in col else "Unknown"
    new_data_df = new_data_df[STANDARD_COLUMNS]
    header = not os.path.exists(path)
    new_data_df.to_csv(path, mode='a', header=header, index=False)

# ==========================================
# 1. AI 推演、大模型接口与 2D 场地引擎
# ==========================================
def calculate_dynamic_prob(p_next, p_match_avg, current_score_diff):
    base_prob = 0.3 * p_next + 0.5 * p_match_avg + 0.2 * 0.5
    if current_score_diff < -5: base_prob *= 0.9 
    elif current_score_diff > 5: base_prob *= 1.1
    return min(max(base_prob, 0.01), 0.99)

def simulate_match_monte_carlo(sA, sB, start_prob, num_simulations=2000):
    a_wins = 0
    for _ in range(num_simulations):
        curr_a, curr_b = sA, sB
        curr_p = start_prob 
        while True:
            if curr_a >= 21 and curr_a >= curr_b + 2: a_wins += 1; break
            if curr_a == 30: a_wins += 1; break
            if curr_b >= 21 and curr_b >= curr_a + 2: break
            if curr_b == 30: break
            
            if random.random() < curr_p:
                curr_a += 1; curr_p = min(0.85, curr_p + 0.03)
            else:
                curr_b += 1; curr_p = max(0.15, curr_p - 0.03)
    return a_wins / num_simulations

def get_external_ai_coach_advice(match_summary, a_name, b_name, score_a, score_b):
    """调用智谱 GLM-4 大模型"""
    prompt_text = f"""
    你现在是一位世界顶级的羽毛球战术分析师和国家队教练。
    刚刚结束了一局比赛，比赛双方是 {a_name} 和 {b_name}。
    最终比分是 {score_a} : {score_b}。
    
    以下是本局比赛的客观数据统计：
    {match_summary}
    
    请根据以上数据，生成一份专业的赛后复盘报告。报告需要包含以下部分：
    1. 🔍 **本局胜负手分析**：谁掌控了节奏？赢在哪项技术或战术上？输的一方暴露了什么致命问题？（必须结合提供的数据）
    2. 📊 **关键数据解读**：指出数据中最亮眼或最拉胯的 1-2 个指标，并解释其在比赛中的影响。
    3. 💡 **下一局战术沙盘（针对重点指导）**：如果 {a_name} 想在下一局提高胜率，具体应该在【线路落点】、【发接发环节】或【节奏控制】上做出哪些改变？提出至少两点强有力的执行建议。
    
    语气要求：专业、犀利、直接、像真实的体育解说员。排版清晰，使用 Markdown 格式展现。
    """
    API_KEY = "94686e033f814ce1adcc7a3c3df02b94.PsXxCOiye1feUVgU"
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "glm-4-flash",
        "messages": [{"role": "system", "content": "你是一位资深的羽毛球国家队教练。"}, {"role": "user", "content": prompt_text}],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ 呼叫 AI 教练失败，请检查网络或 API 配置。错误信息: {e}"

def create_court_with_trajectory(strokes, width=320):
    """在内存中绘制羽毛球场并分别叠加A和B的连贯跑动轨迹"""
    height = int(width * (13.4 / 6.1))
    img = Image.new("RGB", (width, height), "#2D8A56")
    draw = ImageDraw.Draw(img)
    line_w = 2
    
    # 绘制基础场地线条
    draw.rectangle([10, 10, width-10, height-10], outline="white", width=line_w)
    draw.line([10, height//2, width-10, height//2], fill="white", width=4)
    draw.line([10, height//2 - 40, width-10, height//2 - 40], fill="white", width=line_w)
    draw.line([10, height//2 + 40, width-10, height//2 + 40], fill="white", width=line_w)
    draw.line([width//2, 10, width//2, height//2 - 40], fill="white", width=line_w)
    draw.line([width//2, height//2 + 40, width//2, height-10], fill="white", width=line_w)

    if len(strokes) > 0:
        # 1. 严格按顺序提取 A 和 B 的坐标
        a_coords = [(s[0], s[1]) for s in strokes if s[2] == "A"]
        b_coords = [(s[0], s[1]) for s in strokes if s[2] == "B"]

        # 2. 画A轨迹
        if len(a_coords) > 1:
           draw.line(a_coords, fill="#FFA07A", width=3)

        # 3. 画B轨迹
        if len(b_coords) > 1:
           draw.line(b_coords, fill="#87CEFA", width=3)

        # 4. 最后绘制落点圆圈（盖在线上）
        for i, p in enumerate(strokes):
            x, y, player = p
            color = "#FF4136" if player == "A" else "#0074D9"  
            r = 6 if i == len(strokes)-1 else 4 
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline="white")
            
    return img

def calculate_spatial_pressure(strokes, court_width_px=320):
    """计算单名球员的真实跑动距离，并转化为压制感"""
    if len(strokes) < 2: return 0, 0.0
    px_to_meter = 6.1 / (court_width_px - 20) 
    
    a_strokes = [s for s in strokes if s[2] == "A"]
    b_strokes = [s for s in strokes if s[2] == "B"]
    
    dist_A_px = sum(math.hypot(a_strokes[i+1][0] - a_strokes[i][0], a_strokes[i+1][1] - a_strokes[i][1]) for i in range(len(a_strokes)-1)) if len(a_strokes) > 1 else 0
    dist_B_px = sum(math.hypot(b_strokes[i+1][0] - b_strokes[i][0], b_strokes[i+1][1] - b_strokes[i][1]) for i in range(len(b_strokes)-1)) if len(b_strokes) > 1 else 0
    
    max_dist_px = max(dist_A_px, dist_B_px)
    real_distance_m = max_dist_px * px_to_meter
    pressure_score = (real_distance_m / 15.0) * 100
    return min(int(pressure_score), 100), real_distance_m

# ==========================================
# 2. 页面与全局状态初始化
# ==========================================
st.title("🏸 实况追踪 | 现场追踪与 AI 教练沙盘")
st.markdown("通过点选落点实时计算空间压制力，结合大语言模型与蒙特卡洛算法进行战术预演。")
st.divider() # 添加统一的分割线

if "live_strokes" not in st.session_state:
    st.session_state.live_strokes = []
if "live_last_click" not in st.session_state:
    st.session_state.live_last_click = None

with st.container():
    all_files = get_all_match_files()
    col_setup1, col_setup2, col_setup3 = st.columns([2, 2, 1])
    with col_setup1:
        match_mode = st.radio("录入目标", ["存入已有比赛", "新开一场比赛"], horizontal=True)
        target_file = st.text_input("输入新比赛名称", "Live_Match_01") + '.csv' if match_mode == "新开一场比赛" else st.selectbox("选择比赛文件", all_files) if all_files else "Live_Match_01.csv"
    with col_setup2:
        player_a = st.text_input("A 选手姓名", "选手 A")
        player_b = st.text_input("B 选手姓名", "选手 B")
    with col_setup3:
        st.write("")
        st.write("")
        base_exp_win = st.slider("A 赛前硬实力评估", 0.0, 1.0, 0.50)

st.markdown("---")

df_current = load_and_fix_data(target_file)
match_score_A, match_score_B = 0, 0

if not df_current.empty:
    last_row = df_current.iloc[-1]
    curr_set, curr_rally = int(last_row['set_id']), int(last_row['rally']) + 1
    curr_sA, curr_sB = int(last_row['score_A']), int(last_row['score_B'])
    
    for s in range(1, curr_set):
        past_set_data = df_current[df_current['set_id'] == s]
        if not past_set_data.empty:
            end_sA, end_sB = int(past_set_data.iloc[-1]['score_A']), int(past_set_data.iloc[-1]['score_B'])
            if end_sA > end_sB: match_score_A += 1
            elif end_sB > end_sA: match_score_B += 1
            
    current_match_data = df_current[df_current['set_id'] == curr_set]
    if len(current_match_data) > 0:
        current_smash_rate = len(current_match_data[current_match_data['smasher'] == 'A']) / len(current_match_data)
        current_spatial = current_match_data['pressure_score'].mean()
    else: current_smash_rate, current_spatial = 0.3, 50.0
else:
    curr_set, curr_rally, curr_sA, curr_sB = 1, 1, 0, 0
    current_smash_rate, current_spatial = 0.3, 50.0

is_set_over = (curr_sA >= 21 and curr_sA - curr_sB >= 2) or curr_sA == 30 or (curr_sB >= 21 and curr_sB - curr_sA >= 2) or curr_sB == 30

col_live, col_ai = st.columns([1, 1.2], gap="large")

# ================= 左侧：现场实况录入 (紧凑版) =================
with col_live:
    st.subheader(f"🔴 场边实况录入 (Set {curr_set} - Rally {curr_rally})")
    
    if is_set_over:
        set_winner_A = curr_sA > curr_sB
        final_match_A = match_score_A + (1 if set_winner_A else 0)
        final_match_B = match_score_B + (0 if set_winner_A else 1)
        is_match_over = final_match_A >= 2 or final_match_B >= 2
        
        if is_match_over:
            match_winner = player_a if final_match_A > final_match_B else player_b
            st.success(f"🏆 全场比赛结束！{match_winner} 以大比分 {final_match_A} : {final_match_B} 取得最终胜利！")
            st.balloons() 
            st.info("💡 请在页面最顶部的【录入目标】处，选择“新开一场比赛”来开启新的数据录入。")
            if st.button("🏁 结束本场监控", type="primary", use_container_width=True):
                st.rerun()
        else:
            st.success(f"⏳ 本局比赛已结束！当前比分 {curr_sA} : {curr_sB}")
            if st.button("进入下一局 (Set + 1)", type="primary", use_container_width=True):
                save_to_db(pd.DataFrame([{
                    "match_id": target_file.replace('.csv', ''), "set_id": curr_set + 1, "rally": 1, 
                    "player_A": player_a, "player_B": player_b, "score_A": 0, "score_B": 0, 
                    "winner": "Unknown", "type": "开局", "pressure_score": 50, "smasher": "None", 
                    "hit_x":0, "hit_y":0, "landing_x":0, "landing_y":0
                }]), target_file)
                st.rerun()
    else:
        st.markdown(f"""
            <div style='display: flex; justify-content: space-around; align-items: center; padding: 10px 5px; background-color: #f0f2f6; border-radius: 8px; border: 1px solid #e1e4e8;'>
                <div style='text-align: center; flex: 1;'>
                    <p style='margin: 0; color: #666; font-size: 12px; font-weight: bold;'>大比分 (SETS)</p>
                    <h3 style='margin: 0; color: #FF4B4B; letter-spacing: 2px;'>{match_score_A} : {match_score_B}</h3>
                </div>
                <div style='border-left: 2px dashed #ccc; height: 35px;'></div>
                <div style='text-align: center; flex: 1.5;'>
                    <p style='margin: 0; color: #888; font-size: 12px;'>第 {curr_set} 局 (POINTS)</p>
                    <h3 style='margin: 0; color: #1f77b4;'>{player_a} &nbsp; {curr_sA} : {curr_sB} &nbsp; {player_b}</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.write("---")

        rec_c1, rec_c2 = st.columns([1, 1.3])

        with rec_c1:
            st.write("**📍 1. 标记关键帧落点**")
            current_brush = st.radio("记录谁的落点？", ["A (红)", "B (蓝)"], horizontal=True, label_visibility="collapsed")
            player_label = "A" if "A" in current_brush else "B"

            court_width = 180 # 进一步稍微缩小一点保证不会错位
            court_img = create_court_with_trajectory(st.session_state.live_strokes, width=court_width)
            
            # --- 关键修改在这里，加上了 use_column_width=True ---
            click_value = streamlit_image_coordinates(court_img, key="live_court", use_column_width=True)
            # --------------------------------------------------
            
            if click_value is not None and click_value != st.session_state.live_last_click:
                st.session_state.live_strokes.append((click_value["x"], click_value["y"], player_label))
                st.session_state.live_last_click = click_value
                st.rerun()
                
            if st.button("↩️ 撤销落点", use_container_width=True):
                if st.session_state.live_strokes: st.session_state.live_strokes.pop()
                st.rerun()
            if st.button("🗑️ 清空轨迹", use_container_width=True):
                st.session_state.live_strokes = []
                st.rerun()

        with rec_c2:
            st.write("**📝 2. 录入结果**")
            calc_pressure, calc_dist = calculate_spatial_pressure(st.session_state.live_strokes, court_width_px=court_width)
            st.info(f"🏃 跑动: **{calc_dist:.1f}m** | 🔥 压制: **{calc_pressure}分**")

            winner = st.radio("得分方", ["A", "B"], horizontal=True)
            shot_type = st.selectbox("结束动作", ["杀球", "网前球", "平抽", "挑球", "吊球", "失误"])
            smash = st.selectbox("发起杀球", ["无", "A", "B"])
            
            st.write("") 
            if st.button("💾 记录并更新 AI", type="primary", use_container_width=True):
                new_sA = curr_sA + 1 if winner == "A" else curr_sA
                new_sB = curr_sB + 1 if winner == "B" else curr_sB
                
                h_x, h_y = st.session_state.live_strokes[0][:2] if st.session_state.live_strokes else (0, 0)
                l_x, l_y = st.session_state.live_strokes[-1][:2] if len(st.session_state.live_strokes) >= 2 else (h_x, h_y)
                
                new_row = pd.DataFrame([{
                    "match_id": target_file.replace('.csv', ''), "set_id": curr_set, "rally": curr_rally,
                    "player_A": player_a, "player_B": player_b, "score_A": new_sA, "score_B": new_sB,
                    "winner": winner, "type": shot_type, "pressure_score": calc_pressure, "smasher": "None" if smash == "无" else smash,
                    "hit_x": h_x, "hit_y": h_y, "landing_x": l_x, "landing_y": l_y
                }])
                save_to_db(new_row, target_file)
                st.session_state.live_strokes = []
                st.session_state.live_last_click = None
                st.rerun()

# ================= 右侧：实时 AI 教练与沙盘 =================
with col_ai:
    st.subheader("🤖 实时战术沙盘推演")
    
            
    score_diff = curr_sA - curr_sB
    base_p_next = 0.4 + (current_spatial/200) + (current_smash_rate * 0.1) 
    base_prob = calculate_dynamic_prob(base_p_next, base_exp_win, score_diff)
    base_win_rate = simulate_match_monte_carlo(curr_sA, curr_sB, base_prob)
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("A 实时预测胜率", f"{base_win_rate:.1%}")
    mc2.metric("A 当前杀球率", f"{current_smash_rate:.1%}")
    mc3.metric("A 空间压制均值", f"{current_spatial:.1f}")

    if not is_set_over:
        st.divider()
        st.markdown("##### 🎛️ AI 教练沙盘推演")
        st.caption("拖动滑块，预演下一球战术变化带来的胜率波动：")
        
        hypo_smash = st.slider("假设：A 杀球频率变为", 0.0, 1.0, float(current_smash_rate))
        hypo_spatial = st.slider("假设：A 场均压制力升至", 0.0, 100.0, float(current_spatial))
        
        hypo_p_next = 0.4 + (hypo_spatial/200) + (hypo_smash * 0.1)
        hypo_win_rate = simulate_match_monte_carlo(curr_sA, curr_sB, calculate_dynamic_prob(hypo_p_next, base_exp_win, score_diff))
        delta_rate = (hypo_win_rate - base_win_rate) * 100
        
        st.metric("调整后 A 预测胜率", f"{hypo_win_rate:.1%}", f"{delta_rate:+.1f}%")
        
        st.markdown("###### 🗣️ 教练战术解析")
        tactical_intent = []
        if hypo_smash > current_smash_rate + 0.1: tactical_intent.append("大幅**增加主动下压**")
        elif hypo_smash < current_smash_rate - 0.1: tactical_intent.append("刻意**收起锋芒**")
        if hypo_spatial > current_spatial + 15: tactical_intent.append("强调**大范围调动**")
            
        if not tactical_intent: st.caption("*保持当前比赛节奏，在稳定中寻找破绽。*")
        else: st.caption(f"*核心思路：{'，同时'.join(tactical_intent)}*")
            
        if len(df_current) >= 5:
            if delta_rate > 8: st.success(f"🔥 **强烈建议执行！** 此战术直击对手软肋。")
            elif delta_rate < -5: st.error("⚠️ **警告**：盲目强攻可能陷入反击陷阱，保持耐心。")
            elif score_diff < -3: st.warning("📉 **落后追分策略**：尝试网前调动，避免盲目重杀失误。")
            else: st.info("⚖️ **局势焦灼**：双方胜负在毫厘之间，重点看心态。")

# ==========================================
# 3. 局末 AI 赛后深度复盘
# ==========================================
if is_set_over:
    st.balloons()
    st.markdown("---")
    st.header(f"📈 局末战术沙盘：AI 教练深度复盘 (Set {curr_set})")
    
    set_df = df_current[df_current['set_id'] == curr_set]
    if len(set_df) > 0:
        a_wins, b_wins = set_df[set_df['winner'] == 'A'], set_df[set_df['winner'] == 'B']
        a_smash, b_smash = len(a_wins[a_wins['type'] == '杀球']), len(b_wins[b_wins['type'] == '杀球'])
        a_err, b_err = len(b_wins[b_wins['type'] == '失误']), len(a_wins[a_wins['type'] == '失误'])
        avg_pressure = set_df['pressure_score'].mean()
        
        data_summary = f"- 比赛总回合数：{len(set_df)}\n- 平均空间压制感：{avg_pressure:.1f}\n- {player_a} 得分分布：总分 {curr_sA}，杀球得分 {a_smash} 次，获送失误 {b_err} 次。\n- {player_b} 得分分布：总分 {curr_sB}，杀球得分 {b_smash} 次，获送失误 {a_err} 次。"
        
        rep_col1, rep_col2 = st.columns([1, 2.5])
        with rep_col1:
            st.markdown("### 📊 本局核心数据")
            st.metric(f"{player_a} 杀球得分", f"{a_smash} 分")
            st.metric(f"{player_a} 收到失误分", f"{b_err} 分")
            st.metric(f"全场平均压制强度", f"{avg_pressure:.1f}")
            
        with rep_col2:
            st.markdown("### 🤖 专家教练指导")
            with st.container(border=True):
                cache_key = f"live_ai_review_set_{curr_set}"
                if cache_key not in st.session_state:
                    with st.spinner("GLM-4 正在分析击球落点与战术克制关系..."):
                        st.session_state[cache_key] = get_external_ai_coach_advice(data_summary, player_a, player_b, curr_sA, curr_sB)
                st.markdown(st.session_state[cache_key])
