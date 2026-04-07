import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import json
import io
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

st.set_page_config(page_title="比赛复盘与势能", page_icon="📈", layout="wide")

# ==========================================
# 0. 弹窗函数定义 (保持原样)
# ==========================================
@st.dialog("🤖 AI 深度复盘报告", width="large")
def show_ai_dialog(prompt_text):
    with st.spinner("AI 教练正在疯狂分析数据中，请稍候..."):
        ai_result = call_llm_api(prompt_text)
        st.success("分析完成！")
        st.markdown(ai_result)

@st.dialog("📸 专业数据战报预览", width="large")
def show_report_dialog(p_A, p_B, final_score_A, final_score_B, s_a, s_b, extra_data):
    with st.spinner("正在合成高颜值战报..."):
        card = create_match_card(p_A, p_B, final_score_A, final_score_B, s_a, s_b, extra_data)
        
        # 进一步压缩中间的比例，让图片变得更小，完美适应屏幕高度
        c1, c2, c3 = st.columns([2, 1.5, 2])
        with c2:
            st.image(card, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        buf = io.BytesIO()
        card.save(buf, format='PNG')
        st.download_button("📥 下载高清战报图片", buf.getvalue(), f"Report_{p_A}_vs_{p_B}.png", "image/png", use_container_width=True)

# ==========================================
# 1. 核心配置与工具函数 (保持原样)
# ==========================================
DATA_DIR = 'match_database'

def get_all_match_files():
    if not os.path.exists(DATA_DIR):
        return []
    return sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

@st.cache_data
def load_data(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    
    # 兼容层
    if 'rally' not in df.columns:
        if 'rally_id' in df.columns:
            df['rally'] = df['rally_id']
    if 'score_A' not in df.columns:
        if 'roundscore_A' in df.columns:
            df['score_A'] = df['roundscore_A']
            df['score_B'] = df['roundscore_B']
        else:
            df['score_A'] = 0
            df['score_B'] = 0
    if 'winner' not in df.columns:
        if 'getpoint_player' in df.columns:
            df['winner'] = df['getpoint_player']
        elif 'winner_is_A' in df.columns:
            df['winner'] = df['winner_is_A'].apply(lambda x: 'A' if x == 1 else 'B')
        else:
            df['winner'] = 'Unknown'
    if 'type' not in df.columns:
        df['type'] = '未知'
    if 'player_A' not in df.columns:
        df['player_A'] = 'Player A'
    if 'player_B' not in df.columns:
        df['player_B'] = 'Player B'
    if 'match_id' not in df.columns:
        df['match_id'] = 'Default_Match'
    if 'set_id' not in df.columns:
        df['set_id'] = 1
    return df

def process_match_momentum(df, match_id, set_id):
    match_df = df[(df['match_id'] == match_id) & (df['set_id'] == set_id)].copy()
    if match_df.empty:
        return match_df
    
    if 'rally' not in match_df.columns or match_df['rally'].isnull().all():
        match_df['rally'] = range(1, len(match_df) + 1)
        
    match_df['rally'] = pd.to_numeric(match_df['rally'], errors='coerce').fillna(0)
    match_df = match_df.sort_values('rally').reset_index(drop=True)
    match_df['score_A'] = pd.to_numeric(match_df['score_A'], errors='coerce').fillna(0)
    match_df['score_B'] = pd.to_numeric(match_df['score_B'], errors='coerce').fillna(0)
    match_df['score_diff'] = match_df['score_A'] - match_df['score_B']
    
    streaks = []
    current_streak = 1
    last_winner = None
    for winner in match_df['winner']:
        if pd.notna(winner) and winner == last_winner and winner != 'Unknown':
            current_streak += 1
        else:
            current_streak = 1
        streaks.append(current_streak)
        last_winner = winner
    match_df['streak'] = streaks
    
    events = [None] * len(match_df)
    is_clutch = [False] * len(match_df)
    
    for idx in range(len(match_df)):
        row = match_df.iloc[idx]
        is_tight_game = abs(row['score_diff']) <= 2 or row['score_A'] >= 18 or row['score_B'] >= 18
        is_clutch[idx] = is_tight_game  
        
        event_texts = []
        
        # 1. 判断关键失误
        if str(row['type']) == '失误' and is_tight_game:
            loser = 'B' if row['winner'] == 'A' else 'A'
            event_texts.append(f"💔 {loser} 关键失误")
            
        # 2. 判断连得分 (保持之前修改：只在连击中断时标注最大数)
        if row['streak'] >= 4:
            is_end_of_streak = False
            # 如果是最后一个回合，必然是连击结束
            if idx == len(match_df) - 1:
                is_end_of_streak = True
            # 如果下一个回合的连击数变回 1，说明连击在当前回合终结
            elif match_df.iloc[idx + 1]['streak'] <= 1:
                is_end_of_streak = True
                
            if is_end_of_streak:
                event_texts.append(f"🔥 {row['winner']} 连得 {int(row['streak'])} 分")
                
        if event_texts:
            events[idx] = " | ".join(event_texts)
            
    match_df['is_clutch'] = is_clutch
    match_df['event'] = events
    return match_df

# ==========================================
# 2. AI 接口与绘图函数 (修改 draw_momentum_chart)
# ==========================================
def call_llm_api(prompt_text):
    API_KEY = "94686e033f814ce1adcc7a3c3df02b94.PsXxCOiye1feUVgU"
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "glm-4-flash",
        "messages": [
            {"role": "system", "content": "你是一位资深的羽毛球国家队教练，语言风格专业、犀利、一针见血。"},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ 呼叫 AI 教练失败，错误信息: {e}"

def draw_momentum_chart(match_df, player_a, player_b, height=400): # 新增 height 参数，默认 400
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=match_df['rally'],
        y=match_df['score_diff'],
        fill='tozeroy',
        mode='lines+markers',
        line=dict(color='gray', width=2),
        marker=dict(size=6, color='gray'),
        name='分差走势',
        hovertemplate="<b>回合: %{x}</b><br>分差: %{y}<br>比分: " + match_df['score_A'].astype(int).astype(str) + " - " + match_df['score_B'].astype(int).astype(str) + "<extra></extra>"
    ))
    
    if 'event' in match_df.columns:
        highlight_df = match_df[match_df['event'].notna()]
        if not highlight_df.empty:
            streaks_df = highlight_df[highlight_df['event'].str.contains('🔥')]
            errors_df = highlight_df[highlight_df['event'].str.contains('💔')]
            
            if not streaks_df.empty:
                fig.add_trace(go.Scatter(
                    x=streaks_df['rally'],
                    y=streaks_df['score_diff'],
                    mode='markers+text',
                    marker=dict(symbol='star', size=14, color='gold', line=dict(width=1, color='orange')),
                    text=streaks_df['event'],
                    textposition="top center",
                    textfont=dict(size=11), # 调小字体防止拥挤
                    name='进攻高潮',
                    hoverinfo='text'
                ))
            if not errors_df.empty:
                fig.add_trace(go.Scatter(
                    x=errors_df['rally'],
                    y=errors_df['score_diff'],
                    mode='markers+text',
                    marker=dict(symbol='x', size=12, color='red'),
                    text=errors_df['event'],
                    textposition="bottom center",
                    textfont=dict(size=11), # 调小字体防止拥挤
                    name='致命失误',
                    hoverinfo='text'
                ))
                
    max_diff = max(abs(match_df['score_diff'].max()), abs(match_df['score_diff'].min())) + 3
    fig.update_layout(
        title=f"📊 比赛势能波动: {player_a} vs {player_b}",
        xaxis_title="回合数 (Rally)",
        yaxis_title=f"← {player_b} 领先 | 分差 | {player_a} 领先 →",
        yaxis=dict(range=[-max_diff, max_diff], zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        xaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
        hovermode="x unified",
        height=height, # 使用参数
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ==========================================
# 3. 战报计算与生成函数 (保持原样)
# ==========================================
def calculate_single_player_stats(df, player_name):
    part = df[(df['player_A'] == player_name) | (df['player_B'] == player_name)].copy()
    if len(part) < 5: return None
    part['is_A'] = part['player_A'] == player_name
    part['i_won'] = ((part['is_A']) & (part['winner'] == 'A')) | ((~part['is_A']) & (part['winner'] == 'B'))
    wins = part[part['i_won']]; losses = part[~part['i_won']]
    if len(wins) == 0: return None
    offense = len(wins[wins['type'] == '杀球']) / len(wins) * 200
    tech = len(wins[wins['type'].isin(['网前球', '平抽', '吊球', '挑球'])]) / len(wins) * 200
    def_sit = part[((part['is_A']) & (part['smasher'] == 'B')) | ((~part['is_A']) & (part['smasher'] == 'A'))]
    defense = def_sit['i_won'].mean() * 200 if len(def_sit) > 0 else 60
    clutch = part[(abs(part['score_A'] - part['score_B']) <= 3) | (part['score_A'] >= 18) | (part['score_B'] >= 18)]
    mental = clutch['i_won'].mean() * 100 if len(clutch) > 0 else 60
    high_p = part[part['pressure_score'] > 60]
    physical = high_p['i_won'].mean() * 100 + 20 if len(high_p) > 0 else 60
    stability = 100 - (len(losses[losses['type'] == '失误']) / max(len(losses), 1)) * 100
    def clip(x): return int(np.clip(x, 40, 95))
    return {"进攻": clip(offense), "防守": clip(defense), "技术": clip(tech), "心理": clip(mental), "身体": clip(physical), "稳定性": clip(stability)}

def generate_radar_image(stats_dict, player_name, color='#3182CE'):
    categories = list(stats_dict.keys()); values = list(stats_dict.values())
    N = len(categories); angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]; angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True), facecolor='none')
    ax.set_facecolor('none')
    ax.plot(angles, values, 'o-', linewidth=2, color=color, markersize=4)
    ax.fill(angles, values, alpha=0.3, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100); ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([], color='gray') 
    ax.grid(color='white', linestyle='--', alpha=0.2)
    ax.spines['polar'].set_visible(False)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, transparent=True)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGBA")

def create_match_card(player_a, player_b, final_score_a, final_score_b, stats_a, stats_b, extra_data):
    width, height = 1000, 1500
    bg = Image.new("RGBA", (width, height), (15, 23, 42, 255))
    draw = ImageDraw.Draw(bg)
    
    for i in range(0, height, 80):
        draw.line([(0, i), (width, i)], fill=(30, 41, 59, 255), width=1)

    font_path = "simhei.ttf" 
    for f in ["msyhbd.ttc", "simhei.ttf", "PingFang.ttc"]:
        if os.path.exists(f): font_path = f; break
    
    def get_font(size): return ImageFont.truetype(font_path, size)
    
    draw.text((width//2, 100), "羽毛球赛事数据战报", fill="#F8FAFC", anchor="mm", font=get_font(60))
    draw.rectangle([width//2-150, 150, width//2+150, 155], fill="#3182CE")
    
    draw.text((250, 280), player_a, fill="#63B3ED", anchor="mm", font=get_font(50))
    draw.text((750, 280), player_b, fill="#F87171", anchor="mm", font=get_font(50))
    
    score_str = f"{final_score_a} - {final_score_b}"
    draw.text((width//2, 400), score_str, fill="#FFFFFF", anchor="mm", font=get_font(120))
    
    radar_a = generate_radar_image(stats_a, player_a, "#63B3ED").resize((400, 400))
    radar_b = generate_radar_image(stats_b, player_b, "#F87171").resize((400, 400))
    bg.paste(radar_a, (50, 520), radar_a)
    bg.paste(radar_b, (550, 520), radar_b)
    
    def draw_stat_row(x, y, stats, color):
        for i, (k, v) in enumerate(stats.items()):
            draw.text((x, y + i*50), f"{k}: {v}", fill=color, font=get_font(28))

    draw_stat_row(150, 950, stats_a, "#63B3ED")
    draw_stat_row(650, 950, stats_b, "#F87171")
    
    draw.rounded_rectangle([100, 1280, 900, 1420], radius=15, fill=(30, 41, 59, 255), outline="#475569", width=2)
    clutch_rate = f"关键分胜率: {player_a} {extra_data['clutch_a']:.0f}% | {player_b} {extra_data['clutch_b']:.0f}%"
    draw.text((width//2, 1330), clutch_rate, fill="#E2E8F0", anchor="mm", font=get_font(32))
    
    misc_info = f"杀球得分: {extra_data['smash_a']} vs {extra_data['smash_b']} | 全场最大领先: {extra_data['max_lead']} 分"
    draw.text((width//2, 1380), misc_info, fill="#94A3B8", anchor="mm", font=get_font(28))
    
    draw.text((width//2, height-30), "PRO BADMINTON ANALYSIS SYSTEM · POWERED BY AI", fill="#475569", anchor="mm", font=get_font(18))
    
    return bg

# ==========================================
# 4. 页面主逻辑交互 (重新设计布局)
# ==========================================
st.title("📈 智能复盘 | 赛后势能波动与数据战报")
st.markdown("抓取关键节点事件，一键生成高颜值多维能力雷达与专业级大模型复盘报告。")
st.divider()

all_files = get_all_match_files()

if not all_files:
    st.warning("⚠️ 数据库为空。请先在【数据录入】页面生成数据文件，或上传 CSV 数据。")
else:
    # 顶部选择器区域 (保持原样)
    col0, col1, col2 = st.columns(3)
    
    with col0:
        selected_file = st.selectbox("📁 选择数据文件", all_files)
        
    df = load_data(selected_file)

    if df.empty:
        st.info(f"文件 {selected_file} 暂无有效数据。")
    else:
        match_list = df['match_id'].dropna().unique()
        with col1:
            selected_match = st.selectbox("🎯 选择比赛 (Match ID)", match_list)
            
        set_list = df[df['match_id'] == selected_match]['set_id'].dropna().unique()
        with col2:
            selected_set = st.selectbox("局数 (Set ID)", set_list)
            
        st.markdown("---")
        
        match_df = process_match_momentum(df, selected_match, selected_set)
        
        if match_df.empty:
            st.info("该对局暂无有效回合数据。")
        else:
            p_A = match_df.iloc[0]['player_A']
            p_B = match_df.iloc[0]['player_B']
            
            # --- 提前计算所需数据 ---
            final_score_A = int(match_df.iloc[-1]['score_A'])
            final_score_B = int(match_df.iloc[-1]['score_B'])
            max_lead_A = match_df['score_diff'].max()
            max_lead_B = abs(match_df['score_diff'].min())
            a_max_streak = match_df[match_df['winner'] == 'A']['streak'].max()
            b_max_streak = match_df[match_df['winner'] == 'B']['streak'].max()
            
            streak_a_val = int(a_max_streak) if pd.notna(a_max_streak) else 0
            streak_b_val = int(b_max_streak) if pd.notna(b_max_streak) else 0

            # ==========================================
            # 优化布局：使用简单 2 主列，消除硬编码边距，使文本对齐自然
            # ==========================================
            col_chart, col_data_ai = st.columns([1.5, 1]) # 左侧图表，右侧复盘和统计
            
            # ---------------- 左侧：图表区域 ----------------
            with col_chart:
                st.subheader("🌊 势能走势图")
                st.caption("走势图在 0 轴上方表示 A 选手领先，下方表示 B 选手领先。")
                
                # 调用修改后的 draw_momentum_chart，并增加高度到 500
                fig = draw_momentum_chart(match_df, p_A, p_B, height=500) 
                st.plotly_chart(fig, use_container_width=True)
                
                # 移出战局数据盘点，使左侧更紧凑
                # st.subheader("📋 战局数据盘点")
                # ...

            # ---------------- 右侧：复盘和数据区域 ----------------
            with col_data_ai:
                # 1. AI 智能教练复盘 (移到这里)
                st.subheader("🤖 AI 智能教练复盘")
                st.caption("一键召唤国家队级别AI教练，挖掘比赛数据背后的技战术深意。")
                
                events_text = "无关键失误或高光时刻。"
                if 'event' in match_df.columns:
                    events_df = match_df[match_df['event'].notna()]
                    if not events_df.empty:
                        events_list = []
                        for _, r in events_df.iterrows():
                            events_list.append(f"第{int(r['rally'])}回合 (比分 {int(r['score_A'])}:{int(r['score_B'])}): {r['event']}")
                        events_text = "\n".join(events_list)
                
                prompt = f"""
                请根据以下羽毛球比赛数据，为这场比赛做深度复盘，并给出技术改进建议。
                
                【比赛数据】
                - 对阵双方：{p_A} vs {p_B}
                - 最终比分：{final_score_A} : {final_score_B}
                - 最大领先：{max(max_lead_A, max_lead_B)}分 ({'A' if max_lead_A > max_lead_B else 'B'} 领先)
                - 最长连得分：{p_A} 连得 {streak_a_val} 分，{p_B} 连得 {streak_b_val} 分。
                - 关键比赛节点日志：
                {events_text}
                """
                
                if st.button("💡 召唤 AI 教练进行深度复盘", type="primary", use_container_width=True):
                    show_ai_dialog(prompt)
                    
                with st.expander("🔍 查看详细关键事件日志"):
                    if 'event' in match_df.columns:
                        events_only = match_df[match_df['event'].notna()][['rally', 'score_A', 'score_B', 'event']].copy()
                        events_only['score_A'] = events_only['score_A'].astype(int)
                        events_only['score_B'] = events_only['score_B'].astype(int)
                        events_only.columns = ['回合数', f'{p_A} 得分', f'{p_B} 得分', '关键事件']
                        st.dataframe(events_only, hide_index=True, use_container_width=True)

                # 增加一个小间隔
                st.markdown("<br>", unsafe_allow_html=True)

                # 2. 战局数据盘点 (移到这里，放在 AI 复盘之后)
                st.subheader("📋 战局数据盘点")
                # 使用 Markdown 展示更饱满，移出之前的 subheader
                st.markdown(f"""
- **🎯 最终比分:** `{final_score_A} : {final_score_B}`
- **⚖️ 最大分差:** `{max(max_lead_A, max_lead_B)} 分` ({'A' if max_lead_A > max_lead_B else 'B'} 领先)
- **🔥 {p_A} 最长连得分:** `{streak_a_val} 分`
- **🔥 {p_B} 最长连得分:** `{streak_b_val} 分`
""")

                st.divider()

                # 3. 战报海报生成 (保持原样)
                st.subheader("📸 战报海报生成")
                st.caption("生成包含六维雷达图的高颜值数据长图，适合复盘与分享。")
                if st.button("🎨 渲染专业数据战报", type="primary", use_container_width=True):
                    s_a = calculate_single_player_stats(df, p_A)
                    s_b = calculate_single_player_stats(df, p_B)
                    if s_a and s_b:
                        extra = {
                            'smash_a': match_df[(match_df['winner']=='A') & (match_df['type']=='杀球')].shape[0],
                            'smash_b': match_df[(match_df['winner']=='B') & (match_df['type']=='杀球')].shape[0],
                            'max_lead': max(match_df['score_diff'].max(), abs(match_df['score_diff'].min())),
                            'clutch_a': (match_df[(match_df['is_clutch']) & (match_df['winner']=='A')].shape[0] / max(match_df[match_df['is_clutch']].shape[0], 1)) * 100,
                            'clutch_b': (match_df[(match_df['is_clutch']) & (match_df['winner']=='B')].shape[0] / max(match_df[match_df['is_clutch']].shape[0], 1)) * 100
                        }
                        show_report_dialog(p_A, p_B, final_score_A, final_score_B, s_a, s_b, extra)
                    else:
                        st.error("数据不足，无法生成雷达图（每位选手至少需要 5 回合记录才能计算六维能力）。")