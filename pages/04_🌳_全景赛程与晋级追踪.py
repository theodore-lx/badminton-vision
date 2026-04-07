import streamlit as st
import pandas as pd
import graphviz
import os
import re

# ==========================================
# 页面与样式设置
# ==========================================
st.set_page_config(page_title="🏆 赛事晋级树", page_icon="🌳", layout="wide")

st.markdown("""
<style>
    /* 允许横向滚动，确保固定原生大小时可以拖动查看，解除最大宽度限制 */
    .stGraphvizChart > div { 
        overflow-x: auto; 
    }
    .stGraphvizChart svg { 
        max-width: none !important; 
        height: auto !important; 
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 数据加载与解析 (融合逻辑 & 导入数据分段)
# ==========================================
@st.cache_data(ttl=60)
def load_all_match_data():
    all_data = []
    
    if os.path.exists('match_data_tree.csv'):
        df_tree = pd.read_csv('match_data_tree.csv')
        df_tree['match_id'] = df_tree['match_id'].replace({
            'A组': '北邮A组',
            'B组': '北邮B组'
        }, regex=True)
        all_data.append(df_tree)
    
    if os.path.exists('match_data.csv'):
        df_manual = pd.read_csv('match_data.csv')
        if not df_manual.empty:
            # === 将导入的超长比赛进行分段切分 (每 15 球为一个独立树) ===
            def generate_chunk_id(row):
                try:
                    r = int(row['rally'])
                except:
                    r = 1
                start = ((r - 1) // 15) * 15 + 1
                end = start + 14
                m_id = str(row['match_id'])
                s_id = str(row['set_id'])
                return f"用户导入-{m_id}-第{s_id}局 ({start}-{end}球)"

            manual_tree = pd.DataFrame()
            manual_tree['match_id'] = df_manual.apply(generate_chunk_id, axis=1)
            manual_tree['encounter_id'] = range(1000, 1000 + len(df_manual))
            manual_tree['轮次'] = "第" + df_manual['set_id'].astype(str) + "局-R" + df_manual['rally'].astype(str)
            manual_tree['p1'] = df_manual['player_A']
            manual_tree['p2'] = df_manual['player_B']
            manual_tree['s1'] = df_manual['score_A']
            manual_tree['s2'] = df_manual['score_B']
            manual_tree['赢家'] = df_manual.apply(lambda r: r['player_A'] if r['winner'] == 'A' else r['player_B'], axis=1)
            all_data.append(manual_tree)

    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['encounter_id'] = pd.to_numeric(combined['encounter_id'], errors='coerce').fillna(0).astype(int)
    combined['base_group'] = combined['match_id'].apply(lambda x: str(x).split('(')[0].strip())
    return combined

df = load_all_match_data()

if df is None:
    st.error("❌ 未发现数据！请先在【数据录入】页面导入文件或确保 match_data_tree.csv 存在。")
    st.stop()

# ==========================================
# 侧边栏设置
# ==========================================
with st.sidebar:
    st.image("https://p3-pc-sign.douyinpic.com/tos-cn-i-0813c000-ce/ok9AJEBKAeAImA0osHeFEdrMEEMfPAGw8wDdAF~tplv-dy-aweme-images:q75.webp?biz_tag=aweme_images&from=327834062&lk3s=138a59ce&s=PackSourceEnum_SEARCH&sc=image&se=false&x-expires=1775437200&x-signature=It5dIwRqkFrzihlV6PGcQclxeYs%3D", use_container_width=True)
    st.markdown("### ⚙️ 赛事树控制台")
    
    base_groups = sorted(df['base_group'].unique())
    selected_base = st.selectbox("🎯 选择赛区大类:", base_groups, help="包含系统内置赛程与用户实时录入的同步数据")
    
    st.divider()
    st.info("数据会自动与录入页面同步。如果未看到最新数据，可手动刷新。")
    if st.button("♻️ 强制刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

df_base = df[df['base_group'] == selected_base].copy()

# ==========================================
# 页面主标题区域
# ==========================================
# 注意：这里保留了你的 selected_base 变量
st.title(f"🌳 赛事追踪 | 全景赛程与晋级树 ({selected_base})")
if "用户导入" in selected_base:
    st.markdown("✨ 当前视图：来自底层数据库的实时同步记录（已自动拆分为多个局内对战树）")
else:
    st.markdown("✨ 当前视图：官方标准赛程晋级路线图")
st.divider()

# ==========================================
# 辅助函数
# ==========================================
def get_sort_key(mid):
    nums = [int(n) for n in re.findall(r'\d+', mid)]
    if '主' in mid: 
        return [0] + nums
    elif '用户导入' in mid: 
        return [999] + nums
    else: 
        return [1] + nums

sub_match_ids = sorted(df_base['match_id'].unique(), key=get_sort_key)

def format_medal_label(rank, name, is_user_import=False):
    if is_user_import:
        colors = {
            1: ("#FFD700", "#FFFACD", "#B8860B", "🎉 胜出"),
            2: ("#C0C0C0", "#F5F5F5", "#708090", "🤝 完成")
        }
    else:
        colors = {
            1: ("#FFD700", "#FFFACD", "#B8860B", "🥇 冠军"),
            2: ("#C0C0C0", "#F5F5F5", "#708090", "🥈 亚军"),
            3: ("#CD7F32", "#FFFAF0", "#D2691E", "🥉 季军")
        }
    border, bg, font, text = colors.get(rank, ("#DDDDDD", "#F9F9F9", "#555555", f"🏅 第 {rank} 名"))
    
    return f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">
        <TR><TD BGCOLOR="{bg}" ALIGN="CENTER"><FONT COLOR="{font}" POINT-SIZE="12"><B>{text} : {name}</B></FONT></TD></TR>
    </TABLE>>'''

def get_color(rank):
    if rank == 1: return "#FFD700"
    if rank == 2: return "#C0C0C0"
    if rank == 3: return "#CD7F32"
    return "#A0A0A0"

def truncate_str(text, max_len=10):
    text_str = str(text)
    return text_str[:max_len-1] + ".." if len(text_str) > max_len else text_str

# ==========================================
# 核心业务逻辑：双轨制隔离
# ==========================================
player_final_info = {}

if "用户导入" not in selected_base:
    # --- 轨一：官方赛程（100% 恢复你的原版全局计算逻辑） ---
    players = set(df_base['p1'].dropna().unique()) | set(df_base['p2'].dropna().unique())
    players = {p for p in players if p and p not in ["轮空", "待定", "无"] and not str(p).startswith("未知ID")}
    
    for p in players:
        p_matches = df_base[(df_base['p1'] == p) | (df_base['p2'] == p)].sort_values('encounter_id')
        if not p_matches.empty:
            last_match = p_matches.iloc[-1]
            last_eid = str(last_match['encounter_id'])
            is_winner = (last_match['赢家'] == p)
            stage = str(last_match['轮次'])
            mid = str(last_match['match_id'])
            
            rank = None
            if "决赛" in stage and not any(x in stage for x in ["半", "四分", "8进4", "排位"]):
                rank = 1 if is_winner else 2
            elif "季军" in stage or "3-4" in stage:
                rank = 3 if is_winner else 4
            else:
                m = re.search(r'(\d+)-(\d+)名', stage)
                if not m:
                    m = re.search(r'(\d+)-(\d+)名', mid)
                    
                if m:
                    r1 = int(m.group(1))
                    r2 = int(m.group(2))
                    
                    if r2 - r1 == 2: 
                        mid_matches = df_base[df_base['match_id'] == mid]
                        wins = len(mid_matches[mid_matches['赢家'] == p])
                        if wins >= 2:
                            rank = r1
                        elif wins == 1:
                            rank = r1 + 1
                        else:
                            rank = r2
                    else:
                        rank = r1 if is_winner else r2
            
            player_final_info[p] = {
                'last_eid': last_eid,
                'rank': rank
            }
else:
    # --- 轨二：用户导入（切片化的局部计算逻辑） ---
    for mid in sub_match_ids:
        df_mid = df_base[df_base['match_id'] == mid].sort_values('encounter_id')
        players_mid = set(df_mid['p1'].dropna().unique()) | set(df_mid['p2'].dropna().unique())
        players_mid = {p for p in players_mid if p and p not in ["轮空", "待定", "无"] and not str(p).startswith("未知ID")}
        
        for p in players_mid:
            p_matches = df_mid[(df_mid['p1'] == p) | (df_mid['p2'] == p)]
            if not p_matches.empty:
                last_match = p_matches.iloc[-1]
                last_eid = str(last_match['encounter_id'])
                is_winner = (last_match['赢家'] == p)
                
                rank = 1 if is_winner else 2
                
                player_final_info[(mid, p)] = {
                    'last_eid': last_eid,
                    'rank': rank
                }

# ==========================================
# 循环渲染树状图
# ==========================================
for mid in sub_match_ids:
    with st.container():
        is_user_import = "用户导入" in mid
        
        # --- UI 布局优化：如果是用户导入数据，在右侧添加无损缩放滑块 ---
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"#### 📍 赛程子版块：`{mid}`")
        
        zoom_level = 1.0
        if is_user_import:
            with col2:
                # 修改点：将最小值设为 0.0，并在文案中提示用户
                zoom_level = st.slider("🔍 缩放比赛树 (0.0为自适应满屏)", min_value=0.0, max_value=3.0, value=1.0, step=0.1, key=f"zoom_{mid}")
        
        df_view = df_base[df_base['match_id'] == mid].sort_values('encounter_id').copy()

        def ensure_winner_on_top(row):
            should_swap = False
            try:
                s1_val = float(row['s1']) if pd.notna(row['s1']) else 0
                s2_val = float(row['s2']) if pd.notna(row['s2']) else 0
                if s2_val > s1_val:
                    should_swap = True
            except:
                pass
            
            if not should_swap and str(row['赢家']) == str(row['p2']) and pd.notna(row['p2']):
                should_swap = True
                
            if should_swap:
                new_row = row.copy()
                new_row['p1'], new_row['p2'] = row['p2'], row['p1']
                new_row['s1'], new_row['s2'] = row['s2'], row['s1']
                return new_row
            return row

        df_view = df_view.apply(ensure_winner_on_top, axis=1)
        
        dot = graphviz.Digraph(comment=mid)
        dot.attr(rankdir='LR', splines='polyline', ranksep='2.2', nodesep='0.8')
        dot.attr('node', shape='none', fontname='Microsoft YaHei', margin='0')
        dot.attr('edge', color='#999999', penwidth='1.5', arrowsize='0.7')

        stages = df_view['轮次'].unique()
        for stage in stages:
            stage_df = df_view[df_view['轮次'] == stage]
            
            with dot.subgraph() as s:
                s.attr(rank='same') 
                
                for _, row in stage_df.iterrows():
                    eid = str(row['encounter_id'])
                    p1, p2 = row['p1'], row['p2']
                    try: s1, s2 = int(float(row['s1'])), int(float(row['s2']))
                    except: s1, s2 = 0, 0

                    bg1 = "#FFF3CD" if s1 >= s2 and p1 not in ["待定", "轮空"] else "#FFFFFF"
                    bg2 = "#FFFFFF"
                    c1 = "#D35400" if s1 >= s2 else "#333333"
                    c2 = "#333333"

                    if is_user_import:
                        p1_disp = truncate_str(p1, 12)
                        p2_disp = truncate_str(p2, 12)
                        stage_disp = truncate_str(stage, 18)
                        
                        label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" COLOR="#CCCCCC" WIDTH="220" FIXEDSIZE="TRUE">
                            <TR><TD COLSPAN="2" BGCOLOR="#E9ECEF" HEIGHT="26" WIDTH="220" FIXEDSIZE="TRUE"><FONT POINT-SIZE="11" COLOR="#555555">{stage_disp}</FONT></TD></TR>
                            <TR>
                                <TD PORT="p1_name" BGCOLOR="{bg1}" ALIGN="LEFT" WIDTH="180" HEIGHT="28" FIXEDSIZE="TRUE"><FONT COLOR="{c1}"><B>{p1_disp}</B></FONT></TD>
                                <TD PORT="p1_score" BGCOLOR="{bg1}" ALIGN="CENTER" WIDTH="40" HEIGHT="28" FIXEDSIZE="TRUE"><FONT COLOR="{c1}"><B>{s1}</B></FONT></TD>
                            </TR>
                            <TR>
                                <TD PORT="p2_name" BGCOLOR="{bg2}" ALIGN="LEFT" WIDTH="180" HEIGHT="28" FIXEDSIZE="TRUE"><FONT COLOR="{c2}"><B>{p2_disp}</B></FONT></TD>
                                <TD PORT="p2_score" BGCOLOR="{bg2}" ALIGN="CENTER" WIDTH="40" HEIGHT="28" FIXEDSIZE="TRUE"><FONT COLOR="{c2}"><B>{s2}</B></FONT></TD>
                            </TR>
                        </TABLE>>'''
                    else:
                        label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3" COLOR="#CCCCCC" WIDTH="160">
                            <TR><TD COLSPAN="2" BGCOLOR="#E9ECEF" HEIGHT="20"><FONT POINT-SIZE="10" COLOR="#555555">{stage}</FONT></TD></TR>
                            <TR>
                                <TD PORT="p1_name" BGCOLOR="{bg1}" ALIGN="LEFT" WIDTH="130" HEIGHT="24"><FONT COLOR="{c1}"><B>{p1}</B></FONT></TD>
                                <TD PORT="p1_score" BGCOLOR="{bg1}" ALIGN="CENTER" WIDTH="30" HEIGHT="24"><FONT COLOR="{c1}"><B>{s1}</B></FONT></TD>
                            </TR>
                            <TR>
                                <TD PORT="p2_name" BGCOLOR="{bg2}" ALIGN="LEFT" WIDTH="130" HEIGHT="24"><FONT COLOR="{c2}"><B>{p2}</B></FONT></TD>
                                <TD PORT="p2_score" BGCOLOR="{bg2}" ALIGN="CENTER" WIDTH="30" HEIGHT="24"><FONT COLOR="{c2}"><B>{s2}</B></FONT></TD>
                            </TR>
                        </TABLE>>'''
                        
                    s.node(eid, label=label)

        if "17-19" not in str(mid):
            for _, curr_row in df_view.iterrows():
                curr_eid = str(curr_row['encounter_id'])
                for player_port_prefix, p in [("p1", curr_row['p1']), ("p2", curr_row['p2'])]:
                    if pd.notna(p) and p not in ["轮空", "待定", "无"] and not str(p).startswith("未知ID"):
                        prev_matches = df_view[
                            (df_view['encounter_id'] < curr_row['encounter_id']) & 
                            ((df_view['p1'] == p) | (df_view['p2'] == p))
                        ]
                        if not prev_matches.empty:
                            prev_match = prev_matches.iloc[-1]
                            prev_eid = str(prev_match['encounter_id'])
                            
                            prev_port = "p1_score" if prev_match['p1'] == p else "p2_score"
                            curr_port = f"{player_port_prefix}_name"
                            
                            dot.edge(f"{prev_eid}:{prev_port}", f"{curr_eid}:{curr_port}", color="#A0A0A0")

        terminal_nodes = []
        with dot.subgraph() as s:
            s.attr(rank='sink')
            
            for _, row in df_view.iterrows():
                eid = str(row['encounter_id'])
                for p in [row['p1'], row['p2']]:
                    lookup_key = (mid, p) if is_user_import else p
                    
                    if lookup_key in player_final_info and player_final_info[lookup_key]['last_eid'] == eid:
                        rank = player_final_info[lookup_key]['rank']
                        if rank is not None:
                            node_id = f"rank_{rank}_{eid}_{abs(hash(p))}"
                            s.node(node_id, label=format_medal_label(rank, p, is_user_import=is_user_import))
                            
                            prev_port = "p1_score" if row['p1'] == p else "p2_score"
                            dot.edge(f"{eid}:{prev_port}", f"{node_id}", color=get_color(rank), penwidth="1.5")
                            terminal_nodes.append((rank, node_id))

            terminal_nodes.sort(key=lambda x: x[0])
            for i in range(len(terminal_nodes) - 1):
                s.edge(terminal_nodes[i][1], terminal_nodes[i+1][1], style='invis')

        # --- 核心：双轨制渲染层 ---
        if is_user_import:
            try:
                svg_str = dot.pipe(format='svg').decode('utf-8')
                
                # 修改点：如果拉到 0.0，则将 SVG 设置为 100% 宽度，自适应容器满屏显示；否则按比例计算宽高。
                if zoom_level == 0.0:
                    svg_scaled = re.sub(
                        r'<svg\s+width="([0-9.]+)([a-zA-Z]*)"\s+height="([0-9.]+)([a-zA-Z]*)"', 
                        r'<svg width="100%" height="auto"', 
                        svg_str
                    )
                else:
                    def scale_svg(match):
                        w = float(match.group(1)) * zoom_level
                        unit_w = match.group(2)
                        h = float(match.group(3)) * zoom_level
                        unit_h = match.group(4)
                        return f'<svg width="{w}{unit_w}" height="{h}{unit_h}"'
                    
                    svg_scaled = re.sub(r'<svg\s+width="([0-9.]+)([a-zA-Z]*)"\s+height="([0-9.]+)([a-zA-Z]*)"', scale_svg, svg_str)
                
                st.write(f'<div style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #f0f2f6; border-radius: 8px; padding: 10px;">{svg_scaled}</div>', unsafe_allow_html=True)
            except Exception as e:
                # 异常容错机制：当解析失败时提供自适应后备渲染方案
                st.graphviz_chart(dot, use_container_width=True if zoom_level == 0.0 else False)
        else:
            # 轨一：北邮AB组完全使用原版方式渲染（保持不变）
            st.graphviz_chart(dot, use_container_width=False)

        st.write("<br>", unsafe_allow_html=True)