import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
from src.backend import ModelResearcher, ModelManager
from src.benchmarks import BenchmarkSuite

# --- Styling & Config ---
st.set_page_config(page_title="DeepBench: AI Researcher Workbench", layout="wide", page_icon="🧪")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    h1, h2, h3 { color: #00d4ff; }
    /* Metric Card Styling */
    .metric-card {
        background-color: #262730; border: 1px solid #41424C;
        border-radius: 8px; padding: 15px; margin-bottom: 10px;
        text-align: center;
    }
    .metric-val { font-size: 24px; font-weight: bold; }
    .metric-good { color: #00ff00; }
    .metric-avg { color: #ffff00; }
    .metric-bad { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'manager' not in st.session_state:
    st.session_state['manager'] = ModelManager(device="cuda" if torch.cuda.is_available() else "cpu")

# --- Sidebar ---
with st.sidebar:
    st.title("🧪 DeepBench")
    st.markdown("### Researcher Control Panel")
    task = st.selectbox("Domain", ["Language", "Vision"])
    arch = st.radio("Architecture", ["All", "Transformer", "RNN/RWKV"])
    st.markdown("---")
    st.info(f"Device: {st.session_state['manager'].device.upper()}")

# --- Tabs for Workflow ---
tab1, tab2, tab3 = st.tabs(["🔍 Model Discovery", "⚔️ Battle Arena (Compare)", "💬 Generation Playground"])

# ================= TAB 1: DISCOVERY =================
with tab1:
    researcher = ModelResearcher()
    col_search, col_res = st.columns([1, 4])
    
    with col_search:
        if st.button("Fetch Models", use_container_width=True):
            st.session_state['models'] = researcher.search_models(task_domain=task, architecture_type=arch)

    with col_res:
        if 'models' in st.session_state:
            st.dataframe(
                st.session_state['models'],
                column_config={
                    "downloads": st.column_config.ProgressColumn("Downloads", format="%d", min_value=0, max_value=1000000),
                },
                use_container_width=True, height=400
            )

# ================= TAB 2: BATTLE ARENA =================
with tab2:
    if 'models' in st.session_state:
        all_ids = st.session_state['models']['model_id'].tolist()
        
        c1, c2 = st.columns(2)
        with c1:
            model_a = st.selectbox("Select Model A (Champion)", all_ids, index=0)
        with c2:
            model_b = st.selectbox("Select Model B (Challenger)", all_ids, index=1 if len(all_ids)>1 else 0)

        # Benchmark Selection
        bench_opts = ["Perplexity", "MMLU", "GSM8K", "ARC-C", "ARC-E", "HellaSwag", "PIQA"]
        selected_bench = st.multiselect("Select Benchmarks to Run", bench_opts, default=["Perplexity", "MMLU"])
        
        if st.button("⚔️ FIGHT! (Run Comparison)"):
            
            # Layout for results
            col_a, col_mid, col_b = st.columns([1, 0.2, 1])
            
            results_a, results_b = {}, {}
            
            # --- RUN MODEL A ---
            with col_a:
                st.subheader(f"🔵 {model_a}")
                with st.spinner(f"Loading {model_a}..."):
                    succ, msg = st.session_state['manager'].load_model(model_a)
                
                if succ:
                    mod, tok = st.session_state['manager'].get_components(model_a)
                    suite = BenchmarkSuite(mod, tok)
                    for b in selected_bench:
                        res = suite.run_benchmark(b, simulation_mode=True)
                        results_a[b] = res
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size:14px; color:#aaa;'>{b}</div>
                            <div class='metric-val'>{res['score']:.2f} <span style='font-size:12px;'>{res['unit']}</span></div>
                            <div style='font-size:14px;'>{res['rating']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(msg)

            # --- RUN MODEL B ---
            with col_b:
                st.subheader(f"🔴 {model_b}")
                with st.spinner(f"Loading {model_b}..."):
                    succ, msg = st.session_state['manager'].load_model(model_b)
                
                if succ:
                    mod, tok = st.session_state['manager'].get_components(model_b)
                    suite = BenchmarkSuite(mod, tok)
                    for b in selected_bench:
                        res = suite.run_benchmark(b, simulation_mode=True)
                        results_b[b] = res
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size:14px; color:#aaa;'>{b}</div>
                            <div class='metric-val'>{res['score']:.2f} <span style='font-size:12px;'>{res['unit']}</span></div>
                            <div style='font-size:14px;'>{res['rating']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(msg)

            # --- RADAR CHART COMPARISON ---
            if results_a and results_b:
                st.markdown("### 🕸️ Architectural Capabilities Map")
                categories = list(results_a.keys())
                
                # Normalize PPL for chart (inverse)
                vals_a = [r['score'] if r['unit'] == "%" else (100-r['score']) for r in results_a.values()]
                vals_b = [r['score'] if r['unit'] == "%" else (100-r['score']) for r in results_b.values()]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=vals_a, theta=categories, fill='toself', name=model_a, line_color='#00d4ff'))
                fig.add_trace(go.Scatterpolar(r=vals_b, theta=categories, fill='toself', name=model_b, line_color='#ff0055'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Go to 'Discovery' tab and fetch models first.")

# ================= TAB 3: GENERATION PLAYGROUND =================
with tab3:
    st.subheader("💬 Qualitative Analysis (Chat)")
    
    # User Input
    user_prompt = st.text_area("Enter a prompt to test reasoning/generation:", value="Explain quantum entanglement to a 5-year old.")
    
    if st.button("Generate Responses"):
        if 'models' in st.session_state:
            # We assume models were selected in Tab 2, or we pick defaults
            # For simplicity, we use the variables from Tab 2 context if available, else pick top 2
            ids = st.session_state['models']['model_id'].tolist()[:2] # Fallback
            
            c1, c2 = st.columns(2)
            
            # Gen A
            with c1:
                st.markdown(f"**Model: {ids[0]}**")
                succ, _ = st.session_state['manager'].load_model(ids[0])
                if succ:
                    with st.spinner("Generating..."):
                        out = st.session_state['manager'].generate_text(ids[0], user_prompt)
                    st.info(out)
            
            # Gen B
            with c2:
                if len(ids) > 1:
                    st.markdown(f"**Model: {ids[1]}**")
                    succ, _ = st.session_state['manager'].load_model(ids[1])
                    if succ:
                        with st.spinner("Generating..."):
                            out = st.session_state['manager'].generate_text(ids[1], user_prompt)
                        st.success(out)
        else:
            st.error("Please search for models first.")