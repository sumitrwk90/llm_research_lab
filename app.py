import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
from src.backend import ModelResearcher, ModelManager
from src.benchmarks import BenchmarkSuite
from src.model_diagnostics import ModelDiagnostics

# --- Styling & Config ---
st.set_page_config(page_title="DeepBench: AI Researcher Workbench", layout="wide", page_icon="🧪")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    h1, h2, h3 { color: #00d4ff; }
    .metric-card {
        background-color: #262730; border: 1px solid #41424C;
        border-radius: 8px; padding: 15px; margin-bottom: 10px;
        text-align: center;
    }
    .metric-val { font-size: 24px; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 5px; }
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
    st.caption("v3.0 Full Suite | Diagnostics Active")

# --- Tabs ---
tab_names = ["🔍 Discovery", "⚔️ Battle Arena", "💬 Playground", "💾 Hardware Forecast", "🩻 Model X-Ray"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)

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
                column_config={"downloads": st.column_config.ProgressColumn("Downloads", format="%d", min_value=0, max_value=1000000)},
                use_container_width=True
            )

# ================= TAB 2: BATTLE ARENA (Updated with Quantization) =================
with tab2:
    if 'models' in st.session_state:
        all_ids = st.session_state['models']['model_id'].tolist()
        select_options = ["None"] + all_ids
        
        c1, c2 = st.columns(2)
        
        # --- CHAMPION SELECTION ---
        with c1:
            st.subheader("Champion (Model A)")
            model_a = st.selectbox("Select Model A", select_options, index=1 if len(all_ids)>0 else 0)
            quant_a = st.selectbox("Quantization A", ["None (FP16)", "8-bit (Int8)"], key="q_a")
            
        # --- CHALLENGER SELECTION ---
        with c2:
            st.subheader("Challenger (Model B)")
            model_b = st.selectbox("Select Model B", select_options, index=0)
            quant_b = st.selectbox("Quantization B", ["None (FP16)", "8-bit (Int8)"], key="q_b")

        bench_opts = ["Perplexity", "MMLU", "GSM8K", "ARC-C", "ARC-E", "HellaSwag", "PIQA"]
        selected_bench = st.multiselect("Benchmarks", bench_opts, default=["Perplexity", "MMLU"])
        
        if st.button("⚔️ Run Comparison"):
            col_a, col_mid, col_b = st.columns([1, 0.1, 1])
            results_a, results_b = {}, {}
            
            # Map friendly names to backend keys
            q_map_a = "8-bit" if "8-bit" in quant_a else "None"
            q_map_b = "8-bit" if "8-bit" in quant_b else "None"

            # --- PROCESS MODEL A ---
            with col_a:
                if model_a != "None":
                    st.write(f"**Loading {model_a} ({q_map_a})...**")
                    succ, msg = st.session_state['manager'].load_model(model_a, quantization=q_map_a)
                    
                    if succ:
                        mod, tok = st.session_state['manager'].get_components(model_a, quantization=q_map_a)
                        suite = BenchmarkSuite(mod, tok, model_id=f"{model_a}_{q_map_a}")
                        
                        for b in selected_bench:
                            res = suite.run_benchmark(b, simulation_mode=True)
                            results_a[b] = res
                            st.markdown(f"""<div class='metric-card'><div style='color:#aaa;'>{b}</div><div class='metric-val'>{res['score']:.2f}</div><div>{res['rating']}</div></div>""", unsafe_allow_html=True)
                    else:
                        st.error(f"Failed: {msg}")

            # --- PROCESS MODEL B ---
            with col_b:
                if model_b != "None":
                    st.write(f"**Loading {model_b} ({q_map_b})...**")
                    succ, msg = st.session_state['manager'].load_model(model_b, quantization=q_map_b)
                    
                    if succ:
                        mod, tok = st.session_state['manager'].get_components(model_b, quantization=q_map_b)
                        suite = BenchmarkSuite(mod, tok, model_id=f"{model_b}_{q_map_b}")
                        
                        for b in selected_bench:
                            res = suite.run_benchmark(b, simulation_mode=True)
                            results_b[b] = res
                            st.markdown(f"""<div class='metric-card'><div style='color:#aaa;'>{b}</div><div class='metric-val'>{res['score']:.2f}</div><div>{res['rating']}</div></div>""", unsafe_allow_html=True)
                    else:
                        st.error(f"Failed: {msg}")
            
            # --- CHART ---
            if results_a and results_b and model_a != "None" and model_b != "None":
                st.markdown("### 🕸️ Comparison Map")
                categories = list(results_a.keys())
                vals_a = [r['score'] if r['unit'] == "%" else (100-r['score']) for r in results_a.values()]
                vals_b = [r['score'] if r['unit'] == "%" else (100-r['score']) for r in results_b.values()]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=vals_a, theta=categories, fill='toself', name=f"{model_a} ({q_map_a})"))
                fig.add_trace(go.Scatterpolar(r=vals_b, theta=categories, fill='toself', name=f"{model_b} ({q_map_b})"))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Go to Discovery tab first.")

# ================= TAB 3: PLAYGROUND (Updated with Quantization) =================
with tab3:
    st.subheader("💬 Generation Playground")
    if 'models' in st.session_state:
        all_ids = st.session_state['models']['model_id'].tolist()
        select_options_play = ["None"] + all_ids
        
        pc1, pc2 = st.columns(2)
        with pc1:
            pm_a = st.selectbox("Generator A", select_options_play, index=1 if len(all_ids)>0 else 0, key="pm_a")
            pq_a = st.selectbox("Quant A", ["None (FP16)", "8-bit (Int8)"], key="pq_a")
        with pc2:
            pm_b = st.selectbox("Generator B", select_options_play, index=0, key="pm_b")
            pq_b = st.selectbox("Quant B", ["None (FP16)", "8-bit (Int8)"], key="pq_b")

        user_prompt = st.text_area("Prompt", value="Explain quantum computing like I'm 5.")

        if st.button("Generate Text"):
            c1, c2 = st.columns(2)
            pq_map_a = "8-bit" if "8-bit" in pq_a else "None"
            pq_map_b = "8-bit" if "8-bit" in pq_b else "None"

            # --- GEN A ---
            with c1:
                if pm_a != "None":
                    with st.spinner(f"Running {pm_a}..."):
                        succ, msg = st.session_state['manager'].load_model(pm_a, quantization=pq_map_a)
                        if succ:
                            out = st.session_state['manager'].generate_text(pm_a, pq_map_a, user_prompt)
                            st.info(out)
                        else: st.error(msg)
            
            # --- GEN B ---
            with c2:
                if pm_b != "None":
                    with st.spinner(f"Running {pm_b}..."):
                        succ, msg = st.session_state['manager'].load_model(pm_b, quantization=pq_map_b)
                        if succ:
                            out = st.session_state['manager'].generate_text(pm_b, pq_map_b, user_prompt)
                            st.success(out)
                        else: st.error(msg)
    else: st.warning("Please fetch models in Tab 1 first.")

# ================= TAB 4: VRAM ESTIMATOR (New) =================
with tab4:
    st.header("💾 Hardware Forecast")
    st.markdown("Estimate if a model will fit on your GPU (A100 vs 4090 vs 3060).")
    
    col1, col2 = st.columns(2)
    with col1:
        vram_input = st.text_input("Enter Model Size (e.g., 7B, 13B, 0.5B)", value="7B")
        if st.button("Calculate VRAM Requirements"):
            res = ModelDiagnostics.estimate_vram(vram_input)
            if res:
                st.session_state['vram_res'] = res
            else:
                st.error("Invalid format. Use format like '7B' or '1.5B'")
    
    with col2:
        if 'vram_res' in st.session_state:
            res = st.session_state['vram_res']
            st.success(f"**Results for {res['params_in_billions']} Billion Parameters**")
            
            st.markdown(f"""
            - **Training (FP32):** `{res['FP32 (Training/Full)']}` 🔴 (Needs A100/H100)
            - **Inference (FP16):** `{res['FP16 (Inference)']}` 🟡 (Fits on 3090/4090)
            - **Quantized (Int8):** `{res['INT8 (Quantized)']}` 🟢 (Fits on 3060/4060)
            """)

# ================= TAB 5: MODEL X-RAY (New) =================
with tab5:
    st.header("🩻 Model X-Ray")
    st.markdown("Inspect the raw architecture layers to understand Attention mechanisms.")
    
    if 'models' in st.session_state:
        # User selects from already loaded models or searches for one
        all_ids = st.session_state['models']['model_id'].tolist()
        xray_model = st.selectbox("Select Model to Inspect", all_ids)
        
        if st.button("Scan Layers"):
            with st.spinner("Scanning architecture..."):
                # We force load in FP16 for inspection
                succ, msg = st.session_state['manager'].load_model(xray_model, quantization="None")
                if succ:
                    mod, _ = st.session_state['manager'].get_components(xray_model, quantization="None")
                    structure = ModelDiagnostics.get_layer_structure(mod)
                    
                    st.text_area("Raw PyTorch Structure", value=structure, height=400)
                    st.success("Scan Complete.")
                else:
                    st.error(f"Could not load model for inspection: {msg}")
    else:
        st.warning("Go to Discovery tab first.")