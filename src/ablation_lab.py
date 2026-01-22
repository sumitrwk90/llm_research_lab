import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import copy
from src.backend import ModelManager

class AblationEngine:
    """
    Handles the 'Virtual Surgery' of models using PyTorch Hooks.
    Instead of deleting code, we intercept signals during inference.
    """
    def __init__(self, model_manager):
        self.manager = model_manager
        self.active_hooks = []
        self.ablation_log = []

    def clear_hooks(self):
        """Removes all active ablations (restores model to baseline)."""
        for handle in self.active_hooks:
            handle.remove()
        self.active_hooks = []

    def register_ablation(self, model, layer_name, ablation_type="zero_out", noise_level=0.1):
        """
        Injects a hook into a specific layer to modify its output.
        """
        target_module = dict(model.named_modules())[layer_name]
        
        def hook_fn(module, input, output):
            if ablation_type == "zero_out":
                # Structural Ablation: Kill the signal
                return output * 0.0
            
            elif ablation_type == "add_noise":
                # Robustness Test: Inject Gaussian noise
                noise = torch.randn_like(output) * noise_level
                return output + noise
            
            elif ablation_type == "freeze_mean":
                # Information Bottleneck: Replace with batch mean
                return torch.mean(output, dim=0, keepdim=True).expand_as(output)
                
            return output

        # Register the hook
        handle = target_module.register_forward_hook(hook_fn)
        self.active_hooks.append(handle)
        return f"Ablated {layer_name} ({ablation_type})"

class ArchitectureVisualizer:
    """
    Builds a Netron-style interactive graph of the model layers using NetworkX + Plotly.
    """
    @staticmethod
    def build_layer_graph(model):
        G = nx.DiGraph()
        prev_node = "Input"
        G.add_node("Input", type="Input")

        # Walk through modules (simplified for visualization)
        # We limit depth to avoid 10,000 node graphs for LLMs
        for name, module in model.named_modules():
            # Filter for high-level blocks only (Layers, Attention, MLP)
            if any(k in name for k in ["layer", "block", "attn", "mlp"]) and "." not in name.split(".")[-1]:
                # Heuristic: Connect sequential blocks
                G.add_node(name, type=module.__class__.__name__, params=sum(p.numel() for p in module.parameters()))
                G.add_edge(prev_node, name)
                prev_node = name
        
        G.add_node("Output", type="Output")
        G.add_edge(prev_node, "Output")
        return G

    @staticmethod
    def plot_interactive_graph(G):
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none', mode='lines'
        )

        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            info = G.nodes[node]
            node_text.append(f"{node}<br>{info.get('type', 'Unknown')}<br>Params: {info.get('params', 'N/A')}")
            
            # Color coding
            if "attn" in node.lower(): node_color.append("#FF0055") # Attention
            elif "mlp" in node.lower(): node_color.append("#00CC96") # MLP
            elif "layer" in node.lower(): node_color.append("#AB63FA") # Blocks
            else: node_color.append("#FFFFFF")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(showscale=False, color=node_color, size=15, line_width=2)
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0,l=0,r=0,t=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig

def render_ablation_dashboard():
    # --- Custom CSS for the Dashboard Feel ---
    st.markdown("""
    <style>
        .ablation-header { 
            background: linear-gradient(90deg, #FF4B4B 0%, #FF9068 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 30px; font-weight: 900;
        }
        .stat-box {
            background-color: #1E1E1E; border: 1px solid #333;
            padding: 15px; border-radius: 5px; text-align: center;
        }
        .risk-high { border-left: 5px solid #FF4B4B; }
        .risk-med { border-left: 5px solid #FFAA00; }
        .risk-low { border-left: 5px solid #00FF00; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ablation-header">🧪 SYSTEMATIC ABLATION LAB</div>', unsafe_allow_html=True)
    st.caption("Surgically alter model components to measure contribution and robustness.")

    if 'models' not in st.session_state:
        st.warning("Please load models in the Discovery tab first.")
        return

    # 1. Select Subject
    col_sel, col_viz = st.columns([1, 3])
    
    with col_sel:
        st.subheader("1. Subject")
        all_ids = st.session_state['models']['model_id'].tolist()
        target_model_id = st.selectbox("Select Model for Surgery", all_ids)
        
        # Load Model Button
        if st.button("Initialize Surgery Table"):
            with st.spinner("Preparing model for hooks..."):
                succ, msg = st.session_state['manager'].load_model(target_model_id)
                if succ:
                    st.success("Ready.")
                    st.session_state['ablation_target'] = target_model_id
                    # Initialize engine
                    st.session_state['ablation_engine'] = AblationEngine(st.session_state['manager'])
                else:
                    st.error(msg)

    # 2. Main Workspace
    if 'ablation_target' in st.session_state:
        target_id = st.session_state['ablation_target']
        model_pkg = st.session_state['manager'].loaded_models.get(f"{target_id}_None") # Default FP32/16 key
        
        if not model_pkg:
            st.error("Model lost from memory. Please reload.")
            return

        model = model_pkg['model']
        
        # --- TAB LAYOUT FOR ABLATION ---
        t1, t2, t3 = st.tabs(["🧬 Structural Map", "🔪 Ablation Controls", "📊 Impact Report"])

        # === TAB 1: ARCHITECTURE GRAPH ===
        with t1:
            st.markdown("### Interactive Architecture Map")
            st.markdown("Visualize the flow to decide where to cut.")
            
            if st.button("Generate Graph (Heavy Compute)"):
                with st.spinner("Tracing neural pathways..."):
                    G = ArchitectureVisualizer.build_layer_graph(model)
                    fig = ArchitectureVisualizer.plot_interactive_graph(G)
                    st.plotly_chart(fig, use_container_width=True)

        # === TAB 2: CONTROLS ===
        with t2:
            st.subheader("Configure Ablation Experiment")
            
            c1, c2 = st.columns(2)
            with c1:
                # Get all layers
                all_layers = [n for n, _ in model.named_modules() if len(n) > 0]
                target_layers = st.multiselect("Select Target Layers", all_layers, max_selections=5)
            
            with c2:
                method = st.selectbox("Ablation Method", 
                                      ["Zero-Out (Remove)", "Add Noise (Corrupt)", "Freeze Mean (Bottleneck)"])
                if method == "Add Noise (Corrupt)":
                    noise_val = st.slider("Noise Level (Std Dev)", 0.0, 2.0, 0.1)
                else:
                    noise_val = 0.0

            if st.button("🔴 RUN ABLATION TEST"):
                engine = st.session_state['ablation_engine']
                engine.clear_hooks() # Reset previous
                
                results_log = []
                
                # 1. Establish Baseline
                st.write("Measuring Baseline Performance...")
                # We simply use a generation prompt length as a proxy for "Performance" 
                # or run a quick perplexity check if integrated with benchmarks.
                # For this dashboard, we run the "Prompt Integrity Test"
                
                prompt = "The capital of France is"
                base_out = st.session_state['manager'].generate_text(target_id, "None", prompt)
                results_log.append({"State": "Baseline", "Output": base_out, "Integrity": 100})
                
                # 2. Apply Hooks
                for layer in target_layers:
                    msg = engine.register_ablation(model, layer, method.lower().split()[0].replace("-","_"), noise_val)
                    st.toast(msg)
                
                # 3. Measure Ablated Performance
                st.write("Running Ablated Inference...")
                ablated_out = st.session_state['manager'].generate_text(target_id, "None", prompt)
                
                # Simple heuristic: String similarity or length retention
                integrity = (len(ablated_out) / len(base_out)) * 100 if len(base_out) > 0 else 0
                results_log.append({"State": "Ablated", "Output": ablated_out, "Integrity": integrity})
                
                st.session_state['ablation_results'] = results_log
                
                # Cleanup
                engine.clear_hooks()
                st.success("Experiment Complete. Hooks Removed.")

        # === TAB 3: RESULTS ===
        with t3:
            if 'ablation_results' in st.session_state:
                res = st.session_state['ablation_results']
                
                # Visual Diff
                st.markdown("### 📝 Output Degradation Analysis")
                
                col_base, col_abl = st.columns(2)
                with col_base:
                    st.info(f"**Baseline:** {res[0]['Output']}")
                with col_abl:
                    st.warning(f"**Ablated:** {res[1]['Output']}")
                
                # Metrics
                deg = 100 - res[1]['Integrity']
                st.metric("Model Degradation", f"{deg:.1f}%", delta=f"-{deg:.1f}%", delta_color="inverse")
                
                # Sensitivity Chart (Mocked for single run, would need loop for real sensitivity analysis)
                st.markdown("### 🔥 Layer Sensitivity Heatmap")
                
                # Creating dummy data to show what the "full suite" would look like
                sens_data = pd.DataFrame({
                    "Layer": ["embed", "layer.0", "layer.1", "layer.2", "head"],
                    "Sensitivity Score": [95, 10, 15, 80, 100]
                })
                
                fig = px.bar(sens_data, x="Layer", y="Sensitivity Score", 
                             color="Sensitivity Score", color_continuous_scale="RdYlGn_r",
                             title="Estimated Contribution to Output (Simulated)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run an experiment in Tab 2 to see results.")