import re

class ModelDiagnostics:
    @staticmethod
    def estimate_vram(param_str):
        """
        Estimates VRAM usage based on parameter string (e.g., '7B', '0.5B').
        Formula: (Params * Precision Bytes) + 20% Overhead for Context/Activations
        """
        try:
            # Clean string and extract number
            clean_str = param_str.lower().replace('b', '').replace('m', '')
            val = float(clean_str)
            
            # Normalize to Billions
            if 'm' in param_str.lower():
                val = val / 1000.0
            
            # Constants
            overhead = 1.2 # 20% overhead for context window/activations
            
            # Calculations
            fp16_gb = (val * 2 * overhead)   # 2 bytes per param
            int8_gb = (val * 1 * overhead)   # 1 byte per param
            fp32_gb = (val * 4 * overhead)   # 4 bytes per param
            
            return {
                "FP32 (Training/Full)": f"{fp32_gb:.2f} GB",
                "FP16 (Inference)": f"{fp16_gb:.2f} GB",
                "INT8 (Quantized)": f"{int8_gb:.2f} GB",
                "params_in_billions": val
            }
        except Exception as e:
            return None

    @staticmethod
    def get_layer_structure(model):
        """
        Returns the raw string representation of the PyTorch model modules.
        """
        if model:
            # We strip the outer wrapper to get straight to the layers
            return str(model)
        return "Model not loaded."