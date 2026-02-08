import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re

class ModelResearcher:
    def __init__(self):
        self.api = HfApi()

    def search_models(self, task_domain="Language", architecture_type="All", sort_by="downloads", limit=50):
        hf_task = "text-generation" if task_domain == "Language" else "image-classification"
        filter_tags = []
        if architecture_type == "Recurrent (RNN/RWKV/Mamba)": filter_tags.append("rwkv") 
        elif architecture_type == "Attention (Transformer)": filter_tags.append("transformers")
        
        models = self.api.list_models(
            sort=sort_by, direction=-1, limit=limit,
            filter=filter_tags if filter_tags else None, #task=hf_task
        )
        
        model_list = []
        for m in models:
            size_match = re.search(r'([0-9\.]+)b', m.modelId.lower())
            size_label = f"{size_match.group(1)}B" if size_match else "N/A"
            if size_label == "N/A": # Fallback check for millions
                 size_match_m = re.search(r'([0-9\.]+)m', m.modelId.lower())
                 size_label = f"{size_match_m.group(1)}M" if size_match_m else "N/A"

            model_list.append({
                "model_id": m.modelId, "likes": m.likes, "downloads": m.downloads,
                "created_at": str(m.created_at)[:10], "estimated_params": size_label
            })
        return pd.DataFrame(model_list)

class ModelManager:
    def __init__(self, device="cpu"):
        self.device = device
        self.loaded_models = {} 

    def load_model(self, model_id, quantization="None"):
        """
        Loads model with optional 8-bit quantization.
        quantization: "None" (FP16/32) or "8-bit"
        """
        # Create a unique key for caching (e.g., "distilgpt2_8bit")
        cache_key = f"{model_id}_{quantization}"
        
        if cache_key in self.loaded_models:
            return True, "Already Loaded"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            
            # Quantization Logic
            load_kwargs = {"trust_remote_code": True}
            
            if quantization == "8-bit":
                if self.device == "cpu":
                    return False, "8-bit quantization requires a GPU (CUDA)."
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto" # Required for bitsandbytes
            else:
                # Standard Loading
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                load_kwargs["torch_dtype"] = dtype
                
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            
            if quantization != "8-bit":
                model = model.to(self.device)
            
            model.eval()
            self.loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer}
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def generate_text(self, model_id, quantization, prompt, max_new_tokens=100):
        cache_key = f"{model_id}_{quantization}"
        if cache_key not in self.loaded_models: return "Error: Model not loaded."
        
        pkg = self.loaded_models[cache_key]
        inputs = pkg["tokenizer"](prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = pkg["model"].generate(
                **inputs, max_new_tokens=max_new_tokens, pad_token_id=pkg["tokenizer"].eos_token_id
            )
        return pkg["tokenizer"].decode(outputs[0], skip_special_tokens=True)

    def get_components(self, model_id, quantization="None"):
        cache_key = f"{model_id}_{quantization}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]["model"], self.loaded_models[cache_key]["tokenizer"]
        return None, None