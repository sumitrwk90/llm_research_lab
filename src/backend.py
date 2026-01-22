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
            filter=filter_tags if filter_tags else None, task=hf_task
        )
        
        # ... (Same cleaning logic as before) ...
        model_list = []
        for m in models:
            size_match = re.search(r'([0-9\.]+)b', m.modelId.lower())
            size_label = f"{size_match.group(1)}B" if size_match else "N/A"
            model_list.append({
                "model_id": m.modelId, "likes": m.likes, "downloads": m.downloads,
                "created_at": str(m.created_at)[:10], "estimated_params": size_label
            })
        return pd.DataFrame(model_list)

class ModelManager:
    """Manages loading and inference for models."""
    def __init__(self, device="cpu"):
        self.device = device
        self.loaded_models = {} # Store {model_id: {model, tokenizer}}

    def load_model(self, model_id):
        if model_id in self.loaded_models:
            return True, "Already Loaded"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Fix for models with no pad token
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
                
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True
            ).to(self.device)
            model.eval()
            
            self.loaded_models[model_id] = {"model": model, "tokenizer": tokenizer}
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def generate_text(self, model_id, prompt, max_new_tokens=100):
        if model_id not in self.loaded_models: return "Error: Model not loaded."
        
        pkg = self.loaded_models[model_id]
        inputs = pkg["tokenizer"](prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = pkg["model"].generate(
                **inputs, max_new_tokens=max_new_tokens, pad_token_id=pkg["tokenizer"].eos_token_id
            )
        return pkg["tokenizer"].decode(outputs[0], skip_special_tokens=True)

    def get_components(self, model_id):
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]["model"], self.loaded_models[model_id]["tokenizer"]
        return None, None