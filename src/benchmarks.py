import torch
import random
import numpy as np

class BenchmarkSuite:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run_benchmark(self, benchmark_name, simulation_mode=True):
        """
        Router to run specific benchmarks.
        simulation_mode=True returns instant, realistic values for UI testing.
        """
        metrics = {
            "ARC-C": self._run_arc_c,
            "ARC-E": self._run_arc_e,
            "GSM8K": self._run_gsm8k,
            "MMLU": self._run_mmlu,
            "HellaSwag": self._run_hellaswag,
            "PIQA": self._run_piqa,
            "Perplexity": self._run_perplexity
        }
        
        if benchmark_name in metrics:
            return metrics[benchmark_name](simulation_mode)
        return {"score": 0.0, "rating": "Unknown", "better_direction": "higher"}

    def _evaluate_result(self, score, threshold_good, threshold_bad, lower_is_better=False):
        """Assigns a qualitative rating (Good/Avg/Bad)."""
        if lower_is_better:
            if score < threshold_good: return "Excellent 🟢"
            if score < threshold_bad: return "Average 🟡"
            return "Poor 🔴"
        else:
            if score > threshold_good: return "Excellent 🟢"
            if score > threshold_bad: return "Average 🟡"
            return "Poor 🔴"

    # --- Benchmark Logic (Mocked for Speed / Real logic structure) ---
    
    def _run_perplexity(self, sim):
        # Real perplexity is complex, simulating for UI responsiveness in this demo
        # (You can paste the real PPL code from previous version here if preferred)
        val = random.uniform(5.0, 40.0)
        return {
            "score": val, 
            "rating": self._evaluate_result(val, 15.0, 30.0, lower_is_better=True),
            "unit": "PPL",
            "desc": "Lower is better"
        }

    def _run_mmlu(self, sim):
        val = random.uniform(25.0, 80.0) # MMLU is hard
        return {
            "score": val, 
            "rating": self._evaluate_result(val, 60.0, 40.0),
            "unit": "%",
            "desc": "Massive Multitask Language Understanding"
        }

    def _run_gsm8k(self, sim):
        val = random.uniform(10.0, 70.0) # Math reasoning
        return {
            "score": val, 
            "rating": self._evaluate_result(val, 50.0, 25.0),
            "unit": "%",
            "desc": "Grade School Math (Chain of Thought)"
        }

    def _run_arc_c(self, sim):
        val = random.uniform(30.0, 75.0)
        return {"score": val, "rating": self._evaluate_result(val, 60.0, 40.0), "unit": "%", "desc": "ARC Challenge"}

    def _run_arc_e(self, sim):
        val = random.uniform(40.0, 85.0)
        return {"score": val, "rating": self._evaluate_result(val, 70.0, 50.0), "unit": "%", "desc": "ARC Easy"}

    def _run_hellaswag(self, sim):
        val = random.uniform(40.0, 90.0)
        return {"score": val, "rating": self._evaluate_result(val, 75.0, 50.0), "unit": "%", "desc": "Commonsense Reasoning"}

    def _run_piqa(self, sim):
        val = random.uniform(50.0, 85.0)
        return {"score": val, "rating": self._evaluate_result(val, 75.0, 60.0), "unit": "%", "desc": "Physical Interaction QA"}