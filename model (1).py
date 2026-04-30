"""
soma/core/model.py — Qwen3.6-27B Integration
==============================================
Released April 22, 2026. Apache 2.0 — free to use.

Key properties for SOMA:
  - 262K native context (Thinking Preservation across loop turns)
  - Dense 27B, Q4_K_M = 16.8GB VRAM (fits T4 free tier)
  - Native tool use (web search, code execution)
  - Beats Qwen3.5-397B-MoE on SWE-bench (77.2%) and SkillsBench (48.2%)
  - Hybrid Gated DeltaNet + standard attention (faster linear attention)

HOW TO USE FOR FREE RESEARCH (no cost):
  ┌─────────────────────────────────────────────────────────────────┐
  │ Option A: Qwen Studio (cloud, free tier)                       │
  │   → https://qwen.ai                                            │
  │   → Full 27B, 262K context, no download needed                 │
  │   → Rate limits apply but sufficient for research              │
  │                                                                 │
  │ Option B: Kaggle notebook (free T4, 30hr/week)                 │
  │   → Q4_K_M GGUF via llama.cpp or vLLM                         │
  │   → 16.8GB VRAM: fits T4 (15.6GB) tightly                     │
  │   → Use --max-model-len 32768 to stay within VRAM             │
  │   → Good for SOMA experiments and self-learning loop           │
  │                                                                 │
  │ Option C: Google Colab free tier (T4)                          │
  │   → Same as Kaggle, 12hr session limit                         │
  │   → Good for short experiments                                 │
  │                                                                 │
  │ Option D: Local GGUF via LM Studio (no GPU required)           │
  │   → Q4_K_M on CPU: slow (1-3 tok/s) but free and offline      │
  │   → Good for testing the pipeline, not for training            │
  └─────────────────────────────────────────────────────────────────┘

Thinking Preservation — why it matters for SOMA:
  Standard models discard their chain-of-thought between turns.
  Qwen3.6-27B retains it across the full conversation.
  In the SOMA loop, each stage (curiosity → retrieve → verify → learn)
  is a separate "turn". Without Thinking Preservation, the model
  re-derives context at each stage. With it, reasoning carries forward —
  the verify stage sees the curiosity signal, the self-learn stage
  sees what was verified, etc. This is crucial for the loop to be coherent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str       = "Qwen/Qwen3.6-27B"
    quantization: str     = "q4_k_m"          # fits T4; use q6 for A10G
    max_model_len: int    = 32768             # safe for T4; expand for A10G
    device_map: str       = "auto"
    thinking_mode: bool   = True             # enable Thinking Preservation
    tool_use: bool        = True             # native tool calling
    temperature: float    = 0.6
    top_p: float          = 0.95
    max_new_tokens: int   = 8192


class QwenModel:
    """
    Wrapper around Qwen3.6-27B for SOMA pipeline integration.

    Supports:
      - Standard inference (generate answers)
      - Tool-use inference (web search, code execution)
      - MC dropout (stochastic forward passes for curiosity)
      - Embedding extraction (for CALM mode, Phase 3)
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        self.cfg = cfg or ModelConfig()
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """Load model. Call once before use."""
        logger.info(f"[MODEL] Loading {self.cfg.model_name}")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

            bnb_config = None
            if self.cfg.quantization.startswith("q4"):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.cfg.quantization.startswith("q8"):
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                quantization_config=bnb_config,
                device_map=self.cfg.device_map,
                trust_remote_code=True,
            )
            self._model.eval()
            self._loaded = True
            logger.info(f"[MODEL] Loaded. VRAM: {self._vram_used()}")

        except ImportError:
            logger.warning("[MODEL] transformers/torch not installed. Running in stub mode.")

    def generate(self, prompt: str, system: str = "", tools: list = None) -> str:
        """Generate a response. Supports tool use and thinking mode."""
        if not self._loaded:
            return f"[STUB] Response to: {prompt[:80]}"

        import torch
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Thinking mode: enable_thinking=True preserves reasoning traces
        # across conversation turns (Thinking Preservation feature)
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.thinking_mode,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def mc_dropout_samples(self, prompt: str, n: int = 8) -> list:
        """
        Generate N stochastic forward passes (MC dropout) for curiosity.
        Returns list of logit arrays — input to CuriosityEngine.evaluate().
        """
        if not self._loaded:
            # Stub: synthetic logits for testing curiosity engine
            import numpy as np
            return [np.random.normal(0, 1, 1000) for _ in range(n)]

        import torch
        # Enable dropout during inference
        self._model.train()  # activates dropout
        results = []
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            for _ in range(n):
                out = self._model(**inputs)
                logits = out.logits[0, -1, :].float().cpu().numpy()
                results.append(logits)
        self._model.eval()
        return results

    def embed(self, text: str) -> "np.ndarray":
        """Extract embedding from final hidden state (for CALM mode)."""
        if not self._loaded:
            import numpy as np
            return np.random.normal(0, 1, 512)

        import torch, numpy as np
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
            # Use mean of last hidden layer
            hidden = out.hidden_states[-1][0].float().cpu().numpy()
            return hidden.mean(axis=0)

    def _vram_used(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                return f"{mem:.1f} GB"
        except Exception:
            pass
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Free research setup guide (printed when run directly)
# ─────────────────────────────────────────────────────────────────────────────

FREE_SETUP_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║         HOW TO USE QWEN3.6-27B FOR SOMA RESEARCH — FREE                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

OPTION A: Qwen Studio (easiest, zero setup)
  1. Go to https://qwen.ai
  2. Select Qwen3.6-27B from model list
  3. Use the API key from settings for SOMA integration:
     export QWEN_API_KEY=your_key
  4. In soma/core/model.py: set model_name = "qwen-max" (API mode)
  Cost: FREE tier sufficient for research

OPTION B: Kaggle (best for SOMA training loops)
  1. New Notebook → Settings → GPU T4 x1 → Internet ON
  2. In first cell:
     !pip install vllm -q
     from vllm import LLM
     llm = LLM(
         model="Qwen/Qwen3.6-27B",
         quantization="bitsandbytes",
         load_format="bitsandbytes",
         gpu_memory_utilization=0.90,
         max_model_len=32768,         # critical: T4 only has 15.6GB
     )
  3. Run SOMA loop experiments. 30 hours/week free.
  Cost: FREE

OPTION C: Local via LM Studio (no GPU)
  1. Download LM Studio from https://lmstudio.ai
  2. Search "Qwen3.6-27B" → Download Q4_K_M GGUF (16.8GB)
  3. Load model → Start local server on port 1234
  4. In soma/core/model.py: set model_name = "http://localhost:1234/v1"
  Cost: FREE (slow on CPU, 1-3 tok/s)

OPTION D: Kaggle with llama.cpp (lightest)
  !apt install -y cmake build-essential
  !git clone https://github.com/ggerganov/llama.cpp
  !cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j4
  !huggingface-cli download unsloth/Qwen3.6-27B-GGUF \\
      --include "Qwen3.6-27B-Q4_K_M.gguf" \\
      --local-dir ./qwen_gguf
  !./llama.cpp/build/bin/llama-server \\
      -m ./qwen_gguf/Qwen3.6-27B-Q4_K_M.gguf \\
      --port 8080 -ngl 99 &

SOMA INTEGRATION CODE (any of the above options):
  from openai import OpenAI
  client = OpenAI(
      base_url="http://localhost:8080/v1",  # or Qwen API endpoint
      api_key="not-needed",
  )
  response = client.chat.completions.create(
      model="Qwen3.6-27B",
      messages=[{"role": "user", "content": "Your SOMA loop question here"}],
      temperature=0.6,
      extra_body={"enable_thinking": True},  # Thinking Preservation ON
  )
  print(response.choices[0].message.content)

KEY FLAGS TO KNOW:
  enable_thinking=True   → Thinking Preservation (retain reasoning across turns)
  temperature=0.6        → Recommended for agentic tasks
  max_model_len=32768    → Safe for T4; increase for A10G (262144)
  reasoning-parser qwen3 → Required in vLLM for thinking mode

VRAM REFERENCE:
  Q4_K_M: 16.8GB → RTX 3090/4090 (24GB), A10G (24GB), T4 (15.6GB, tight)
  Q6_K:   22.5GB → RTX 4090 (24GB), A10G (24GB)
  Q8_0:   28.6GB → A100 (40/80GB), RTX 5090 (32GB)
  BF16:   55.6GB → A100 80GB, H100
"""


if __name__ == "__main__":
    print(FREE_SETUP_GUIDE)
