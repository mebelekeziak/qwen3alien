# train_grpo_nohuman_think.py
# Simple GRPO loop that discourages human language inside <think>...</think>.
# Policy: Qwen/Qwen3-4B-Instruct (HF)
# Verifier: gemma-3-1b-it (Google GenAI API)

import os
import re
import json
import time
from functools import lru_cache
from typing import List, Tuple, Optional

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
import torch

# --- Google GenAI (verifier) ---
from google import genai
from google.genai import types


# ------------ Config ------------
POLICY_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"  # or "Qwen/Qwen3-4B-Instruct"
VERIFIER_MODEL_ID = "gemma-3-1b-it"              # served by Google GenAI SDK
MAX_THINK_CHARS = 4000                           # cap what we send to the verifier
THINK_REGEX = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# A light prefix so completions include <think>...</think><final>...</final>
PROMPT_PREFIX = (
    "Please answer the user. Format your reply exactly as:\n"
    "<think>your private reasoning</think>\n"
    "<final>the concise final answer for the user</final>\n\n"
)

# ------------ Verifier ------------
class HumanLanguageVerifier:
    """
    Uses Google's GenAI SDK to ask gemma-3-1b-it if text is human language.
    Returns (is_human_language: Optional[bool], confidence: float, detected_language: str)
    """
    def __init__(self, model: str = VERIFIER_MODEL_ID):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for google-genai.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

        # Try JSON/structured output; SDK supports response_mime_type on modern releases.
        # Falls back to free-text parse if necessary.
        self.config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=64,
            # This is widely supported for Gemini/Gemma via the GenAI SDK; if not, we still parse gracefully.
            response_mime_type="application/json",
        )

    @staticmethod
    def _heuristic(text: str) -> Tuple[bool, float, str]:
        """
        Fallback quick check if the API fails:
        - Any long Latin/CJK runs likely mean "human-ish" language.
        """
        # Latin letters 3+ in a row
        if re.search(r"[A-Za-z\u00C0-\u024F]{3,}", text):
            return True, 0.5, "latin-ish"
        # Common CJK block presence
        if re.search(r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]", text):
            return True, 0.5, "CJK-ish"
        return False, 0.3, "unknown"

    @lru_cache(maxsize=512)
    def classify(self, think_text: str) -> Tuple[Optional[bool], float, str]:
        prompt = (
            "You are a strict detector of HUMAN natural language. "
            "Given TEXT, decide if it contains any recognizable human language "
            "(English, Polish, Chinese, Japanese, Korean, German, etc.). "
            "Return only JSON like: "
            '{"human_language": true|false, "detected_language": "iso-or-unknown", "confidence": 0.0..1.0}\n\n'
            f"TEXT:\n{think_text}"
        )
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )],
                config=self.config,
            )
            out = resp.text or ""
            data = json.loads(out)
            human = data.get("human_language", None)
            conf = float(data.get("confidence", 0.5))
            lang = str(data.get("detected_language", "unknown"))
            if isinstance(human, bool):
                return human, conf, lang
            # If the LLM didn't follow schema, fallback:
            return self._heuristic(think_text)
        except Exception:
            # Network/JSON hiccup — fallback heuristic
            return self._heuristic(think_text)


# ------------ Reward function ------------
# TRL passes `completions` shaped like: List[List[{"role": "assistant", "content": "..."}]]
# (see TRL GRPO docs). We'll robustly extract text.
_verifier = None  # lazy init so script can import without Google key

def _extract_text_from_completion(completion) -> str:
    # completion could be: [{"role": "assistant", "content": "..."}] or just {"content": "..."} or a plain str
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        if isinstance(first, str):
            return first
    elif isinstance(completion, dict):
        return str(completion.get("content", ""))
    elif isinstance(completion, str):
        return completion
    return ""

def reward_no_human_language_in_think(completions, **kwargs) -> List[float]:
    """
    Reward policy for each sampled completion:
    +1 * confidence if <think> exists and is NOT human language,
    -1 * confidence if <think> exists and IS human language,
    -0.25 if <think> block is missing.
    """
    global _verifier
    if _verifier is None:
        _verifier = HumanLanguageVerifier()

    rewards = []
    for completion in completions:
        text = _extract_text_from_completion(completion)
        m = THINK_REGEX.search(text)
        if not m:
            rewards.append(-0.25)  # must include <think>...</think>
            continue

        think = m.group(1)[:MAX_THINK_CHARS]
        human, conf, _lang = _verifier.classify(think)
        if human is None:
            rewards.append(0.0)  # uncertain
        elif human:
            rewards.append(-1.0 * conf)  # penalize human language
        else:
            rewards.append(+1.0 * conf)  # reward emergent / non-human-ish
    return rewards


# ------------ Training ------------
def main():
    # Prompts: UltraFeedback prompts are fine as generic stimulus.
    ds = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

    # Prepend a tiny formatting instruction so the model reliably emits <think> then <final>.
    def _normalize_prompt(p) -> str:
        """Coerce dataset prompt field to a plain string.
        Handles str, list[str|dict], dict, or other types gracefully.
        """
        if isinstance(p, str):
            return p
        if isinstance(p, list):
            parts = []
            for item in p:
                if isinstance(item, dict):
                    parts.append(str(item.get("content", "")))
                else:
                    parts.append(str(item))
            return "\n".join(s for s in parts if s)
        if isinstance(p, dict):
            # Common chat-like single dict with content
            if "content" in p:
                return str(p.get("content", ""))
            return json.dumps(p, ensure_ascii=False)
        return str(p)

    def add_prefix(example):
        p = _normalize_prompt(example.get("prompt", ""))
        example["prompt"] = PROMPT_PREFIX + p
        return example

    ds = ds.map(add_prefix)

    # Ensure tokenizer pad side is left (recommended for generation training)
    tok = AutoTokenizer.from_pretrained(POLICY_MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Prefer CUDA if available and choose a suitable dtype
    use_cuda = torch.cuda.is_available()
    torch_dtype = (
        torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else
        (torch.float16 if use_cuda else torch.float32)
    )
    print(f"Device: {'cuda' if use_cuda else 'cpu'} | dtype: {torch_dtype}")

    # GRPO settings — simple/sane defaults; tune as you wish.
    args = GRPOConfig(
        output_dir="qwen3-4b-grpo-nohuman-think",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        num_train_epochs=1.0,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        bf16=True,  # if your GPUs support it
        gradient_checkpointing=True,

        # Generation settings used for sampling during GRPO:
        max_prompt_length=768,
        max_completion_length=512,
        # num_generations must divide generation_batch_size. By default,
        # TRL infers generation_batch_size ~= per_device_train_batch_size * gradient_accumulation_steps (=4).
        # Set to 4 to satisfy 4 % 4 == 0 and reduce VRAM use.
        num_generations=4,     # G in GRPO
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.0,

        # Reward scaling / loss flavor:
        loss_type="dapo",      # or "dr_grpo" per TRL docs
        scale_rewards="group", # default behavior in TRL
        beta=0.0,              # KL off by default per TRL guidance
        log_completions=True,
        num_completions_to_print=1,
        model_init_kwargs={
            "device_map": "auto" if use_cuda else None,
            "torch_dtype": torch_dtype,
            # Optional: enable if your setup supports it
            # "attn_implementation": "flash_attention_2",
            # Optional: reduce VRAM via 4-bit quantization (requires bitsandbytes)
            # "load_in_4bit": True,
        },
        # If you’re tight on VRAM, you can also pass:
        # model_init_kwargs={"load_in_4bit": True}
    )

    trainer = GRPOTrainer(
        model=POLICY_MODEL_ID,
        args=args,
        train_dataset=ds,
        reward_funcs=reward_no_human_language_in_think,
        processing_class=tok,  # ensures left padding + pad token setup
    )

    trainer.train()
    trainer.save_model()  # save PEFT/full model per your settings


if __name__ == "__main__":
    main()
