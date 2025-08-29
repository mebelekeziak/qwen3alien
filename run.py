#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL script that penalizes human-language-like tokens **inside** <think>…</think>
while keeping the final answer human-readable. Supports PPO (default) and
optionally GRPO if your TRL install provides it.

Tested assumptions:
- Hugging Face Transformers >= 4.40
- TRL >= 0.9 (PPO); GRPO is optional and may require newer TRL.
- Qwen family or any chat/instruct model that can follow a template and emit
  <think>…</think>. Default model is set to a broadly available one; change to
  your preferred Qwen-3-8B if you have it.

This script uses a synthetic arithmetic task for R_task so you can verify the
behavior end-to-end without external datasets.

Usage (PPO):
  python rl_no_human_language_think.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --ppl-model gpt2 \
      --output-dir runs/qwen_no_human_think \
      --steps 2000 --bsz 4 --gen-k 1 --max-new-tokens 192

Usage (attempt GRPO, will fallback to PPO if unavailable):
  python rl_no_human_language_think.py --algo grpo ...

Notes:
- The language penalty only looks at the <think> span.
- The script includes an adversarial char-histogram classifier updated online
  (tiny and fast). You can disable it with --no-adv.
- A small length penalty discourages bloated thinking.
- We anchor the final answer with a tiny KL to a frozen ref model (optional).

"""

import argparse
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    set_seed,
)

# Optional Google GenAI client for remote scoring
try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
    HAS_GOOGLE_GENAI = True
except Exception:
    HAS_GOOGLE_GENAI = False

# TRL imports (PPO always; GRPO optional)
try:
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
except Exception as e:
    print("[WARN] TRL PPO imports failed. Please install 'trl' >= 0.9.")
    raise

try:
    from trl import GRPOConfig, GRPOTrainer
    HAS_GRPO = True
except Exception:
    HAS_GRPO = False

try:
    from peft import LoraConfig
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

# ------------------------- Utilities ---------------------------------------

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)

HUMAN_SEED_TEXT = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, researchers proposed a new method for robust alignment.",
    "Warsaw is the capital of Poland and has a vibrant tech scene.",
    "Mathematics is the language of nature, revealing patterns and symmetries.",
    "Large language models can reason, summarize, and generate creative text.",
    "Open problems remain in interpretability and controllable generation.",
]

# Small ASCII set for statistics
ASCII_SET = [chr(i) for i in range(32, 127)]
ASCII2IDX = {c: i for i, c in enumerate(ASCII_SET)}


def safe_base64_decode(s: str) -> Optional[str]:
    import base64

    s_clean = re.sub(r"\s+", "", s)
    if len(s_clean) < 8:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", s_clean or " "):
        return None
    try:
        b = base64.b64decode(s_clean, validate=True)
        if not b:
            return None
        # decode as utf-8 if possible; else return repr
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return None
    except Exception:
        return None


def parse_think_and_final(txt: str) -> Tuple[str, str]:
    m = THINK_RE.search(txt)
    if not m:
        return "", txt.strip()
    think = m.group(1).strip()
    final = txt[m.end():].strip()
    return think, final


# -------------------- Human-likeness detector ------------------------------

class CharHistDisc(nn.Module):
    """Tiny adversarial classifier on ASCII char histograms."""

    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(len(ASCII_SET), hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):  # x: [B, 95]
        h = F.gelu(self.fc1(x))
        logit = self.fc2(h)
        return torch.sigmoid(logit).squeeze(-1)


def char_histogram(s: str) -> torch.Tensor:
    counts = torch.zeros(len(ASCII_SET), dtype=torch.float32)
    for ch in s:
        if ch in ASCII2IDX:
            counts[ASCII2IDX[ch]] += 1.0
    if counts.sum() > 0:
        counts = counts / counts.sum()
    return counts


def char_stats_score(s: str) -> float:
    if not s:
        return 0.0
    total = len(s)
    letters = sum(ch.isalpha() for ch in s)
    spaces = s.count(" ")
    digits = sum(ch.isdigit() for ch in s)
    punct = sum(ch in ",.;:!?-'\"()[]{}" for ch in s)
    frac_letters = letters / total
    frac_spaces = spaces / total
    frac_digits = digits / total
    frac_punct = punct / total
    # entropy
    import math

    hist = {}
    for ch in s:
        hist[ch] = hist.get(ch, 0) + 1
    probs = [c / total for c in hist.values()]
    entropy = -sum(p * math.log(p + 1e-9) for p in probs)
    # Normalize entropy to ~[0,1] assuming max~log(V) with V<=95
    entropy_norm = min(entropy / math.log(95), 1.0)

    # Heuristic: human-ish strings tend to have many letters and spaces,
    # moderate punctuation, fewer digits; entropy in mid-high range.
    score = (
        0.45 * frac_letters
        + 0.35 * frac_spaces
        + 0.10 * (1 - frac_digits)
        + 0.10 * min(frac_punct * 3.0, 1.0)
    )
    # Blend with entropy
    score = 0.7 * score + 0.3 * entropy_norm
    return float(max(0.0, min(1.0, score)))


class HumanDetector:
    """Ensemble: LM perplexity + char stats + tiny adversarial classifier.
    Returns score in [0,1] where 1 = very human-like.
    """

    def __init__(self, ppl_model_name: str = "gpt2", device: Optional[str] = None, use_adv: bool = True, google_api_key: Optional[str] = None):
        # Device for adversary and optional local PPL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_adv = use_adv
        self.remote_provider = None

        # If ppl_model_name starts with "google:", configure remote PPL via Google API (Gemini/Gemma)
        if isinstance(ppl_model_name, str) and ppl_model_name.lower().startswith("google:"):
            self.remote_provider = "google"
            self.google_model = ppl_model_name.split(":", 1)[1] or "gemma-3-4b-it"
            # Prefer GEMINI_API_KEY, then GOOGLE_API_KEY
            self.google_api_key = (
                google_api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if not self.google_api_key:
                print("[WARN] GEMINI_API_KEY/GOOGLE_API_KEY not set; falling back to local PPL model 'gpt2'.")
                self.remote_provider = None
                ppl_model_name = "gpt2"
            elif not HAS_GOOGLE_GENAI:
                print("[WARN] google-genai package not found. Install with 'pip install google-genai'. Falling back to local PPL 'gpt2'.")
                self.remote_provider = None
                ppl_model_name = "gpt2"
            else:
                # Initialize client
                self.genai_client = google_genai.Client(api_key=self.google_api_key)

        # Initialize local PPL model only if not using remote
        if self.remote_provider is None:
            self.ppl_tok = AutoTokenizer.from_pretrained(ppl_model_name)
            self.ppl_lm = AutoModelForCausalLM.from_pretrained(ppl_model_name).to(self.device)
            self.ppl_lm.eval()
        # Tiny adversary
        self.disc = CharHistDisc(hidden=64).to(self.device)
        self.disc_opt = torch.optim.AdamW(self.disc.parameters(), lr=1e-3)
        # Warm-up adversary with seed data
        self._bootstrap_adv()

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        if not text.strip():
            return 100.0  # very high ppl for empty (so low human score)
        # Remote provider branch (Google API): return pseudo-PPL from a 0..1 score
        if self.remote_provider == "google":
            score = self._google_humanlikeness_score(text)
            if score is None:
                return 100.0
            # Map score in (0,1] to a pseudo-perplexity so that s1 ~= score
            # s1 = 1 / ppl  => ppl = 1 / s1
            ppl = 1.0 / max(1e-6, min(1.0, float(score)))
            return float(max(1.0, ppl))
        # Local PPL branch
        toks = self.ppl_tok(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.ppl_lm(**toks, labels=toks.input_ids)
            loss = out.loss.item()
        ppl = math.exp(min(20.0, max(0.0, loss)))
        return float(ppl)

    def _google_humanlikeness_score(self, text: str) -> Optional[float]:
        try:
            # Build instruction to return only a numeric score in [0,1]
            instruction = (
                "You are a strict evaluator. Given the following text, output only a single number between 0 and 1 "
                "representing how human-natural-language-like it is (1 = very human-like, 0 = not human-like). "
                "Do not include any extra words or symbols."
            )
            user_prompt = f"{instruction}\n\nText:\n```\n{text}\n```\n\nScore:"
            contents = [
                google_genai_types.Content(
                    role="user",
                    parts=[google_genai_types.Part.from_text(text=user_prompt)],
                )
            ]
            config = google_genai_types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.0,
                max_output_tokens=4,
            )
            out_text = ""
            for chunk in self.genai_client.models.generate_content_stream(
                model=self.google_model,
                contents=contents,
                config=config,
            ):
                if hasattr(chunk, "text") and chunk.text:
                    out_text += chunk.text
            out_text = (out_text or "").strip()
            m = re.search(r"([01](?:\.\d+)?|0?\.\d+)", out_text)
            if not m:
                return None
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return val
            return None
        except Exception as e:
            print(f"[WARN] Google API PPL call failed: {e}")
            return None

    def p_human(self, text: str) -> float:
        # Preprocess: try base64 decode; if plausible, score on decoded too (take max)
        candidates = [text]
        dec = safe_base64_decode(text)
        if dec:
            candidates.append(dec)
        scores = []
        for t in candidates:
            ppl = self.perplexity(t)
            s1 = 1.0 / (1.0 + max(0.0, ppl - 1.0))  # lower ppl -> closer to 1
            s2 = char_stats_score(t)
            if self.use_adv:
                with torch.no_grad():
                    h = char_histogram(t).to(self.device).unsqueeze(0)
                    s3 = float(self.disc(h).item())
            else:
                s3 = 0.0
            scores.append( (s1 + s2 + s3) / (3.0 if self.use_adv else 2.0) )
        return float(max(scores))

    def _bootstrap_adv(self):
        if not self.use_adv:
            return
        # Train a few steps on seed human vs. synthetic negative (random symbol soup)
        pos = HUMAN_SEED_TEXT
        neg = [
            "".join(random.choice("#$%&@^~_+") for _ in range(80)),
            "1010011100100011110001110000111000",
            "z1 z2 z3 z1 z2 z3 z1 z2 z3",
            "AB12|AB12|AB12|AB12|AB12|AB12",
        ]
        self.update(pos, neg, steps=100)

    def update(self, positives: List[str], negatives: List[str], steps: int = 30, batch: int = 32):
        if not self.use_adv:
            return
        data = [(s, 1.0) for s in positives] + [(s, 0.0) for s in negatives]
        random.shuffle(data)
        X = torch.stack([char_histogram(s) for s, _ in data]).to(self.device)
        y = torch.tensor([y for _, y in data], dtype=torch.float32, device=self.device)
        for _ in range(steps):
            # random minibatch
            idx = torch.randint(0, X.size(0), (min(batch, X.size(0)),), device=self.device)
            xb, yb = X[idx], y[idx]
            p = self.disc(xb)
            loss = F.binary_cross_entropy(p, yb)
            self.disc_opt.zero_grad()
            loss.backward()
            self.disc_opt.step()


# -------------------- Task: simple arithmetic ------------------------------

@dataclass
class ArithConfig:
    min_terms: int = 2
    max_terms: int = 3
    min_val: int = 1
    max_val: int = 99
    ops: Tuple[str, ...] = ("+", "-")


def sample_arith(cfg: ArithConfig) -> Tuple[str, int]:
    n = random.randint(cfg.min_terms, cfg.max_terms)
    nums = [random.randint(cfg.min_val, cfg.max_val) for _ in range(n)]
    ops = [random.choice(cfg.ops) for _ in range(n - 1)]
    expr = []
    val = nums[0]
    for i, op in enumerate(ops):
        expr.append(str(nums[i]))
        expr.append(op)
        if op == "+":
            val += nums[i + 1]
        else:
            val -= nums[i + 1]
    expr.append(str(nums[-1]))
    prompt = "Compute the result of: " + " ".join(expr)
    return prompt, val


def extract_int(s: str) -> Optional[int]:
    # Find last integer in the string
    m = re.findall(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m[-1])
    except Exception:
        return None


# -------------------- Prompt Template --------------------------------------

TEMPLATE = (
    "You are a helpful assistant. Solve the user's task.\n"
    "Always respond in the following format:\n"
    "<think>your internal reasoning here</think>\n"
    "Final: the final short answer here.\n"
)


def build_query(user_prompt: str) -> str:
    return TEMPLATE + "\nUser: " + user_prompt + "\nAssistant:"


# -------------------- Training loop ----------------------------------------

@dataclass
class Args:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    ref_model: Optional[str] = None
    # Default to Google API Gemma for perplexity via remote call.
    # Use format: "google:<model_name>" to hit Google API.
    # Example: --ppl-model google:gemma-3-4b-it
    # To use a local model instead, pass a HF id, e.g. --ppl-model gpt2
    ppl_model: str = "google:gemma-3-4b-it"
    output_dir: str = "runs/no_human_think"
    seed: int = 42
    steps: int = 2000
    eval_every: int = 200
    save_every: int = 500
    bsz: int = 4                # prompts per PPO/GRPO update
    mini_bsz: int = 2
    gen_k: int = 1              # GRPO generations per prompt (K)
    max_new_tokens: int = 192
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    w_task: float = 1.0
    w_lang: float = 0.8
    w_len: float = 0.02
    use_adv: bool = True
    algo: str = "ppo"           # or "grpo"
    lr: float = 1e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    no_lora: bool = False
    bf16: bool = True
    max_prompt_len: int = 512
    arith_min_terms: int = 2
    arith_max_terms: int = 3


def make_policy_and_tokenizer(a: Args):
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    # Ensure <think> tokens exist (not strictly necessary but helpful)
    special = {"additional_special_tokens": ["<think>", "</think>"]}
    tok.add_special_tokens(special)

    peft_config = None
    if HAS_PEFT and not a.no_lora:
        peft_config = LoraConfig(
            r=a.lora_r,
            lora_alpha=a.lora_alpha,
            lora_dropout=a.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    if a.algo.lower() == "ppo":
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            a.model,
            torch_dtype=torch.bfloat16 if a.bf16 and torch.cuda.is_available() else None,
            device_map="auto",
            peft_config=peft_config,
        )
        model.pretrained_model.resize_token_embeddings(len(tok))
        ref_model = None
        if a.ref_model:
            ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                a.ref_model,
                torch_dtype=torch.bfloat16 if a.bf16 and torch.cuda.is_available() else None,
                device_map="auto",
            )
            ref_model.pretrained_model.resize_token_embeddings(len(tok))
        return model, ref_model, tok
    else:
        # GRPO policy doesn't use value head directly; TRL handles it internally
        policy = AutoModelForCausalLM.from_pretrained(
            a.model,
            torch_dtype=torch.bfloat16 if a.bf16 and torch.cuda.is_available() else None,
            device_map="auto",
        )
        policy.resize_token_embeddings(len(tok))
        ref = None
        if a.ref_model:
            ref = AutoModelForCausalLM.from_pretrained(
                a.ref_model,
                torch_dtype=torch.bfloat16 if a.bf16 and torch.cuda.is_available() else None,
                device_map="auto",
            )
            ref.resize_token_embeddings(len(tok))
        return policy, ref, tok


def task_score(prompt: str, final: str, correct: int) -> float:
    pred = extract_int(final)
    return 1.0 if pred is not None and pred == correct else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=Args.model)
    p.add_argument("--ref-model", type=str, default=None)
    p.add_argument("--ppl-model", type=str, default=Args.ppl_model)
    p.add_argument("--gemini-api-key", type=str, default=os.environ.get("GEMINI_API_KEY"))
    p.add_argument("--output-dir", type=str, default=Args.output_dir)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--steps", type=int, default=Args.steps)
    p.add_argument("--eval-every", type=int, default=Args.eval_every)
    p.add_argument("--save-every", type=int, default=Args.save_every)
    p.add_argument("--bsz", type=int, default=Args.bsz)
    p.add_argument("--mini-bsz", type=int, default=Args.mini_bsz)
    p.add_argument("--gen-k", type=int, default=Args.gen_k)
    p.add_argument("--max-new-tokens", type=int, default=Args.max_new_tokens)
    p.add_argument("--temperature", type=float, default=Args.temperature)
    p.add_argument("--top-p", type=float, default=Args.top_p)
    p.add_argument("--top-k", type=int, default=Args.top_k)
    p.add_argument("--w-task", type=float, default=Args.w_task)
    p.add_argument("--w-lang", type=float, default=Args.w_lang)
    p.add_argument("--w-len", type=float, default=Args.w_len)
    p.add_argument("--algo", type=str, choices=["ppo", "grpo"], default=Args.algo)
    p.add_argument("--lr", type=float, default=Args.lr)
    p.add_argument("--lora-r", type=int, default=Args.lora_r)
    p.add_argument("--lora-alpha", type=int, default=Args.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=Args.lora_dropout)
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--no-adv", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--max-prompt-len", type=int, default=Args.max_prompt_len)
    p.add_argument("--arith-min-terms", type=int, default=Args.arith_min_terms)
    p.add_argument("--arith-max-terms", type=int, default=Args.arith_max_terms)
    a = p.parse_args()

    if a.algo == "grpo" and not HAS_GRPO:
        print("[WARN] GRPO not available in your TRL. Falling back to PPO.")
        a.algo = "ppo"

    set_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    # Build models
    model, ref_model, tok = make_policy_and_tokenizer(a)

    # Build detector
    detector = HumanDetector(ppl_model_name=a.ppl_model, use_adv=not a.no_adv, google_api_key=a.gemini_api_key)

    # Optim/trainer configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.algo == "ppo":
        ppo_cfg = PPOConfig(
            learning_rate=a.lr,
            batch_size=a.bsz,
            mini_batch_size=a.mini_bsz,
            log_with=None,
            project_kwargs=None,
            optimize_cuda_cache=True,
            target_kl=0.1,
            ppo_epochs=4,
            kl_penalty="kl",
            seed=a.seed,
        )
        trainer = PPOTrainer(
            config=ppo_cfg,
            model=model,
            ref_model=ref_model,
            tokenizer=tok,
            dataset=None,
        )
    else:
        grpo_cfg = GRPOConfig(
            learning_rate=a.lr,
            batch_size=a.bsz,
            mini_batch_size=a.mini_bsz,
            num_generations=a.gen_k,
            ppo_epochs=2,
            target_kl=0.1,
            seed=a.seed,
        )
        trainer = GRPOTrainer(
            config=grpo_cfg,
            model=model,
            ref_model=ref_model,
            tokenizer=tok,
        )

    # Arithmetic sampler
    ar = ArithConfig(min_terms=a.arith_min_terms, max_terms=a.arith_max_terms)

    def compute_reward(full_text: str, prompt_text: str, correct: int) -> Tuple[float, float, float, int, str, str]:
        think, final = parse_think_and_final(full_text)
        # Scores
        r_task = task_score(prompt_text, final, correct)
        p_h = detector.p_human(think)
        # length in tokens (rough count via tokenizer)
        len_tokens = len(tok(think).input_ids) if think else 0
        # Total
        R = a.w_task * r_task - a.w_lang * p_h - a.w_len * len_tokens
        return float(R), float(r_task), float(p_h), int(len_tokens), think, final

    # Training loop
    print(f"[INFO] Starting training with algo={a.algo}, steps={a.steps}")

    running = {"R": [], "task": [], "p_h": [], "len": []}

    for step in range(1, a.steps + 1):
        # Build a batch of prompts
        prompts = []
        golds = []
        for _ in range(a.bsz):
            ptxt, gold = sample_arith(ar)
            prompts.append(build_query(ptxt))
            golds.append(gold)

        # Generate
        gen_kwargs = dict(
            max_new_tokens=a.max_new_tokens,
            temperature=a.temperature,
            top_p=a.top_p,
            top_k=a.top_k,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

        if a.algo == "ppo":
            samples = trainer.generate(prompts, **gen_kwargs)
            texts = [tok.decode(s, skip_special_tokens=True) if isinstance(s, torch.Tensor) else s for s in samples]
            rewards = []
            reward_logs = []
            for txt, pr, gold in zip(texts, prompts, golds):
                R, r_task, p_h, lth, think, final = compute_reward(txt, pr, gold)
                rewards.append(R)
                running["R"].append(R)
                running["task"].append(r_task)
                running["p_h"].append(p_h)
                running["len"].append(lth)
                reward_logs.append((r_task, p_h, lth))
            trainer.step(prompts, texts, torch.tensor(rewards, dtype=torch.float32, device=trainer.accelerator.device))
        else:
            # GRPO expects K generations per prompt and a reward per generation
            batch_generations = []
            batch_rewards = []
            texts = []
            for pr, gold in zip(prompts, golds):
                gens = trainer.generate([pr] * a.gen_k, **gen_kwargs)
                gens = [tok.decode(g, skip_special_tokens=True) if isinstance(g, torch.Tensor) else g for g in gens]
                texts.extend(gens)
                Rs = []
                for gtxt in gens:
                    R, r_task, p_h, lth, think, final = compute_reward(gtxt, pr, gold)
                    Rs.append(R)
                    running["R"].append(R)
                    running["task"].append(r_task)
                    running["p_h"].append(p_h)
                    running["len"].append(lth)
                batch_generations.append(gens)
                batch_rewards.append(Rs)
            # Flatten inputs for GRPO step depending on API
            trainer.step(prompts, batch_generations, batch_rewards)

        if step % a.eval_every == 0:
            meanR = sum(running["R"][-100:]) / max(1, len(running["R"][-100:]))
            meanTask = sum(running["task"][-100:]) / max(1, len(running["task"][-100:]))
            meanPH = sum(running["p_h"][-100:]) / max(1, len(running["p_h"][-100:]))
            meanLen = sum(running["len"][-100:]) / max(1, len(running["len"][-100:]))
            print(f"[step {step}] R={meanR:.3f} | task={meanTask:.3f} | p_h={meanPH:.3f} | think_len={meanLen:.1f}")

            # Online adversary update: make it better at spotting current think codes
            if not a.no_adv and len(running["len"]) >= 10:
                # Collect a small buffer of recent thinks and random human sentences
                recent_thinks = []
                # We don't store thinks directly; re-generate a few quick samples instead for simplicity
                for _ in range(8):
                    ptxt, gold = sample_arith(ar)
                    q = build_query(ptxt)
                    g = trainer.generate([q], **gen_kwargs)[0]
                    gtxt = tok.decode(g, skip_special_tokens=True) if isinstance(g, torch.Tensor) else g
                    th, _ = parse_think_and_final(gtxt)
                    recent_thinks.append(th)
                detector.update(HUMAN_SEED_TEXT, recent_thinks, steps=50)

        if step % a.save_every == 0:
            # Save policy adapter (if LoRA) or full model
            out_dir = os.path.join(a.output_dir, f"step_{step}")
            os.makedirs(out_dir, exist_ok=True)
            try:
                trainer.model.save_pretrained(out_dir)
            except Exception:
                # GRPO path with plain HF model
                model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            print(f"[INFO] Saved checkpoint to: {out_dir}")

    print("[DONE]")


if __name__ == "__main__":
    main()
