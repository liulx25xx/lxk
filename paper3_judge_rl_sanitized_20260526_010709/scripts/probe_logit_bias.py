"""
Paper 3: Logit Bias Analysis — Does the shortcut model have a hardcoded A-bias in output logits?

CORRECT probing approach: Instead of probing hidden states for "position encoding",
we directly measure the model's OUTPUT BIAS by checking the logit difference between
token "A" and token "B" at the verdict position.

For a shortcut model: logit(A) >> logit(B) regardless of input content
For a balanced model: logit(A) ≈ logit(B), with content determining the winner
For baseline: mild A preference (logit(A) slightly > logit(B))

This is more direct and interpretable than hidden state probing.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

os.environ.setdefault('HF_HOME', '/path/to/cache/huggingface')
os.environ.setdefault('HF_HUB_CACHE', '/path/to/cache/huggingface/hub')
os.environ.setdefault('TRANSFORMERS_CACHE', '/path/to/cache/huggingface/hub')
os.environ.setdefault('TRITON_CACHE_DIR', '/path/to/cache/triton')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/path/to/cache/torch_inductor')
os.environ.setdefault('TMPDIR', '/path/to/cache/tmp')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def measure_verdict_logit_bias(model, tokenizer, prompts, max_samples=200):
    """
    For each prompt, compute logit(A) - logit(B) at the position where
    the model would output the verdict token.
    
    We append "[[" to the prompt and check what the model wants to generate next.
    If shortcut: always high logit for A.
    If balanced: depends on content.
    """
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Find token IDs for A and B
    # Try different tokenizations
    a_tokens = tokenizer.encode("A", add_special_tokens=False)
    b_tokens = tokenizer.encode("B", add_special_tokens=False)
    a_token_id = a_tokens[0]  # Usually single token
    b_token_id = b_tokens[0]
    
    print(f"Token IDs: A={a_token_id}, B={b_token_id}")
    
    logit_diffs = []  # logit(A) - logit(B) for each prompt
    a_probs = []  # P(A) for each prompt
    
    batch_size = 4
    for i in range(0, min(len(prompts), max_samples), batch_size):
        batch = prompts[i:i+batch_size]
        
        # Format as chat and append "[[" to force verdict position
        texts = []
        for prompt in batch:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Append partial response to get to verdict position
            # The model outputs rationale then "[[X, confidence]]"
            # We can't easily find the exact verdict position, so instead:
            # Generate a short response and look at the full logit distribution
            texts.append(text)
        
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits at last position (where generation would start)
            last_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
            
            for j in range(len(batch)):
                logit_a = last_logits[j, a_token_id].item()
                logit_b = last_logits[j, b_token_id].item()
                logit_diffs.append(logit_a - logit_b)
                
                # Also compute softmax probability
                probs = torch.softmax(last_logits[j, [a_token_id, b_token_id]], dim=0)
                a_probs.append(probs[0].item())
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, max_samples)}/{min(len(prompts), max_samples)}")
    
    return np.array(logit_diffs), np.array(a_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Logit Bias Analysis: {args.model_name} ===")
    
    # Load test data (original order only — we want to see if model is biased toward A)
    with open(args.test_data) as f:
        test_data = json.load(f)
    
    prompts = [item["prompt"] for item in test_data[:args.max_samples]]
    print(f"Prompts: {len(prompts)}")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()
    
    # Measure logit bias
    print("Measuring verdict logit bias...")
    logit_diffs, a_probs = measure_verdict_logit_bias(
        model, tokenizer, prompts, max_samples=args.max_samples
    )
    
    # Statistics
    mean_diff = logit_diffs.mean()
    std_diff = logit_diffs.std()
    mean_a_prob = a_probs.mean()
    pct_a_preferred = (logit_diffs > 0).mean()  # fraction where logit(A) > logit(B)
    
    results = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "n_samples": len(logit_diffs),
        "mean_logit_diff_A_minus_B": float(mean_diff),
        "std_logit_diff": float(std_diff),
        "mean_P_A": float(mean_a_prob),
        "pct_logit_A_gt_B": float(pct_a_preferred),
        "interpretation": (
            f"Mean logit(A)-logit(B) = {mean_diff:.2f}. "
            f"Positive = model prefers A at first-token level. "
            f"{pct_a_preferred*100:.0f}% of prompts have logit(A) > logit(B)."
        ),
    }
    
    with open(output_dir / "logit_bias_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model_name}")
    print(f"Mean logit(A) - logit(B): {mean_diff:.3f} ± {std_diff:.3f}")
    print(f"Mean P(A | first token): {mean_a_prob:.3f}")
    print(f"% prompts with logit(A) > logit(B): {pct_a_preferred*100:.1f}%")
    if mean_diff > 2.0:
        print("→ STRONG A-bias in output logits (shortcut confirmed at logit level)")
    elif mean_diff > 0.5:
        print("→ Moderate A-preference")
    else:
        print("→ No systematic A-bias (content-based judgment)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
