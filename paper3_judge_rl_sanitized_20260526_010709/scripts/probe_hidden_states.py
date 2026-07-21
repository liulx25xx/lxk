"""
Paper 3: Hidden State Probing — Does the shortcut change internal representations?

Hypothesis: Unbalanced-trained models encode POSITION in their hidden states,
while balanced-trained models encode CONTENT QUALITY.

Method:
1. Run forward pass on test instances through both models
2. Extract hidden states at the last token before generation
3. Train linear probe: predict position (A/B) from hidden states
4. If shortcut model's hidden states are MORE predictive of position → confirms
   that RL rewired representations to encode position rather than content.

Expected result: 
- Shortcut model: position probe accuracy ~90%+ (position is encoded)
- Balanced model: position probe accuracy ~50-55% (position not encoded)
- Baseline: position probe accuracy ~55% (mild pre-existing bias)
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

os.environ.setdefault('HF_HOME', '/path/to/cache/huggingface')
os.environ.setdefault('TRITON_CACHE_DIR', '/path/to/cache/triton')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/path/to/cache/torch_inductor')
os.environ.setdefault('TMPDIR', '/path/to/cache/tmp')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def extract_hidden_states(model, tokenizer, prompts, layer_idx=-1, max_samples=200):
    """Extract hidden states from the specified layer for each prompt."""
    hidden_states_list = []
    
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    batch_size = 4
    for i in range(0, min(len(prompts), max_samples), batch_size):
        batch = prompts[i:i+batch_size]
        
        texts = []
        for prompt in batch:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get hidden states from specified layer (default: last)
        hs = outputs.hidden_states[layer_idx]  # shape: (batch, seq_len, hidden_dim)
        
        # Take the last non-padding token's hidden state
        for j in range(len(batch)):
            # Find last non-padding position
            attention_mask = inputs['attention_mask'][j]
            last_pos = attention_mask.sum() - 1
            h = hs[j, last_pos, :].cpu().float().numpy()
            hidden_states_list.append(h)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Extracted {len(hidden_states_list)}/{min(len(prompts), max_samples)} hidden states")
    
    return np.array(hidden_states_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="model", help="Label for this model in results")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--layer", type=int, default=-1, help="Layer to probe (-1 = last)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Hidden State Probing: {args.model_name} ===")
    print(f"Model: {args.model_path}")
    print(f"Adapter: {args.adapter_path or 'None'}")
    print(f"Layer: {args.layer}")
    
    # Load test data
    with open(args.test_data) as f:
        test_data = json.load(f)
    
    # Create labels: position of correct answer (always "A" in our unswapped data)
    # We'll probe whether hidden states encode "which position the model WILL select"
    # For this we use both original and swapped prompts
    swap_path = Path(args.test_data).parent / "rewardbench_test_swap.json"
    with open(swap_path) as f:
        swap_data = json.load(f)
    
    # Combine: original prompts (position A = first) and swapped (position A = second)
    n = min(args.max_samples // 2, len(test_data))
    prompts = [item["prompt"] for item in test_data[:n]]
    prompts += [item["prompt"] for item in swap_data[:n]]
    
    # Labels: 0 = original order (preferred in first position), 1 = swapped
    labels = np.array([0]*n + [1]*n)
    
    print(f"\nTotal prompts: {len(prompts)} ({n} original + {n} swapped)")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if args.adapter_path:
        print(f"Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()
    
    # Extract hidden states
    print(f"\nExtracting hidden states (layer {args.layer})...")
    hidden_states = extract_hidden_states(model, tokenizer, prompts, 
                                           layer_idx=args.layer, max_samples=len(prompts))
    
    print(f"Hidden state shape: {hidden_states.shape}")
    
    # Train linear probe: can position be predicted from hidden states?
    print("\nTraining position probe (5-fold CV)...")
    probe = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(probe, hidden_states, labels, cv=5, scoring='accuracy')
    
    position_probe_acc = scores.mean()
    position_probe_std = scores.std()
    
    print(f"\nPosition probe accuracy: {position_probe_acc:.4f} ± {position_probe_std:.4f}")
    print(f"  (50% = position not encoded, 100% = perfectly encoded)")
    
    # Also train content probe: can we predict gold label correctness?
    # For original prompts, gold = A; for swapped, gold = B
    # A model that learned content would predict correctly regardless of position
    # A model that learned position would predict "A" regardless
    
    # Save results
    results = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "layer": args.layer,
        "n_samples": len(prompts),
        "hidden_dim": hidden_states.shape[1],
        "position_probe_accuracy": float(position_probe_acc),
        "position_probe_std": float(position_probe_std),
        "position_probe_cv_scores": [float(s) for s in scores],
        "interpretation": (
            "High accuracy (>80%) = model's hidden states encode positional information, "
            "confirming the shortcut operates at the representation level. "
            "Near-chance (50-55%) = position not encoded in hidden states."
        ),
    }
    
    with open(output_dir / "probe_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'probe_results.json'}")
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model_name}")
    print(f"Position probe accuracy: {position_probe_acc:.1%} ± {position_probe_std:.1%}")
    if position_probe_acc > 0.8:
        print("→ Position is strongly encoded in hidden states (shortcut confirmed)")
    elif position_probe_acc > 0.6:
        print("→ Mild positional encoding")
    else:
        print("→ Position NOT encoded (content-based representations)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
