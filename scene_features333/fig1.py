#!/usr/bin/env python3
"""
Generate probs.npy and labels.npy for reliability plotting.

Usage examples:
  # Top-confidence calibration (prob = max softmax, label = correct?)
  python generate_probs_labels.py \
    --features scene_features.npz \
    --index scene_index.csv \
    --preproc preproc.pkl \
    --ckpt classifier_checkpoint.pt \
    --labels_csv ./labels_for_training.csv \
    --out_dir ./calib_out \
    --mode topconf

  # Per-class calibration for class 'curiosity'
  python generate_probs_labels.py \
    --features scene_features.npz \
    --index scene_index.csv \
    --preproc preproc.pkl \
    --ckpt classifier_checkpoint.pt \
    --labels_csv ./labels_for_training.csv \
    --out_dir ./calib_out \
    --mode perclass \
    --class_name curiosity
"""
import argparse
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

def load_features_npz(path):
    data = np.load(path, allow_pickle=True)
    # prefer 'X' if present
    if 'X' in data:
        X = data['X']
    else:
        # try to reconstruct from embeddings/lexicon/metadata
        parts = []
        if 'embeddings' in data:
            parts.append(np.asarray(data['embeddings']))
        if 'lexicon' in data:
            parts.append(np.asarray(data['lexicon']))
        if 'metadata' in data:
            parts.append(np.asarray(data['metadata']))
        if len(parts) == 0:
            raise RuntimeError(f"No usable arrays found in {path}. Keys: {list(data.keys())}")
        # align on rows by taking min rows if mismatch
        shapes0 = [p.shape[0] for p in parts if isinstance(p, np.ndarray) and p.ndim == 2]
        nrows = min(shapes0) if shapes0 else parts[0].shape[0]
        parts_trim = [p[:nrows] if (isinstance(p, np.ndarray) and p.shape[0] >= nrows) else p for p in parts]
        X = np.concatenate(parts_trim, axis=1)
    # index info if present
    index_df = None
    if 'index' in data:
        try:
            idx_obj = data['index'].item() if isinstance(data['index'], np.ndarray) and data['index'].dtype == object else data['index']
            index_df = pd.DataFrame(idx_obj)
        except Exception:
            try:
                index_df = pd.DataFrame(dict(data['index'].tolist()))
            except Exception:
                index_df = None
    return X, index_df, list(data.keys())

def build_model_from_state(state_dict):
    """
    Build a Sequential model that matches the 'model' parts in state_dict created by MLPClassifier.
    We detect shapes for layers:
      - model.0.weight -> (h1, input_dim)
      - model.4.weight -> (h2, h1)
      - model.7.weight -> (num_classes, h2)
    Also detect LayerNorm parameters (model.3.*)
    """
    # find keys starting with 'model.'
    model_keys = [k for k in state_dict.keys() if k.startswith('model.')]
    # helper to get tensor shape
    def s(k):
        return state_dict[k].shape
    # identify first linear weight 'model.0.weight'
    if 'model.0.weight' not in state_dict:
        raise RuntimeError("Cannot find 'model.0.weight' in state_dict; unknown architecture.")
    h1, input_dim = state_dict['model.0.weight'].shape
    # find second linear weight (likely 'model.4.weight')
    # find all weight keys and pick the ones with 2 dims
    weight_keys = [k for k in model_keys if k.endswith('.weight')]
    # sort for deterministic behavior
    weight_keys = sorted(weight_keys)
    # guess next linear is the one whose shape[1] == h1 and shape[0] != h1
    candidate = None
    for k in weight_keys:
        sh = state_dict[k].shape
        if sh[1] == h1 and sh[0] != h1:
            candidate = k
            break
    if candidate is None:
        # fallback: pick the weight key after model.0.weight in sorted list
        idx = weight_keys.index('model.0.weight')
        candidate = weight_keys[idx+1] if idx+1 < len(weight_keys) else None
    if candidate is None:
        raise RuntimeError("Cannot find second linear weight key in state_dict.")
    h2 = state_dict[candidate].shape[0]
    # final linear weight: find weight with shape[1] == h2
    final_key = None
    for k in weight_keys:
        sh = state_dict[k].shape
        if sh[1] == h2 and sh[0] != h2:
            final_key = k
            break
    if final_key is None:
        # as fallback pick last weight key
        final_key = weight_keys[-1]
    num_classes = state_dict[final_key].shape[0]

    # Now construct Sequential matching the original MLPClassifier
    layers = []
    layers.append(nn.Linear(input_dim, h1))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.3))
    layers.append(nn.LayerNorm(h1))
    layers.append(nn.Linear(h1, h2))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(h2, num_classes))
    model = nn.Sequential(*layers)
    return model, (input_dim, h1, h2, num_classes)

def compute_ece_bins_np(probs, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    N = len(probs)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i+1])
        if mask.sum() == 0:
            continue
        p_hat = probs[mask].mean()
        y_hat = labels[mask].mean()
        ece += (mask.sum()/N) * abs(p_hat - y_hat)
    return ece

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True, help='scene_features.npz')
    p.add_argument('--index', required=True, help='scene_index.csv (the index saved by pipeline)')
    p.add_argument('--preproc', required=True, help='preproc.pkl saved by pipeline (joblib)')
    p.add_argument('--ckpt', required=True, help='classifier_checkpoint.pt (torch)')
    p.add_argument('--labels_csv', required=True, help='CSV mapping source_file,scene_id,label (used to build true labels)')
    p.add_argument('--out_dir', default='calib_out')
    p.add_argument('--mode', choices=['topconf','perclass'], default='topconf')
    p.add_argument('--class_name', default=None, help='class name for perclass mode (e.g. curiosity)')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading features...")
    X, index_df_from_npz, keys = load_features_npz(args.features)
    print(f"Features loaded: X.shape={X.shape}. npz keys: {keys}")
    # try to load index csv if provided
    index_csv_df = pd.read_csv(args.index)
    print(f"Index CSV loaded: {index_csv_df.shape[0]} rows")

    print("Loading preproc...")
    preproc = joblib.load(args.preproc)
    label_classes = preproc.get('label_classes', None)
    if label_classes is None:
        print("Warning: preproc.pkl has no 'label_classes' key; falling back to checkpoint classes later.")
    else:
        print(f"Label classes from preproc: {len(label_classes)} classes")

    print("Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    # ckpt may contain {'state_dict':..., 'label_classes':...} or be a state_dict itself
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        ckpt_label_classes = ckpt.get('label_classes', None)
    else:
        # sometimes saved as state_dict directly
        state = ckpt
        ckpt_label_classes = None
    if ckpt_label_classes is not None and label_classes is None:
        label_classes = ckpt_label_classes
    if label_classes is None:
        raise RuntimeError("Could not determine label_classes from preproc.pkl or checkpoint. Provide label_classes.")

    # build model dynamically
    print("Building model from state_dict...")
    model, dims = build_model_from_state(state)
    model.load_state_dict(state)
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    # compute logits / probs on all X
    print("Computing logits and probabilities on all feature rows...")
    X_t = torch.tensor(X.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()
    # handle binary logits shape
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        # convert to two-class probs where p = sigmoid
        probs_all = 1.0 / (1.0 + np.exp(-logits.squeeze()))
        # make shape (N,2) with [1-p, p]
        probs_full = np.vstack([1.-probs_all, probs_all]).T
        preds_all = (probs_all > 0.5).astype(int)
    else:
        # multiclass
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs_full = exp / exp.sum(axis=1, keepdims=True)
        preds_all = np.argmax(probs_full, axis=1)

    # save helpful debug arrays
    np.save(os.path.join(args.out_dir, 'logits_all.npy'), logits)
    np.save(os.path.join(args.out_dir, 'probs_all.npy'), probs_full)
    np.save(os.path.join(args.out_dir, 'preds.npy'), preds_all)

    # Build mapping from scene index rows -> label
    print("Loading ground-truth labels CSV to align true labels...")
    df_labels = pd.read_csv(args.labels_csv, dtype={'source_file':str, 'scene_id':object})
    # build key like earlier code: "source_file:scene_id"
    def make_key_row(row):
        return f"{row['source_file']}:{int(row['scene_id'])}"
    df_labels['key'] = df_labels.apply(make_key_row, axis=1)
    label_map = dict(zip(df_labels['key'], df_labels['label'].astype(str).str.lower()))

    # build keys for index csv rows (index csv likely has 'source_file' and 'scene_id' columns)
    def make_index_key(row):
        return f"{row['source_file']}:{int(row['scene_id'])}"
    index_csv_df['key'] = index_csv_df.apply(make_index_key, axis=1)

    # create arrays aligned to X rows
    keys_index = index_csv_df['key'].tolist()
    N = len(keys_index)
    true_labels = np.full((N,), -1, dtype=int)
    has_label_mask = np.zeros((N,), dtype=bool)
    for i, k in enumerate(keys_index):
        lab = label_map.get(k, None)
        if lab is not None:
            try:
                li = label_classes.index(lab)
            except ValueError:
                # try lowered match
                try:
                    li = label_classes.index(lab.lower())
                except Exception:
                    raise RuntimeError(f"Label '{lab}' (for key {k}) not found in label_classes list.")
            true_labels[i] = int(li)
            has_label_mask[i] = True

    print(f"Total rows: {N}; labeled rows found: {has_label_mask.sum()}")

    # Filter to only labeled rows
    labeled_idx = np.where(has_label_mask)[0]
    if labeled_idx.size == 0:
        raise RuntimeError("No labeled rows found: check labels_csv keys vs scene_index.csv keys")

    probs_out = None
    labels_out = None

    if args.mode == 'topconf':
        # top predicted class probability and binary correct indicator
        if probs_full.ndim == 1:
            top_probs = probs_full if probs_full.ndim==1 else probs_full.squeeze()
            preds = preds_all
        else:
            top_probs = probs_full.max(axis=1)
            preds = preds_all
        probs_out = top_probs[labeled_idx]
        labels_out = (preds[labeled_idx] == true_labels[labeled_idx]).astype(int)
        mode_note = "topconf: prob = max softmax, label = correct?"
        print(f"Mode: {mode_note}  -- computing calibration on {probs_out.size} labeled rows")
    else:
        # perclass
        if args.class_name is None:
            raise RuntimeError("perclass mode requires --class_name CLASSNAME")
        class_name = args.class_name.strip().lower()
        if class_name not in label_classes:
            raise RuntimeError(f"Class name '{class_name}' not present in label_classes")
        class_idx = label_classes.index(class_name)
        probs_out = probs_full[:, class_idx][labeled_idx]
        labels_out = (true_labels[labeled_idx] == class_idx).astype(int)
        print(f"Mode: perclass '{class_name}' (idx {class_idx}) -- computing calibration on {probs_out.size} labeled rows")

    # Save arrays
    probs_path = os.path.join(args.out_dir, 'probs.npy')
    labels_path = os.path.join(args.out_dir, 'labels.npy')
    np.save(probs_path, probs_out)
    np.save(labels_path, labels_out)
    # also save raw alignment arrays for reproducibility
    np.save(os.path.join(args.out_dir, 'probs_all_rows.npy'), probs_full if isinstance(probs_full, np.ndarray) else probs_full)
    np.save(os.path.join(args.out_dir, 'true_labels_aligned.npy'), true_labels)

    # compute simple ECE & Brier
    ece = compute_ece_bins_np(probs_out, labels_out, n_bins=15)
    brier = np.mean((probs_out - labels_out)**2)
    print(f"Saved probs -> {probs_path}; labels -> {labels_path}")
    print(f"ECE (15 bins) = {ece:.6f}; Brier = {brier:.6f}")
    print("Done.")

if __name__ == '__main__':
    main()
