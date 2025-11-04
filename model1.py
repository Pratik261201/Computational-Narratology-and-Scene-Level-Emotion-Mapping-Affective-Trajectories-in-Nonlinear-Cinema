#!/usr/bin/env python3
"""
scene_emotion_extraction_model.py

Saves per-movie feature JSONs to avoid recomputing embeddings repeatedly.

Usage examples:
  python3 model1.py --preproc_dir ./processed_transcripts --out_dir ./scene_features --train_classifier --labels_csv final_label.csv
"""
import argparse
import json
from pathlib import Path
import logging
import sys
import random
from typing import List, Dict, Any, Optional
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# transformers / sentence-transformers / torch
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    S2_AVAILABLE = True
except Exception:
    S2_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Lexicon stuff
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except Exception:
    NRCLEX_AVAILABLE = False

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.utils import resample

# NLTK
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Bayesian optimization
try:
    from bayes_opt import BayesianOptimization
    BAYES_OPT_AVAILABLE = True
except Exception:
    BAYES_OPT_AVAILABLE = False
    # will warn later if user wants to run bayes

# -----------------------
# Utilities
# -----------------------
def setup_logging(level="INFO"):
    logging.basicConfig(stream=sys.stdout,
                        level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")

def list_jsons(d: Path) -> List[Path]:
    return sorted([p for p in d.glob("*.json")])

def safe_load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning(f"Failed to load {p}: {e}")
        return None

def scene_text_from_json(scene: Dict[str, Any]) -> str:
    if scene.get('dialogues'):
        parts = [d.get('text','') for d in scene.get('dialogues') if d.get('text')]
        txt = "\n".join(parts).strip()
        if txt:
            return txt
    return scene.get('raw_text','') or ""

# -----------------------
# Embedding helpers
# -----------------------
def get_sentence_transformer(model_name: str, device: str):
    if not S2_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed (pip install sentence-transformers)")
    logging.info(f"Loading SentenceTransformer model {model_name} on {device}")
    s2 = SentenceTransformer(model_name, device=device)
    try:
        # ensure inference mode
        s2.eval()
    except Exception:
        pass
    return s2

def hf_mean_pooling(model, tokenizer, texts: List[str], device: torch.device, max_len=512) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(last)
        last = last * mask
        summed = last.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / lengths
        arr = mean_pooled.cpu().numpy()
    return arr

from nltk.tokenize import sent_tokenize
def chunk_text_sentences(text: str, approx_chars: int = 900) -> List[str]:
    sents = sent_tokenize(text)
    if not sents:
        return [text[:approx_chars]]
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        cur.append(s)
        cur_len += len(s)
        if cur_len >= approx_chars:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def compute_transformer_embedding_for_scene(text: str, encoder, tokenizer_or_none, device: str, use_s2: bool, s2_batch:int=16) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        if use_s2:
            dim = encoder.get_sentence_embedding_dimension()
        else:
            dim = encoder.config.hidden_size
        return np.zeros((dim,), dtype=np.float32)

    chunks = chunk_text_sentences(text, approx_chars=1000)
    emb_chunks = []
    if use_s2:
        for i in range(0, len(chunks), s2_batch):
            batch = chunks[i:i+s2_batch]
            v = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            emb_chunks.append(v)
        emb = np.vstack(emb_chunks)
    else:
        model = encoder
        tokenizer = tokenizer_or_none
        device_t = torch.device(device)
        for i in range(0, len(chunks), 8):
            batch = chunks[i:i+8]
            v = hf_mean_pooling(model, tokenizer, batch, device_t)
            emb_chunks.append(v)
        emb = np.vstack(emb_chunks)
    mean = np.mean(emb, axis=0)
    return mean.astype(np.float32)

# -----------------------
# Lexicon features
# -----------------------
def init_vader():
    if not VADER_AVAILABLE:
        logging.warning("VADER not available - install vaderSentiment to compute VADER features.")
        return None
    return SentimentIntensityAnalyzer()

NRC_EMOTIONS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]
def compute_vader_scores(vader, text: str) -> np.ndarray:
    # Return unique: compound, pos, neg, neu (4-dim) to avoid duplicate compound entry.
    if vader is None:
        return np.zeros((4,), dtype=np.float32)
    sc = vader.polarity_scores(text or "")
    return np.array([sc.get("compound",0.0), sc.get("pos",0.0), sc.get("neg",0.0), sc.get("neu",0.0)], dtype=np.float32)

def compute_nrclex_vector(text: str) -> np.ndarray:
    if not NRCLEX_AVAILABLE:
        return np.zeros((len(NRC_EMOTIONS),), dtype=np.float32)
    try:
        n = NRCLex(text or "")
        freq = n.raw_emotion_scores
        total = sum(freq.values()) if freq else 0
        out = []
        for e in NRC_EMOTIONS:
            val = freq.get(e, 0)
            out.append(val / total if total > 0 else 0.0)
        return np.array(out, dtype=np.float32)
    except Exception as e:
        logging.warning(f"NRCLex error: {e}")
        return np.zeros((len(NRC_EMOTIONS),), dtype=np.float32)

def extract_goemotions_from_scene(scene: Dict[str, Any]) -> np.ndarray:
    probs = scene.get("goemotions_probs") or []
    if not probs:
        return np.zeros((27,), dtype=np.float32)
    arr = np.array(probs, dtype=np.float32)
    if arr.size < 27:
        arr = np.pad(arr, (0, 27-arr.size))
    elif arr.size > 27:
        arr = arr[:27]
    return arr

# -----------------------
# Scene metadata
# -----------------------
def scene_metadata_vector(scene: Dict[str, Any], scene_idx:int, num_scenes:int) -> np.ndarray:
    length = float(scene.get("scene_length_chars") or len(scene.get("raw_text","") or ""))
    nchars = float(len(scene.get("characters") or []))
    pos = scene_idx / max(1.0, num_scenes)
    return np.array([length, nchars, pos], dtype=np.float32)

# -----------------------
# Classifier and training utilities
# -----------------------
class SceneDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        # Use two hidden layers with normalization + dropout as requested.
        # Map 'hidden' to first layer size (h1). Second layer h2 set to 256 if possible.
        h1 = int(max(32, hidden))
        h2 = 256 if h1 >= 256 else max(32, h1 // 2)
        self.h1 = h1
        self.h2 = h2
        self.model = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(h1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h2, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def _oversample_balance(X_train, y_train, random_state=42):
    """
    Simple oversampling to balance classes. Returns augmented X_train, y_train.
    """
    y_vals, counts = np.unique(y_train, return_counts=True)
    max_count = int(counts.max()) if counts.size else 0
    if max_count <= 0:
        return X_train, y_train
    parts = []
    labels = []
    for v in y_vals:
        idxs = np.where(y_train == v)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) < max_count:
            # resample indices with replacement
            idx_resampled = resample(idxs, replace=True, n_samples=max_count, random_state=random_state)
        else:
            idx_resampled = idxs
        parts.append(X_train[idx_resampled])
        labels.append(y_train[idx_resampled])
    X_bal = np.vstack(parts)
    y_bal = np.concatenate(labels)
    # shuffle
    perm = np.random.RandomState(random_state).permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]

def train_classifier_torch(X: np.ndarray, y: np.ndarray, input_dim:int, num_classes:int, device: str,
                           epochs:int=10, batch_size:int=64, lr:float=2e-4, hidden:int=512, val_split:float=0.15,
                           reduce_lr_patience:int=3, early_stop_patience:int=10, dropout:float=0.2):
    """
    Train classifier with:
     - stratified split for classes with >=2 examples
     - singleton classes (only 1 example) kept in training set
     - class-weighted CrossEntropyLoss (balanced) with label smoothing
     - ReduceLROnPlateau and CosineAnnealingLR (if available)
     - simple oversampling balance before training
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available for training the classifier")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_enc = np.array(y_enc)
    X = np.array(X)

    # create validation split using stratify on multi-class entries (existing logic)
    counts = Counter(y_enc.tolist())
    multi_indices = [i for i, lab in enumerate(y_enc) if counts[lab] >= 2]
    singleton_indices = [i for i, lab in enumerate(y_enc) if counts[lab] == 1]

    tr_idx_multi = []
    val_idx_multi = []
    if len(multi_indices) > 0:
        y_multi = y_enc[multi_indices]
        rel_indices = np.arange(len(multi_indices))
        tr_rel, val_rel = train_test_split(rel_indices, test_size=val_split, stratify=y_multi, random_state=42)
        tr_idx_multi = [multi_indices[i] for i in tr_rel]
        val_idx_multi = [multi_indices[i] for i in val_rel]

    train_indices = sorted(tr_idx_multi + singleton_indices)
    val_indices = sorted(val_idx_multi)

    if len(train_indices) == 0:
        raise RuntimeError("No training examples after split; check labels.")

    X_train = X[train_indices]
    y_train = y_enc[train_indices]
    if len(val_indices) > 0:
        X_val = X[val_indices]
        y_val = y_enc[val_indices]
    else:
        X_val = X_train
        y_val = y_train

    # Oversample balance the training set to reduce class imbalance
    try:
        X_train_bal, y_train_bal = _oversample_balance(X_train, y_train, random_state=42)
        X_train, y_train = X_train_bal, y_train_bal
    except Exception as e:
        logging.warning(f"Oversampling failed: {e}; proceeding without oversampling.")

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    n_total_classes = len(le.classes_)
    weight_vec = np.ones((n_total_classes,), dtype=np.float32)
    for ci, cls in enumerate(classes):
        weight_vec[int(cls)] = float(class_weights[list(classes).index(cls)])
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    model = MLPClassifier(input_dim, hidden, len(le.classes_), dropout=dropout).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    weight_t = torch.tensor(weight_vec, dtype=torch.float32, device=device_t)

    # Try label smoothing if PyTorch supports it, otherwise fallback without smoothing.
    try:
        loss_fn = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=0.1)
    except TypeError:
        # older PyTorch versions
        loss_fn = nn.CrossEntropyLoss(weight=weight_t)

    ds_train = SceneDataset(X_train, y_train)
    ds_val = SceneDataset(X_val, y_val)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # Scheduler: use ReduceLROnPlateau and CosineAnnealingLR if available
    scheduler_plateau = None
    try:
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=reduce_lr_patience)
    except Exception:
        scheduler_plateau = None

    scheduler_cos = None
    try:
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    except Exception:
        scheduler_cos = None

    best_val_acc = -1.0
    best_state = None
    epochs_since_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in dl_train:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / max(1, len(ds_train))

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device_t)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(yb.numpy())
        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            val_acc = float((all_preds == all_labels).mean())
        else:
            val_acc = 0.0

        logging.info(f"Epoch {ep}/{epochs} train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        # step cosine scheduler every epoch if present
        if scheduler_cos is not None:
            try:
                scheduler_cos.step()
            except Exception:
                pass

        # scheduler step (plateau) with validation metric
        if scheduler_plateau is not None:
            try:
                scheduler_plateau.step(val_acc)
            except Exception:
                pass

        if val_acc > best_val_acc + 1e-8:
            best_val_acc = val_acc
            best_state = model.state_dict()
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        # early stopping
        if epochs_since_improve >= early_stop_patience:
            logging.info(f"No improvement for {epochs_since_improve} epochs; early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, le, (X_val, y_val)

# -----------------------
# Temperature scaling
# -----------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    def forward(self, logits: torch.Tensor):
        return logits / self.temperature

def fit_temperature_scaling(logits_val: np.ndarray, labels_val: np.ndarray, device:str='cpu', max_iter=200):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for temperature scaling")
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    logits = torch.tensor(logits_val, dtype=torch.float32, device=device_t)
    labels = torch.tensor(labels_val, dtype=torch.long, device=device_t)
    temp = TemperatureScaler().to(device_t)
    optimizer = torch.optim.LBFGS([temp.temperature], lr=0.01, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        scaled = temp(logits)
        loss = nll(scaled, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(temp.temperature.detach().cpu().numpy())

# -----------------------
# Bayesian hyperopt helper
# -----------------------
def run_bayesian_hyperopt(X_labeled, y_labeled, input_dim, device_for_train,
                          init_points=8, n_iter=50, random_state=42):
    """
    Run Bayesian optimization to tune MLP hyperparameters for train_classifier_torch.
    Returns a dict of best hyperparameters converted to python types or None.
    """
    if not BAYES_OPT_AVAILABLE:
        logging.warning("bayes_opt not installed; skipping Bayesian hyperparameter search.")
        return None

    def bayes_opt_objective(lr, hidden, dropout, batch_size):
        lr_f = float(lr)
        hidden_i = int(max(16, round(hidden)))
        dropout_f = float(dropout)
        batch_i = int(max(8, round(batch_size)))

        # guard: hidden must be reasonable
        if hidden_i <= 0 or batch_i <= 0:
            return 0.0

        try:
            # small quick training to evaluate configuration
            model_tmp, label_encoder_tmp, val_hold_tmp = train_classifier_torch(
                X_labeled, y_labeled,
                input_dim=input_dim,
                num_classes=len(np.unique(y_labeled)),
                device=device_for_train,
                epochs=15,
                batch_size=batch_i,
                lr=lr_f,
                hidden=hidden_i,
                val_split=0.15,
                reduce_lr_patience=2,
                early_stop_patience=5,
                dropout=dropout_f
            )
            X_val_tmp, y_val_tmp = val_hold_tmp
            val_acc = 0.0
            if TORCH_AVAILABLE:
                dev = torch.device(device_for_train)
                model_tmp.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val_tmp.astype(np.float32)).to(dev)
                    logits_val = model_tmp(X_val_t).cpu().numpy()
                    if logits_val.ndim == 1 or (logits_val.ndim > 1 and logits_val.shape[1] == 1):
                        preds = (logits_val.squeeze() > 0.0).astype(int)
                    else:
                        preds = np.argmax(logits_val, axis=1)
                    y_val_arr = np.array(y_val_tmp)
                    if preds.shape[0] == y_val_arr.shape[0]:
                        val_acc = float((preds == y_val_arr).mean())
                    else:
                        val_acc = 0.0
            else:
                val_acc = 0.0
        except Exception as e:
            logging.warning(f"Bayes objective failed for lr={lr_f},hidden={hidden_i},dropout={dropout_f},batch={batch_i}: {e}")
            val_acc = 0.0

        return float(val_acc)

    pbounds = {
        'lr': (1e-5, 1e-2),
        'hidden': (64, 2048),
        'dropout': (0.0, 0.5),
        'batch_size': (16, 256)
    }

    optimizer = BayesianOptimization(
        f=bayes_opt_objective,
        pbounds=pbounds,
        random_state=random_state,
        verbose=2
    )

    logging.info(f"Starting Bayesian optimization (init_points={init_points}, n_iter={n_iter})")
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    if len(optimizer.res) == 0:
        logging.warning("Bayesian optimization produced no results.")
        return None

    best = optimizer.max['params']
    best_converted = {
        'lr': float(best['lr']),
        'hidden': int(round(best['hidden'])),
        'dropout': float(best['dropout']),
        'batch_size': int(round(best['batch_size']))
    }
    logging.info(f"Bayes best params: {best_converted}, best_target={optimizer.max['target']}")
    return best_converted

# -----------------------
# Main runner
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Scene Emotion Extraction -> per-scene vectors (with per-movie caching)")
    p.add_argument("--preproc_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--embed_model", type=str, default="all-roberta-large-v1", help="SentenceTransformer model name or HF model name")
    p.add_argument("--use_hf", action="store_true", help="Use HuggingFace AutoModel+tokenizer pooling instead of sentence-transformers")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--train_classifier", action="store_true")
    p.add_argument("--labels_csv", type=Path, default=None, help="CSV with columns: source_file,scene_id,label")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--force_recompute_features", action="store_true", help="Force recompute embeddings and overwrite per-movie feature JSONs")
    args = p.parse_args()

    setup_logging(args.log_level)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
        if args.use_gpu and torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
    else:
        device_str = "cpu"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    features_json_dir = out_dir / "scene_jsons"
    features_json_dir.mkdir(parents=True, exist_ok=True)

    preproc_dir = args.preproc_dir
    json_files = list_jsons(preproc_dir)
    if not json_files:
        logging.error(f"No preprocessed JSON files found in {preproc_dir}")
        return

    # prepare encoders
    if args.use_hf:
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed, cannot use HF encoding")
        tokenizer = AutoTokenizer.from_pretrained(args.embed_model, use_fast=True)
        model = AutoModel.from_pretrained(args.embed_model)
        if args.use_gpu and torch.cuda.is_available():
            model = model.to("cuda")
        try:
            model.eval()
        except Exception:
            pass
        use_s2 = False
        s2_encoder = model
        tokenizer_obj = tokenizer
    else:
        s2_encoder = get_sentence_transformer(args.embed_model, device="cuda" if args.use_gpu else "cpu")
        tokenizer_obj = None
        use_s2 = True

    vader = init_vader()
    logging.info(f"VADER available: {VADER_AVAILABLE}, NRCLex available: {NRCLEX_AVAILABLE}")

    # Optionally try to load cached global npz to avoid recompute (but prefer per-movie JSONs)
    features_npz = out_dir / "scene_features.npz"
    index_rows = []
    embeddings = []
    lexicon_vectors = []
    metadata_vectors = []

    # If global npz exists and user didn't force recompute, load it (quick)
    if features_npz.exists() and not args.force_recompute_features:
        try:
            logging.info(f"Loading cached global features {features_npz}")
            data = np.load(features_npz, allow_pickle=True)
            embeddings = data['embeddings']
            lexicon_vectors = data['lexicon']
            metadata_vectors = data['metadata']
            idx_obj = data['index'].tolist() if hasattr(data['index'], "tolist") else data['index']
            if isinstance(idx_obj, dict):
                idx_df = pd.DataFrame(idx_obj)
                index_rows = idx_df.to_dict(orient="records")
            else:
                try:
                    index_rows = list(idx_obj)
                except Exception:
                    index_rows = []
            logging.info(f"Loaded cached arrays: embeddings {getattr(embeddings,'shape',None)}, lexicon {getattr(lexicon_vectors,'shape',None)}, metadata {getattr(metadata_vectors,'shape',None)}")
        except Exception as e:
            logging.warning(f"Failed to load cached npz: {e}")
            embeddings = []
            lexicon_vectors = []
            metadata_vectors = []
            index_rows = []

    # If we didn't load a global cache, build arrays by per-movie feature JSON files (or compute and save them)
    if len(embeddings) == 0:
        for jpath in tqdm(json_files, desc="Processing movies (with per-movie caching)"):
            movie_stem = jpath.stem
            feature_file = features_json_dir / f"{movie_stem}.features.json"
            if feature_file.exists() and not args.force_recompute_features:
                try:
                    # load cached per-movie features
                    with feature_file.open("r", encoding="utf-8") as f:
                        movie_obj = json.load(f)
                    for s in movie_obj.get("scenes", []):
                        embeddings.append(np.array(s["embedding"], dtype=np.float32))
                        lexicon_vectors.append(np.array(s["lexicon"], dtype=np.float32))
                        metadata_vectors.append(np.array(s["metadata"], dtype=np.float32))
                        index_rows.append({
                            "source_file": movie_obj.get("source_file"),
                            "title": movie_obj.get("title"),
                            "year": movie_obj.get("year"),
                            "scene_id": s.get("scene_id"),
                            "json_path": str(jpath)
                        })
                    continue
                except Exception as e:
                    logging.warning(f"Failed to load per-movie feature {feature_file}: {e}. Recomputing.")

            # compute features for this movie and save per-movie JSON
            jd = safe_load_json(jpath)
            if jd is None:
                continue
            movie_scenes = []
            scenes = jd.get("scenes") or []
            num_scenes = len(scenes)
            for s in scenes:
                sid = int(s.get("scene_id") or 0)
                txt = scene_text_from_json(s)
                emb = compute_transformer_embedding_for_scene(txt, s2_encoder, tokenizer_obj, device_str, use_s2, s2_batch=args.batch_size)
                vader_vec = compute_vader_scores(vader, txt)
                nrc_vec = compute_nrclex_vector(txt)
                go_vec = extract_goemotions_from_scene(s)
                lex = np.concatenate([vader_vec, nrc_vec, go_vec], axis=0)
                meta = scene_metadata_vector(s, sid, num_scenes)
                embeddings.append(emb)
                lexicon_vectors.append(lex)
                metadata_vectors.append(meta)
                index_rows.append({
                    "source_file": jpath.name,
                    "title": jd.get("title"),
                    "year": jd.get("year"),
                    "scene_id": sid,
                    "json_path": str(jpath)
                })
                movie_scenes.append({
                    "scene_id": sid,
                    "heading": s.get("heading"),
                    "scene_text": s.get("raw_text",""),
                    "embedding": emb.tolist(),
                    "lexicon": lex.tolist(),
                    "metadata": meta.tolist(),
                    "goemotions": go_vec.tolist()
                })
            # write per-movie features JSON (atomic)
            movie_obj = {
                "source_file": jpath.name,
                "title": jd.get("title"),
                "year": jd.get("year"),
                "scenes": movie_scenes
            }
            try:
                tmp = features_json_dir / f"{movie_stem}.features.json.tmp"
                with tmp.open("w", encoding="utf-8") as fo:
                    json.dump(movie_obj, fo, ensure_ascii=False)
                tmp.replace(feature_file)
                logging.info(f"Wrote per-movie feature JSON: {feature_file}")
            except Exception as e:
                logging.warning(f"Failed to write per-movie feature JSON {feature_file}: {e}")

        # convert lists to arrays
        try:
            embeddings = np.vstack([np.asarray(v) for v in embeddings]) if len(embeddings)>0 else np.zeros((0,0), dtype=np.float32)
        except Exception as e:
            logging.warning(f"Failed vstack embeddings: {e}")
            embeddings = np.zeros((0,0), dtype=np.float32)
        try:
            lexicon_vectors = np.vstack([np.asarray(v) for v in lexicon_vectors]) if len(lexicon_vectors)>0 else np.zeros((0,0), dtype=np.float32)
        except Exception as e:
            logging.warning(f"Failed vstack lexicon_vectors: {e}")
            lexicon_vectors = np.zeros((0,0), dtype=np.float32)
        try:
            metadata_vectors = np.vstack([np.asarray(v) for v in metadata_vectors]) if len(metadata_vectors)>0 else np.zeros((0,0), dtype=np.float32)
        except Exception as e:
            logging.warning(f"Failed vstack metadata_vectors: {e}")
            metadata_vectors = np.zeros((0,0), dtype=np.float32)

        # Save a cached global npz (lightweight) for even faster subsequent runs
        try:
            index_df_tmp = pd.DataFrame(index_rows)
            np.savez_compressed(out_dir / "scene_features.cache.npz",
                                embeddings=embeddings, lexicon=lexicon_vectors, metadata=metadata_vectors,
                                index=index_df_tmp.to_dict(orient="list"))
            logging.info("Saved intermediate cache scene_features.cache.npz")
        except Exception as e:
            logging.warning(f"Could not save intermediate cache: {e}")

    logging.info(f"Built arrays: embeddings {getattr(embeddings,'shape',None)}, lexicon {getattr(lexicon_vectors,'shape',None)}, metadata {getattr(metadata_vectors,'shape',None)}")

    # Standardize
    # Determine number of rows from available arrays to create zero-width defaults where needed
    candidate_row_counts = []
    for arr in (embeddings, lexicon_vectors, metadata_vectors):
        if isinstance(arr, np.ndarray) and arr.size and arr.ndim == 2:
            candidate_row_counts.append(arr.shape[0])
    n_rows = max(candidate_row_counts) if candidate_row_counts else 0

    # lexicon scaling
    if isinstance(lexicon_vectors, np.ndarray) and lexicon_vectors.size and lexicon_vectors.ndim == 2 and lexicon_vectors.shape[1] > 0:
        try:
            scaler_lex = StandardScaler().fit(lexicon_vectors)
            lex_scaled = scaler_lex.transform(lexicon_vectors)
        except Exception as e:
            logging.warning(f"Failed to scale lexicon_vectors: {e}")
            lex_scaled = lexicon_vectors.astype(np.float32)
    else:
        # produce empty width array with n_rows rows so concatenation logic works
        lex_scaled = np.zeros((n_rows, 0), dtype=np.float32)

    # metadata scaling
    if isinstance(metadata_vectors, np.ndarray) and metadata_vectors.size and metadata_vectors.ndim == 2 and metadata_vectors.shape[1] > 0:
        try:
            scaler_meta = StandardScaler().fit(metadata_vectors)
            metadata_scaled = scaler_meta.transform(metadata_vectors)
        except Exception as e:
            logging.warning(f"Failed to scale metadata_vectors: {e}")
            metadata_scaled = metadata_vectors.astype(np.float32)
    else:
        metadata_scaled = np.zeros((n_rows, 0), dtype=np.float32)

    classifier_vectors = np.zeros((embeddings.shape[0], 0), dtype=np.float32) if (isinstance(embeddings, np.ndarray) and embeddings.size) else np.zeros((n_rows, 0), dtype=np.float32)

    # ============================
    # Label merging: rare classes -> 'other'
    # ============================
    MERGE_TO_OTHER = {
        # left empty intentionally (same behavior as before)

    }

    # If training classifier head:
    label_list = None
    if args.train_classifier:
        scene_labels_map = {}
        if args.labels_csv and args.labels_csv.exists():
            df_labels = pd.read_csv(args.labels_csv)
            # apply mapping and build map
            for _, row in df_labels.iterrows():
                raw_label = str(row['label']).strip()
                label = raw_label.lower()
                if label in MERGE_TO_OTHER:
                    mapped = "other"
                else:
                    mapped = label
                key = f"{row['source_file']}:{int(row['scene_id'])}"
                scene_labels_map[key] = mapped

            # log counts after mapping
            counts = Counter(scene_labels_map.values())
            logging.info(f"Loaded labels CSV {args.labels_csv}; label counts (after merging rare -> 'other'):")
            for lab, c in counts.most_common():
                logging.info(f"  {lab}: {c}")

        keys = [f"{r['source_file']}:{int(r['scene_id'])}" for r in index_rows]
        y = []
        available_idx = []
        for idx, k in enumerate(keys):
            if k in scene_labels_map:
                y.append(scene_labels_map[k])
                available_idx.append(idx)
        if len(available_idx) == 0:
            logging.error("train_classifier requested but no scene-level labels found (provide --labels_csv). Skipping training.")
        else:
            # ensure embeddings/lex_scaled/metadata_scaled align for building X_full
            # compute X_full using only arrays that exist and have matching rows
            try:
                X_full = np.concatenate([embeddings, lex_scaled, metadata_scaled], axis=1)
            except Exception as e:
                logging.error(f"Failed to concatenate arrays to build X_full for training: {e}")
                return
            X_labeled = X_full[available_idx]
            y_labeled = np.array(y)
            logging.info(f"Training classifier on {len(y_labeled)} labeled scenes (input_dim={X_labeled.shape[1]})")

            # Bayesian optimization: only run if bayes_opt installed and have enough labeled samples
            device_for_train = "cuda" if (args.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
            best_hp = None
            if BAYES_OPT_AVAILABLE and len(y_labeled) >= 30:
                logging.info("Running Bayesian hyperparameter optimization for classifier (this may take some time)...")
                try:
                    best_hp = run_bayesian_hyperopt(X_labeled, y_labeled, input_dim=X_labeled.shape[1],
                                                    device_for_train=device_for_train,
                                                    init_points=8, n_iter=50, random_state=42)
                except Exception as e:
                    logging.warning(f"Bayesian optimization failed or was interrupted: {e}")
                    best_hp = None
            else:
                if not BAYES_OPT_AVAILABLE:
                    logging.info("bayes_opt not available; skipping Bayesian optimization.")
                else:
                    logging.info(f"Not enough labeled data for Bayesian tuning (have {len(y_labeled)}; need >=30). Skipping tuning.")

            # choose hyperparameters
            if best_hp is not None:
                chosen_hidden = best_hp.get('hidden', 512)
                chosen_lr = best_hp.get('lr', args.lr)
                chosen_batch = best_hp.get('batch_size', 64)
                chosen_dropout = best_hp.get('dropout', 0.2)
            else:
                chosen_hidden = 512
                chosen_lr = args.lr
                chosen_batch = 64
                chosen_dropout = 0.2

            logging.info(f"Training final classifier with hidden={chosen_hidden}, lr={chosen_lr}, batch={chosen_batch}, dropout={chosen_dropout}")
            model, label_encoder, val_hold = train_classifier_torch(
                X_labeled, y_labeled,
                input_dim=X_labeled.shape[1],
                num_classes=len(np.unique(y_labeled)),
                device=device_for_train,
                epochs=args.epochs,
                batch_size=chosen_batch,
                lr=chosen_lr,
                hidden=chosen_hidden,
                dropout=chosen_dropout,
                early_stop_patience=10
            )

            model.eval()
            if TORCH_AVAILABLE:
                dev = torch.device(device_for_train)
                with torch.no_grad():
                    all_X = torch.tensor(X_full.astype(np.float32)).to(dev)
                    logits = model(all_X).cpu().numpy()
                classifier_vectors = logits.astype(np.float32)
                label_list = list(label_encoder.classes_)
                ckpt_path = out_dir / "classifier_checkpoint.pt"
                torch.save({'state_dict': model.state_dict(), 'label_classes': label_list}, str(ckpt_path))
                logging.info(f"Saved classifier checkpoint to {ckpt_path}")
                X_val, y_val = val_hold
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val.astype(np.float32)).to(dev)
                    logits_val = model(X_val_t).cpu().numpy()
                try:
                    temp = fit_temperature_scaling(logits_val, y_val, device=device_for_train)
                    logging.info(f"Fitted temperature scaling: T={temp:.4f}")
                    np.save(out_dir / "temp_scaler.npy", np.array([temp], dtype=np.float32))
                except Exception as e:
                    logging.warning(f"Temperature scaling failed: {e}")

                # Print classification report on validation set for per-class metrics
                try:
                    if logits_val.ndim == 1 or (logits_val.ndim > 1 and logits_val.shape[1] == 1):
                        preds_val = (logits_val.squeeze() > 0.0).astype(int)
                    else:
                        preds_val = np.argmax(logits_val, axis=1)
                    logging.info("Validation classification report:\n" + classification_report(y_val, preds_val, zero_division=0))
                except Exception as e:
                    logging.warning(f"Could not build classification report: {e}")
            else:
                logging.warning("PyTorch not available; cannot compute classifier logits or calibration.")
    else:
        logging.info("Classifier training not requested; classifier vector will remain empty.")

    # Final scene vector:
    # Build parts only from non-empty arrays that match row count n_rows (skip mismatches)
    parts = []
    # recompute n_rows from current arrays to be safe
    candidate_row_counts = []
    for arr in (embeddings, classifier_vectors, lex_scaled, metadata_scaled):
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size:
            candidate_row_counts.append(arr.shape[0])
    n_rows_final = max(candidate_row_counts) if candidate_row_counts else 0

    for name, arr in (("embeddings", embeddings), ("classifier_vectors", classifier_vectors),
                      ("lex_scaled", lex_scaled), ("metadata_scaled", metadata_scaled)):
        if not (isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size):
            logging.debug(f"Skipping {name} — empty or not 2D")
            continue
        if arr.shape[0] != n_rows_final:
            logging.warning(f"Skipping {name} due to row mismatch: {arr.shape[0]} != {n_rows_final}")
            continue
        parts.append(arr.astype(np.float32))

    if len(parts) == 0:
        X_final = np.zeros((0,0), dtype=np.float32)
    else:
        try:
            X_final = np.concatenate(parts, axis=1)
        except Exception as e:
            logging.error(f"Failed to concatenate final parts into X_final: {e}")
            # fallback to safe shape
            X_final = np.zeros((n_rows_final, 0), dtype=np.float32)

    logging.info(f"Final scene-feature matrix shape: {X_final.shape}")

    # Save features and index
    index_df = pd.DataFrame(index_rows)
    index_csv = out_dir / "scene_index.csv"
    index_df.to_csv(index_csv, index=False)
    try:
        np.savez_compressed(out_dir / "scene_features.npz", X=X_final, embeddings=embeddings, lexicon=lexicon_vectors, metadata=metadata_vectors, index=index_df.to_dict(orient="list"))
        logging.info(f"Saved features to {out_dir/'scene_features.npz'} and index to {index_csv}")
    except Exception as e:
        logging.warning(f"Failed to save final features: {e}")

    logging.info("Done.")

if __name__ == "__main__":
    main()

