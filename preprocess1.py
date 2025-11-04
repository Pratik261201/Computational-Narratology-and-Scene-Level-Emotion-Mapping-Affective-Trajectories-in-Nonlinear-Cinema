#!/usr/bin/env python3
"""
preprocess_with_goemotions_parallel_refactored.py

Refactored and optimized version of your preprocessing script with improved
parallelism for large CPU counts and an RTX GPU (A4000).

This edition has the CLI removed — all options are embedded in the CONFIG
block near the top. No other logic was changed.
"""
from pathlib import Path
import re
import json
import logging
import gc
import os
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

import numpy as np
import pandas as pd

# NLP & ML imports (optional heavy)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# slugify fallback
try:
    from slugify import slugify
except Exception:
    def slugify(x):
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', x)[:200]

# optional language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

# NRCLex (optional)
try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except Exception:
    NRCLEX_AVAILABLE = False

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from types import SimpleNamespace

# -----------------------
# CONFIG (edit values here; CLI was removed)
# -----------------------
# -----------------------
# CONFIG (edit values here; CLI was removed)
# -----------------------
from types import SimpleNamespace
from types import SimpleNamespace
CONFIG = SimpleNamespace(
    transcripts_dir=Path("./transcripts"),
    out_dir=Path("./processed_transcripts888"),
    workers=None,               # None -> use all CPUs
    limit=15000,                 # set to None to process all files
    detect_lang=False,
    use_goemotions=True,        # produce GoEmotions labels
    go_model="monologg/bert-base-cased-goemotions-original",
    go_batch=16,                # per-forward batch size inside predictor (kept small)
    go_maxlen=256,
    go_topk=3,
    go_threshold=0.40,          # higher => fewer multi-labels (higher confidence)
    use_gpu=True,               # set True if CUDA is available
    fp16=False,                 # set True if your GPU & torch support mixed precision
    dialogue_only=False,
    log_level="INFO",
    gpu_parallel=False,         # RECOMMENDED: shared GPU predictor (safer / more efficient)
    gpu_threads=8,              # threads for shared GPU predictor (tune to your machine)
    predict_batch_size=32       # larger than go_batch for better GPU throughput
)


# -----------------------

# ----------------------------
# Scene-splitting heuristics
# ----------------------------
LINES_PER_SCENE = 20
MIN_SCENE_CHARS_TO_KEEP = 40
SPEAKER_TURN_CHUNK = 12
PUNCT_CHUNK_SIZE = 2200

SCENE_HEADING_RE = re.compile(
    r'^\s*(INT|EXT|SCENE|FADE IN|FADE OUT|CUT TO|MONTAGE|CONTINUED|OMITTED|OPENING CREDITS|ACT)\b',
    flags=re.IGNORECASE
)
BRACKET_CUE_INLINE_RE = re.compile(r'(\[[^\]]+\])')
BRACKET_ONLY_RE = re.compile(r'^\s*\[.+?\]\s*$', flags=re.IGNORECASE)
SPEAKER_LINE_RE = re.compile(
    r'^\s*([A-Z0-9][A-Z0-9 \-\'\.\(\)]+?)(?:\:|\s{2,}|\s*\(V\.O\.?\)|\s*\(O\.S\.?\)|\s*\(OS\))\s*(.*)$'
)
TIMESTAMP_RE = re.compile(r'^\s*\d{1,2}:\d{2}(?::\d{2})?\s*[-–—]?', flags=re.IGNORECASE)
DASH_LINE_RE = re.compile(r'^\s*-\s+')
BRACKET_RE = re.compile(r'[\[\(].*?[\]\)]')
SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

# ----------------------------
# Utilities
# ----------------------------
def ensure_nltk():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except Exception:
        logging.info("Downloading vader_lexicon...")
        nltk.download("vader_lexicon")
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        logging.info("Downloading punkt...")
        nltk.download("punkt")


def read_metadata(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(excel_path)
    except Exception as e:
        logging.warning(f"Could not read metadata.xlsx: {e}")
        return pd.DataFrame()


def is_english_text(text: str) -> bool:
    if not LANGDETECT_AVAILABLE:
        return True
    if not text or len(text.strip()) < 100:
        return True
    try:
        return detect(text) == 'en'
    except Exception:
        return True


def clean_text(s: str) -> str:
    s = s.replace('\r', '\n')
    s = re.sub(r'\t', ' ', s)
    s = re.sub(r' +', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.lstrip('\ufeff').strip()


def surround_bracket_cues(text: str) -> str:
    text = BRACKET_CUE_INLINE_RE.sub(r'\n\1\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def split_initial(text: str) -> List[Dict[str, Any]]:
    text = surround_bracket_cues(text)
    lines = text.splitlines()
    heading_indices = []
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if not stripped:
            continue
        if SCENE_HEADING_RE.match(stripped) or BRACKET_ONLY_RE.match(stripped) or TIMESTAMP_RE.match(stripped) or DASH_LINE_RE.match(ln):
            heading_indices.append(i)
    if not heading_indices:
        return []
    heading_indices = sorted(set(heading_indices))
    scenes = []
    boundaries = heading_indices + [len(lines)]
    for idx, start in enumerate(heading_indices):
        end = boundaries[idx + 1]
        heading_line = lines[start].strip()
        scene_lines = lines[start+1:end]
        scenes.append({'heading': heading_line, 'raw_text': '\n'.join(scene_lines).strip(),
                       'lines': scene_lines, 'method': 'headings_or_brackets'})
    return scenes


def split_paragraphs(text: str) -> List[Dict[str, Any]]:
    parts = re.split(r'\n\s*\n', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return []
    return [{'heading': None, 'raw_text': p, 'lines': p.splitlines(), 'method': 'paragraphs'} for p in parts]


def split_fixed_lines(text: str, chunk_size: int) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    if not lines:
        return []
    if len(lines) <= chunk_size:
        return [{'heading': None, 'raw_text': '\n'.join(lines).strip(), 'lines': lines, 'method': 'single_or_short'}]
    scenes = []
    i = 0
    scene_id = 0
    n = len(lines)
    while i < n:
        scene_id += 1
        chunk = lines[i:i+chunk_size]
        scenes.append({'heading': f'FALLBACK_{scene_id}', 'raw_text': '\n'.join(chunk).strip(),
                       'lines': chunk, 'method': f'fixed_{chunk_size}_lines'})
        i += chunk_size
    return scenes


def split_by_speaker_turns(text: str, group_size: int) -> List[Dict[str, Any]]:
    text2 = surround_bracket_cues(text)
    lines = text2.splitlines()
    speaker_positions = []
    for i, ln in enumerate(lines):
        if SPEAKER_LINE_RE.match(ln):
            speaker_positions.append(i)
    if not speaker_positions:
        return []
    scenes = []
    current_block = []
    speaker_count = 0
    for i, ln in enumerate(lines):
        current_block.append(ln)
        if SPEAKER_LINE_RE.match(ln):
            speaker_count += 1
        if speaker_count >= group_size:
            scenes.append({'heading': None, 'raw_text': '\n'.join(current_block).strip(), 'lines': current_block.copy(), 'method': f'speaker_group_{group_size}'})
            current_block = []
            speaker_count = 0
    if current_block:
        scenes.append({'heading': None, 'raw_text': '\n'.join(current_block).strip(), 'lines': current_block, 'method': f'speaker_group_{group_size}_tail'})
    return scenes


def split_by_punctuation_chunks(text: str, chunk_chars: int) -> List[Dict[str, Any]]:
    sents = SENT_SPLIT_RE.split(text)
    if len(sents) <= 1:
        scenes = []
        i = 0
        n = len(text)
        cid = 0
        while i < n:
            cid += 1
            part = text[i:i+chunk_chars]
            scenes.append({'heading': None, 'raw_text': part.strip(), 'lines': part.splitlines(), 'method': f'punct_fallback_{chunk_chars}'})
            i += chunk_chars
        return scenes
    scenes = []
    current = []
    cur_len = 0
    cid = 0
    for sent in sents:
        if not sent.strip():
            continue
        current.append(sent)
        cur_len += len(sent)
        if cur_len >= chunk_chars:
            cid += 1
            chunk_text = ' '.join(current).strip()
            scenes.append({'heading': None, 'raw_text': chunk_text, 'lines': chunk_text.splitlines(), 'method': f'punct_chunk_{chunk_chars}'})
            current = []
            cur_len = 0
    if current:
        cid += 1
        chunk_text = ' '.join(current).strip()
        scenes.append({'heading': None, 'raw_text': chunk_text, 'lines': chunk_text.splitlines(), 'method': f'punct_chunk_{chunk_chars}_tail'})
    return scenes


def extract_dialogues_and_actions(scene_lines: List[str]) -> Dict[str, Any]:
    dialogues = []
    actions = []
    characters = set()
    i = 0
    n = len(scene_lines)
    while i < n:
        ln = scene_lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue
        m = SPEAKER_LINE_RE.match(ln)
        if m:
            speaker_raw = m.group(1).strip()
            first_line = m.group(2).strip()
            speaker = normalize_speaker(speaker_raw)
            utter_lines = []
            if first_line:
                utter_lines.append(first_line)
            j = i + 1
            while j < n:
                next_ln = scene_lines[j].rstrip()
                if not next_ln.strip():
                    j += 1
                    continue
                if SPEAKER_LINE_RE.match(next_ln):
                    break
                if next_ln.strip().isupper() and len(next_ln.strip().split()) <= 6:
                    break
                utter_lines.append(next_ln.strip())
                j += 1
            text = ' '.join(utter_lines).strip()
            text = BRACKET_RE.sub('', text).strip()
            if not text:
                text = first_line
            dialogues.append({'speaker': speaker, 'text': text})
            characters.add(speaker)
            i = j
            continue
        j = i
        block = []
        while j < n:
            line_j = scene_lines[j].rstrip()
            if not line_j.strip():
                if block:
                    break
                j += 1
                continue
            if SPEAKER_LINE_RE.match(line_j):
                break
            block.append(line_j.strip())
            j += 1
        action_text = ' '.join(block)
        action_text = BRACKET_RE.sub('', action_text).strip()
        if action_text:
            actions.append(action_text)
        i = j
    return {'dialogues': dialogues, 'actions': actions, 'characters': sorted(list(characters))}


def normalize_speaker(name: str) -> str:
    if not name:
        return ''
    name = BRACKET_RE.sub('', name)
    name = name.strip().strip(':').strip().upper()
    name = re.sub(r'\.+', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def merge_empty_and_short_scenes(processed_scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not processed_scenes:
        return processed_scenes
    merged = []
    for s in processed_scenes:
        length = s.get('scene_length_chars', len(s.get('raw_text','')))
        if length < MIN_SCENE_CHARS_TO_KEEP:
            if merged:
                prev = merged[-1]
                prev['raw_text'] = (prev.get('raw_text','') + '\n' + (s.get('heading') or '') + '\n' + s.get('raw_text','')).strip()
                prev['scene_length_chars'] = len(prev['raw_text'])
                prev['dialogues'].extend(s.get('dialogues', []))
                prev['actions'].extend(s.get('actions', []))
                prev['characters'] = sorted(list(set(prev.get('characters',[]) + s.get('characters', []))))
            else:
                merged.append(s)
        else:
            merged.append(s)
    if merged and merged[0].get('scene_length_chars',0) < MIN_SCENE_CHARS_TO_KEEP and len(merged) > 1:
        first = merged.pop(0)
        merged[0]['raw_text'] = (first.get('raw_text','') + '\n' + merged[0].get('raw_text','')).strip()
        merged[0]['scene_length_chars'] = len(merged[0]['raw_text'])
        merged[0]['dialogues'] = first.get('dialogues',[]) + merged[0].get('dialogues',[])
        merged[0]['actions'] = first.get('actions',[]) + merged[0].get('actions',[])
        merged[0]['characters'] = sorted(list(set(first.get('characters',[]) + merged[0].get('characters', []))))
    return merged


def prepare_metadata_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    meta_map = {}
    if df.empty:
        return meta_map
    for _, row in df.iterrows():
        fn = str(row.get('file_name') or row.iloc[0]).strip()
        title, year = extract_title_and_year(fn)
        meta_map[fn] = {
            'file_name': fn,
            'description': row.get('description') if 'description' in row.index else None,
            'title': title,
            'year': year
        }
    return meta_map


def extract_title_and_year(file_name: str) -> Tuple[str, Optional[str]]:
    base = Path(file_name).stem
    m = re.search(r'_(\d{4})_', base)
    year = m.group(1) if m else None
    clean_title = re.sub(r'[_-]\s*\d{4}\s*[_-]', ' ', base)
    clean_title = re.sub(r'-\s*full transcript', '', clean_title, flags=re.IGNORECASE)
    clean_title = clean_title.strip()
    clean_title = re.sub(r'[_]+', ' ', clean_title).strip()
    return clean_title, year

# ----------------------------
# GoEmotions predictor with per-process cache & chunking for long texts
# ----------------------------
# This global will be process-local when using ProcessPoolExecutor
_PROCESS_GO_PREDICTOR = None

class GoEmotionsPredictor:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16, max_length: int = 256, fp16: bool = False):
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError("transformers/torch not installed - cannot use GoEmotions predictor")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = int(max_length)
        self.fp16 = fp16 and (self.device.type == "cuda")
        logging.info(f"[GoEmotionsPredictor] loading {model_name} on {self.device} (fp16={self.fp16})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        cfg = getattr(self.model, "config", None)
        if cfg and hasattr(cfg, "id2label"):
            self.id2label = {int(k): v for k, v in cfg.id2label.items()}
        else:
            l2i = getattr(cfg, "label2id", None) or {}
            inv = {v: k for k, v in l2i.items()}
            self.id2label = inv if inv else {}
        if self.id2label:
            max_idx = max(self.id2label.keys())
            labels = [self.id2label.get(i, f"lab_{i}") for i in range(max_idx + 1)]
            self.labels = labels
        else:
            self.labels = []

    def _chunk_text_by_tokens(self, text: str) -> List[str]:
        # Token-id chunking to avoid semantic break: we split token ids and decode pieces
        try:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            # fallback: simple character-based chunking
            chunks = []
            for i in range(0, len(text), max(256, self.max_length * 4)):
                chunks.append(text[i:i + max(256, self.max_length * 4)])
            return chunks
        if not ids:
            return [""]
        chunk_size = max(1, self.max_length - 2)
        chunks = []
        for i in range(0, len(ids), chunk_size):
            sub = ids[i:i + chunk_size]
            try:
                chunk_text = self.tokenizer.decode(sub, clean_up_tokenization_spaces=True)
            except Exception:
                chunk_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(sub))
            chunks.append(chunk_text)
        return chunks

    def predict_proba_for_text(self, text: str) -> np.ndarray:
        """
        Returns a numpy array of shape (n_labels,) of averaged sigmoid probabilities across chunks.
        """
        if not text or not text.strip():
            # return zeros
            return np.zeros((len(self.labels),), dtype=np.float32)
        chunks = self._chunk_text_by_tokens(text)
        # predict per chunk in batches
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                if self.fp16:
                    # mixed precision inference
                    with torch.cuda.amp.autocast():
                        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                else:
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits = out.logits.detach().cpu().numpy()  # (b, num_labels)
                probs = 1 / (1 + np.exp(-logits))  # sigmoid
                all_probs.append(probs.astype(np.float32))
        if not all_probs:
            return np.zeros((len(self.labels),), dtype=np.float32)
        all_probs = np.vstack(all_probs)  # (n_chunks, n_labels)
        avg = np.mean(all_probs, axis=0)
        return avg.astype(np.float32)

    def predict_proba_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Efficient batched prediction for a list of texts. This function:
         - token-chunks each text independently
         - concatenates all chunks across texts
         - runs batched model inference over the concatenated chunk list
         - averages chunk probabilities per-original-text to produce one probability vector per text
        Returns a list of numpy arrays (one per input text).
        """
        if not texts:
            return []
        # Build flat list of chunks and mapping
        all_chunks: List[str] = []
        mapping: List[Tuple[int, int]] = []
        for t in texts:
            chks = self._chunk_text_by_tokens(t or "")
            start = len(all_chunks)
            all_chunks.extend(chks)
            end = len(all_chunks)
            mapping.append((start, end))
        if not all_chunks:
            # all empty
            return [np.zeros((len(self.labels),), dtype=np.float32) for _ in texts]

        all_probs_list = []
        with torch.no_grad():
            for i in range(0, len(all_chunks), self.batch_size):
                batch = all_chunks[i:i + self.batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                else:
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits = out.logits.detach().cpu().numpy()
                probs = 1 / (1 + np.exp(-logits))
                all_probs_list.append(probs.astype(np.float32))

        if not all_probs_list:
            return [np.zeros((len(self.labels),), dtype=np.float32) for _ in texts]

        all_probs = np.vstack(all_probs_list)

        results: List[np.ndarray] = []
        for (start, end) in mapping:
            if start >= end:
                results.append(np.zeros((len(self.labels),), dtype=np.float32))
            else:
                chunk_probs = all_probs[start:end]
                avg = np.mean(chunk_probs, axis=0)
                results.append(avg.astype(np.float32))
        return results

def get_process_go_predictor(model_name: str, device: str, batch_size: int, max_length: int, fp16: bool):
    """
    Per-process cached getter. Each worker process will hold a single predictor instance.
    """
    global _PROCESS_GO_PREDICTOR
    if _PROCESS_GO_PREDICTOR is None:
        _PROCESS_GO_PREDICTOR = GoEmotionsPredictor(model_name=model_name, device=device, batch_size=batch_size, max_length=max_length, fp16=fp16)
    return _PROCESS_GO_PREDICTOR

# ----------------------------
# Core per-movie processing function (worker-friendly). Refactored so a shared predictor
# can be provided when running in threaded (GPU) mode. In that case we compute predictions
# for all scenes of a file in a single batch (fast, fewer locks).
# ----------------------------
def process_single_movie_core(
    txt_path_str: str,
    meta_row: Dict[str, Any],
    out_dir_str: str,
    use_goemotions: bool,
    go_model: str,
    go_batch: int,
    go_maxlen: int,
    go_topk: int,
    go_threshold: float,
    use_gpu: bool,
    fp16: bool,
    dialogue_only: bool,
    detect_lang: bool,
    shared_predictor=None,
    predictor_lock: threading.Lock = None
) -> Dict[str, Any]:
    """
    If shared_predictor is provided, we perform a single batched prediction for all scenes in the file
    while holding the predictor_lock. This reduces overhead and improves throughput on GPUs.
    """
    txt_path = Path(txt_path_str)
    out_dir = Path(out_dir_str)
    try:
        with txt_path.open('r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
    except Exception as e:
        return {'status': 'error', 'file': str(txt_path), 'error': str(e)}

    raw = clean_text(raw)
    if detect_lang and not is_english_text(raw):
        return {'status': 'skipped_non_english', 'file': str(txt_path)}

    title, year = extract_title_and_year(txt_path.name)

    scenes = split_initial(raw)
    if not scenes:
        scenes = split_paragraphs(raw)
    if not scenes:
        scenes = split_by_speaker_turns(raw, SPEAKER_TURN_CHUNK)
    if not scenes:
        scenes = split_fixed_lines(raw, LINES_PER_SCENE)
    if len(scenes) == 1 and scenes[0].get('scene_length_chars', len(scenes[0].get('raw_text','')) ) > (PUNCT_CHUNK_SIZE * 1.5):
        scenes = split_by_punctuation_chunks(scenes[0].get('raw_text',''), PUNCT_CHUNK_SIZE)

    processed_scenes = []
    movie_characters = set()
    for idx, sc in enumerate(scenes, start=1):
        scene_lines = sc.get('lines', [])
        parsed = extract_dialogues_and_actions(scene_lines)
        raw_text = sc.get('raw_text','')
        scene_dict = {
            'scene_id': idx,
            'heading': sc.get('heading'),
            'raw_text': raw_text,
            'characters': parsed['characters'],
            'dialogues': parsed['dialogues'],
            'actions': parsed['actions'],
            'scene_length_chars': len(raw_text),
            'split_method': sc.get('method')
        }
        processed_scenes.append(scene_dict)
        for c in parsed['characters']:
            movie_characters.add(c)

    processed_scenes = merge_empty_and_short_scenes(processed_scenes)
    num_scenes = len(processed_scenes)
    avg_scene_chars = int(sum(s.get('scene_length_chars',0) for s in processed_scenes) / max(1,num_scenes))
    split_method = scenes[0]['method'] if scenes else 'none'

    # GoEmotions labeling
    if use_goemotions and processed_scenes:
        # select device for predictor
        device_choice = "cuda" if (use_gpu and torch is not None and torch.cuda.is_available()) else "cpu"
        # obtain predictor
        if shared_predictor is None:
            predictor = get_process_go_predictor(go_model, device_choice, go_batch, go_maxlen, fp16)

            # per-scene (per-chunk) prediction inside process: unchanged behavior
            for s in processed_scenes:
                if dialogue_only:
                    dlg_texts = [d.get('text','') for d in s.get('dialogues',[])]
                    label_text = "\n".join([t for t in dlg_texts if t and t.strip()]) or s.get('raw_text','')
                else:
                    label_text = s.get('raw_text','') or ""
                try:
                    probs = predictor.predict_proba_for_text(label_text)
                except Exception as e:
                    logging.warning(f"GoEmotions predict error for {txt_path.name} scene {s.get('scene_id')}: {e}")
                    probs = np.zeros((len(predictor.labels),), dtype=np.float32)

                # pad/truncate probs to match labels
                if len(probs) < len(predictor.labels):
                    probs = np.concatenate([probs, np.zeros(len(predictor.labels) - len(probs), dtype=np.float32)])
                elif len(probs) > len(predictor.labels):
                    probs = probs[:len(predictor.labels)]

                s['goemotions_probs'] = probs.tolist()

                # top-k
                if predictor.labels:
                    idxs = probs.argsort()[::-1]
                    topk_idxs = idxs[:min(go_topk, len(idxs))]
                    topk_labels = [predictor.labels[i] for i in topk_idxs.tolist()]
                    s['goemotions_topk'] = topk_labels
                    s['goemotions_label'] = topk_labels[0] if topk_labels else None
                    # multi-label thresholding
                    if go_threshold is not None and go_threshold > 0:
                        multi_idxs = np.where(probs >= float(go_threshold))[0].tolist()
                        s['goemotions_labels'] = [predictor.labels[i] for i in multi_idxs]
                    else:
                        s['goemotions_labels'] = []
                else:
                    s['goemotions_topk'] = []
                    s['goemotions_label'] = None
                    s['goemotions_labels'] = []

        else:
            # Shared predictor provided (threaded GPU mode). Build label texts for all scenes,
            # then call predict_proba_batch once (under lock if provided) to exploit batching.
            label_texts = []
            for s in processed_scenes:
                if dialogue_only:
                    dlg_texts = [d.get('text','') for d in s.get('dialogues',[])]
                    label_text = "\n".join([t for t in dlg_texts if t and t.strip()]) or s.get('raw_text','')
                else:
                    label_text = s.get('raw_text','') or ""
                label_texts.append(label_text)

            try:
                if predictor_lock is not None:
                    with predictor_lock:
                        probs_list = shared_predictor.predict_proba_batch(label_texts)
                else:
                    probs_list = shared_predictor.predict_proba_batch(label_texts)
            except Exception as e:
                logging.warning(f"GoEmotions batch predict error for {txt_path.name}: {e}")
                # fallback: zeros
                probs_list = [np.zeros((len(shared_predictor.labels),), dtype=np.float32) for _ in label_texts]

            # assign back
            for s, probs in zip(processed_scenes, probs_list):
                if len(probs) < len(shared_predictor.labels):
                    probs = np.concatenate([probs, np.zeros(len(shared_predictor.labels) - len(probs), dtype=np.float32)])
                elif len(probs) > len(shared_predictor.labels):
                    probs = probs[:len(shared_predictor.labels)]
                s['goemotions_probs'] = probs.tolist()

            # compute topk & multi labels
            for s in processed_scenes:
                probs = np.array(s.get('goemotions_probs', []), dtype=np.float32)
                if shared_predictor.labels:
                    idxs = probs.argsort()[::-1]
                    topk_idxs = idxs[:min(go_topk, len(idxs))]
                    topk_labels = [shared_predictor.labels[i] for i in topk_idxs.tolist()]
                    s['goemotions_topk'] = topk_labels
                    s['goemotions_label'] = topk_labels[0] if topk_labels else None
                    if go_threshold is not None and go_threshold > 0:
                        multi_idxs = np.where(probs >= float(go_threshold))[0].tolist()
                        s['goemotions_labels'] = [shared_predictor.labels[i] for i in multi_idxs]
                    else:
                        s['goemotions_labels'] = []
                else:
                    s['goemotions_topk'] = []
                    s['goemotions_label'] = None
                    s['goemotions_labels'] = []

    else:
        for s in processed_scenes:
            s['goemotions_probs'] = []
            s['goemotions_topk'] = []
            s['goemotions_label'] = None
            s['goemotions_labels'] = []

    out = {
        'source_file': txt_path.name,
        'title': title,
        'year': year,
        'meta_description': meta_row.get('description'),
        'num_scenes': num_scenes,
        'avg_scene_chars': avg_scene_chars,
        'characters': sorted(list(movie_characters)),
        'scenes': processed_scenes,
        'split_summary': {'initial_method': split_method, 'final_num_scenes': num_scenes, 'lines_per_scene_fallback': LINES_PER_SCENE}
    }

    # write json
    safe_name = slugify(txt_path.name)[:200]
    out_file = out_dir / f"{safe_name}.json"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with out_file.open('w', encoding='utf-8') as fo:
            json.dump(out, fo, ensure_ascii=False, indent=2)
    except Exception as e:
        return {'status': 'error', 'file': str(txt_path), 'error': f"json write error: {e}"}

    # free memory in process/thread
    if shared_predictor is None and 'predictor' in locals():
        # in process mode predictor is module-global and will persist (but explicit del helps)
        try:
            del predictor
        except Exception:
            pass

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {'status': 'ok', 'file': str(txt_path), 'json': str(out_file), 'title': title, 'year': year, 'num_scenes': num_scenes, 'avg_chars': avg_scene_chars, 'method': split_method}

# thin wrapper kept for backward compatibility with ProcessPoolExecutor (picklable)
def process_single_movie_worker(
    txt_path_str: str,
    meta_row: Dict[str, Any],
    out_dir_str: str,
    use_goemotions: bool,
    go_model: str,
    go_batch: int,
    go_maxlen: int,
    go_topk: int,
    go_threshold: float,
    use_gpu: bool,
    fp16: bool,
    dialogue_only: bool,
    detect_lang: bool
) -> Dict[str, Any]:
    return process_single_movie_core(
        txt_path_str, meta_row, out_dir_str, use_goemotions, go_model, go_batch, go_maxlen, go_topk, go_threshold,
        use_gpu, fp16, dialogue_only, detect_lang, shared_predictor=None, predictor_lock=None
    )

# ----------------------------
# Orchestration: main entry (uses CONFIG instead of CLI)
# ----------------------------
def main():
    args = CONFIG  # use embedded config

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")

    ensure_nltk()

    transcripts_dir = Path(args.transcripts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = read_metadata(Path('./metadata.xlsx')) if Path('./metadata.xlsx').exists() else pd.DataFrame()
    meta_map = prepare_metadata_map(metadata_df)

    all_txt = sorted([p for p in transcripts_dir.glob("*.txt")])
    if not all_txt:
        logging.error(f"No transcripts found in {transcripts_dir}. Exiting.")
        return
    files_to_process = all_txt
    if args.limit:
        files_to_process = files_to_process[:args.limit]

    # default workers -> use all CPUs by default but cap to a safe max
    cpu_count = os.cpu_count() or 1
    # reserve one core for OS/main thread
    safe_max_workers = max(1, cpu_count - 1)

    # resolve requested workers (respect user but cap to safe_max_workers)
    if args.workers is None:
        workers = safe_max_workers
    else:
        try:
            workers = int(args.workers)
            if workers < 1:
                workers = 1
        except Exception:
            workers = safe_max_workers
        workers = min(workers, safe_max_workers)

    # Determine mode
    cuda_available = (torch is not None and torch.cuda.is_available())
    using_gpu = bool(args.use_gpu and cuda_available)

    if args.use_gpu and not cuda_available:
        logging.warning("--use_gpu requested but CUDA not available. Falling back to CPU inference.")
        using_gpu = False

    # If GPU is requested and multiple workers, prefer threaded mode with a shared predictor
    use_threaded_gpu_mode = False
    if using_gpu:
        if workers > 1 and not args.gpu_parallel:
            logging.info("CUDA available and use_gpu set: using threaded mode with a single shared GPU predictor to avoid multiple processes loading the model. Set gpu_parallel to force per-process GPU loads (may OOM).")
            use_threaded_gpu_mode = True
        elif workers > 1 and args.gpu_parallel:
            logging.warning("gpu_parallel set: allowing multiple processes to each load the model on GPU. This can cause OOM or slowdowns.")
            use_threaded_gpu_mode = False

    logging.info(f"Found {len(all_txt)} transcripts; will process {len(files_to_process)} files (workers={workers}, goemotions={args.use_goemotions}, use_gpu={using_gpu})")

    results = []

    shared_predictor = None
    predictor_lock = None

    predict_batch_size = args.predict_batch_size or args.go_batch

    if args.use_goemotions and using_gpu and use_threaded_gpu_mode:
        # Load a single shared predictor on GPU in the main process and use ThreadPoolExecutor
        logging.info("Loading shared GoEmotions predictor on GPU (main process)...")
        try:
            shared_predictor = GoEmotionsPredictor(model_name=args.go_model, device='cuda', batch_size=predict_batch_size, max_length=args.go_maxlen, fp16=args.fp16)
        except Exception as e:
            logging.error(f"Failed to load GPU predictor: {e}. Falling back to CPU/process mode.")
            shared_predictor = None

        predictor_lock = threading.Lock() if shared_predictor is not None else None

        if shared_predictor is not None:
            # determine a safe default and respect user-provided gpu_threads if valid
            default_gpu_threads = min(max(1, workers), 16, cpu_count)

            if args.gpu_threads is None:
                max_workers = min(default_gpu_threads, len(files_to_process))
            else:
                try:
                    requested = int(args.gpu_threads)
                    if requested < 1:
                        requested = 1
                except Exception:
                    requested = default_gpu_threads
                # never exceed CPU count or number of files
                max_workers = min(requested, cpu_count, len(files_to_process))

            logging.info(f"Starting ThreadPoolExecutor with {max_workers} workers (shared GPU predictor).")
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                for pth in files_to_process:
                    key = pth.name
                    meta_row = meta_map.get(key, {'file_name': key, 'description': None, 'title': None, 'year': None})
                    futures[exe.submit(
                        process_single_movie_core,
                        str(pth),
                        meta_row,
                        str(out_dir),
                        args.use_goemotions,
                        args.go_model,
                        args.go_batch,
                        args.go_maxlen,
                        args.go_topk,
                        args.go_threshold,
                        args.use_gpu,
                        args.fp16,
                        args.dialogue_only,
                        args.detect_lang,
                        shared_predictor,
                        predictor_lock
                    )] = pth
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing (threaded, shared GPU)"):
                    pth = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {'status': 'error', 'file': str(pth), 'error': str(e)}
                    results.append(res)
        else:
            # failed to load shared predictor; fallback to process pool below
            logging.info("Shared predictor could not be created; falling back to CPU/process mode.")

    # If not using threaded GPU mode, use ProcessPoolExecutor (or sequential)
    if not (args.use_goemotions and using_gpu and use_threaded_gpu_mode and shared_predictor is not None):
        if workers > 1:
            # cap max_workers to number of files and safe_max_workers
            max_workers = min(workers, max(1, len(files_to_process)))
            max_workers = min(max_workers, safe_max_workers)
            logging.info(f"Starting ProcessPoolExecutor with {max_workers} workers.")
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                for pth in files_to_process:
                    key = pth.name
                    meta_row = meta_map.get(key, {'file_name': key, 'description': None, 'title': None, 'year': None})
                    futures[exe.submit(
                        process_single_movie_worker,
                        str(pth),
                        meta_row,
                        str(out_dir),
                        args.use_goemotions,
                        args.go_model,
                        args.go_batch,
                        args.go_maxlen,
                        args.go_topk,
                        args.go_threshold,
                        args.use_gpu,
                        args.fp16,
                        args.dialogue_only,
                        args.detect_lang
                    )] = pth
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing (parallel)"):
                    pth = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {'status': 'error', 'file': str(pth), 'error': str(e)}
                    results.append(res)
        else:
            # single-process sequential (safer when debugging / small runs)
            for pth in tqdm(files_to_process, desc="Processing (sequential)"):
                key = pth.name
                meta_row = meta_map.get(key, {'file_name': key, 'description': None, 'title': None, 'year': None})
                res = process_single_movie_worker(str(pth), meta_row, str(out_dir),
                                                  args.use_goemotions, args.go_model, args.go_batch, args.go_maxlen,
                                                  args.go_topk, args.go_threshold, args.use_gpu, args.fp16,
                                                  args.dialogue_only, args.detect_lang)
                results.append(res)

    # Save index CSV with diagnostics
    index_rows = []
    for r in results:
        if not r:
            continue
        if r.get('status') == 'ok':
            index_rows.append({'source_file': r['file'], 'json_path': r['json'],
                               'title': r.get('title'), 'year': r.get('year'),
                               'num_scenes': r.get('num_scenes'), 'avg_scene_chars': r.get('avg_chars'),
                               'status': 'ok', 'method': r.get('method')})
        else:
            index_rows.append({'source_file': r.get('file'), 'json_path': None,
                               'title': None, 'year': None, 'num_scenes': None, 'avg_scene_chars': None,
                               'status': r.get('status'), 'error': r.get('error')})
    index_df = pd.DataFrame(index_rows)
    index_csv = out_dir / 'processed_index.csv'
    index_df.to_csv(index_csv, index=False)
    logging.info(f"Processed index saved to {index_csv}")

    # === New: Build scene-level CSV ===
    scene_rows = []
    for idx, row in index_df.iterrows():
        json_path = row.get('json_path')
        source_file = row.get('source_file')
        if not json_path:
            continue
        jp = Path(json_path)
        if not jp.exists():
            logging.warning(f"Expected JSON {jp} not found for {source_file}; skipping scene extraction.")
            continue
        try:
            with jp.open('r', encoding='utf-8') as f:
                doc = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to read {jp}: {e}")
            continue

        title = doc.get('title')
        year = doc.get('year')
        meta_description = doc.get('meta_description')

        scenes = doc.get('scenes', [])
        for s in scenes:
            scene_id = s.get('scene_id')
            heading = s.get('heading')
            raw_text = s.get('raw_text', '')
            scene_len = s.get('scene_length_chars', len(raw_text))
            characters = s.get('characters', [])
            dialogues = s.get('dialogues', [])
            actions = s.get('actions', [])
            go_label = s.get('goemotions_label')
            go_topk = s.get('goemotions_topk', [])
            go_multi = s.get('goemotions_labels', [])
            go_probs = s.get('goemotions_probs', [])

            scene_rows.append({
                'source_file': source_file,
                'title': title,
                'year': year,
                'meta_description': meta_description,
                'scene_id': scene_id,
                'heading': heading,
                'scene_text': raw_text,
                'scene_length_chars': scene_len,
                'characters': '|'.join(characters) if characters else '',
                'n_dialogues': len(dialogues) if dialogues else 0,
                'n_actions': len(actions) if actions else 0,
                'goemotions_label': go_label,
                'goemotions_topk': '|'.join(go_topk) if go_topk else '',
                'goemotions_labels': json.dumps(go_multi, ensure_ascii=False),
                'goemotions_probs': json.dumps(go_probs),
                'split_method': s.get('split_method', doc.get('split_summary', {}).get('initial_method'))
            })

    if scene_rows:
        scene_df = pd.DataFrame(scene_rows)
        scene_csv = out_dir / 'scene_labels.csv'
        # Save: let pandas handle quoting; JSON fields are already strings.
        scene_df.to_csv(scene_csv, index=False)
        logging.info(f"Scene-level CSV saved to {scene_csv} (rows={len(scene_df)})")
    else:
        logging.info("No scene rows produced; scene_labels.csv will not be created.")

    logging.info(f"Done. Processed {len(results)} files. Index saved to {index_csv}")

if __name__ == "__main__":
    main()
