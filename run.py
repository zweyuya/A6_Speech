#!/usr/bin/env python3
"""
run.py — A6 Speech Processing submission CLI

Usage:
    # Train the toy CTC model (Part 3 / Exercise 2)
    python3 run.py --model ctc --epochs 300 --train

    # Linear-probe a pretrained wav2vec2 checkpoint (Part 4 / Exercise 3)
    python3 run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train

    # Extract tone color from your reference clip
    python3 run.py --model voice-clone --extract-se --reference my_voice.wav

    # Synthesize in a given style with your cloned voice
    python3 run.py --model voice-clone --accent us --text "I got the job!" --generate

    # Synthesize all styles for comparison
    python3 run.py --model voice-clone --accent all --text "Hello world" --generate

    # Cross-lingual cloning
    python3 run.py --model voice-clone --language es --text "Hola, como estas?" --generate
"""

import argparse
import os
import re
import random
import shutil
import subprocess
import sys

import numpy as np


# ============================================================================
# Shared utilities
# ============================================================================

def get_device():
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def ensure_dirs():
    os.makedirs('data/speech', exist_ok=True)
    os.makedirs('data/speechcommands', exist_ok=True)
    os.makedirs('data/voice_clone', exist_ok=True)
    os.makedirs('data/voice_clone/processed', exist_ok=True)


def ensure_ffmpeg_on_path():
    """Make sure a working ffmpeg binary is discoverable on PATH.

    Falls back to the ffmpeg binary bundled with imageio-ffmpeg if the
    system doesn't have one already, since several downstream libraries
    (whisper, pydub) shell out to a literal `ffmpeg` command.
    """
    if shutil.which('ffmpeg') is not None:
        return
    try:
        import imageio_ffmpeg
        ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = os.path.expanduser('~/bin')
        os.makedirs(bin_dir, exist_ok=True)
        ffmpeg_dst = os.path.join(bin_dir, 'ffmpeg')
        if not os.path.exists(ffmpeg_dst):
            os.symlink(ffmpeg_src, ffmpeg_dst)
            os.chmod(ffmpeg_dst, 0o755)
        os.environ['PATH'] = bin_dir + os.pathsep + os.environ['PATH']
        print(f'[setup] ffmpeg not found on PATH; using bundled binary via {ffmpeg_dst}')
    except ImportError:
        print('[setup] WARNING: ffmpeg not found and imageio-ffmpeg not installed. '
              'Run: pip install imageio-ffmpeg')


def convert_to_wav(src_path, dst_path=None, target_sr=22050):
    """Convert any audio file (m4a/mp3/etc.) to a mono WAV via ffmpeg."""
    if dst_path is None:
        base, _ = os.path.splitext(src_path)
        dst_path = base + '.wav'

    ffmpeg_bin = shutil.which('ffmpeg') or 'ffmpeg'
    result = subprocess.run([
        ffmpeg_bin, '-y', '-i', src_path,
        '-ar', str(target_sr), '-ac', '1', dst_path,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg conversion failed:\n{result.stderr[-2000:]}')

    return dst_path


# ============================================================================
# PART A: SpeechTokenizer  (Exercise 1)
# ============================================================================

class SpeechTokenizer:
    """Character-level tokenizer for TTS. Handles: text normalization,
    special tokens, accent tags."""

    ACCENTS = ['[EN-US]', '[EN-BR]', '[EN-INDIA]', '[EN-AU]', '[EN-DEFAULT]']

    def __init__(self):
        chars = " !',-.?abcdefghijklmnopqrstuvwxyz"
        self.vocab = {c: i + 3 for i, c in enumerate(chars)}
        self.vocab['<PAD>'] = 0
        self.vocab['<BOS>'] = 1
        self.vocab['<EOS>'] = 2
        for a in self.ACCENTS:
            self.vocab[a] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def normalize(self, text):
        text = text.lower()
        text = re.sub(r'dr\.', 'doctor', text)
        text = re.sub(r'mr\.', 'mister', text)
        text = re.sub(r'(\d+)', lambda m: self._num_to_words(int(m.group())), text)
        text = re.sub(r"[^a-z !',\-.?\[\]]", '', text)
        return text.strip()

    def _num_to_words(self, n):
        words = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                  6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
        return words.get(n, str(n))

    def encode(self, text, add_special=True):
        tag_pattern = '|'.join(re.escape(a) for a in self.ACCENTS)
        parts = re.split(f'({tag_pattern})', text)
        tokens = []
        if add_special:
            tokens.append(self.vocab['<BOS>'])
        for part in parts:
            if part in self.ACCENTS:
                tokens.append(self.vocab[part])
            else:
                normalized = self.normalize(part)
                for ch in normalized:
                    if ch in self.vocab:
                        tokens.append(self.vocab[ch])
        if add_special:
            tokens.append(self.vocab['<EOS>'])
        return tokens

    def decode(self, ids):
        return ''.join(
            self.inv_vocab.get(i, '?') for i in ids
            if i not in (self.vocab['<PAD>'], self.vocab['<BOS>'], self.vocab['<EOS>'])
        )

    def __len__(self):
        return len(self.vocab)


def run_tokenizer_demo():
    """Exercise 1: tokenize the 5 sentences and print the results table."""
    tokenizer = SpeechTokenizer()
    sentences = [
        "Hello, how are you?",
        "Dr. Smith prescribed 10 tablets.",
        "[EN-US] I got the job!",
        "[EN-BR] I lost my wallet.",
        "[EN-INDIA] This is completely unacceptable!",
    ]

    rows = []
    for sent in sentences:
        ids_with_special = tokenizer.encode(sent, add_special=True)
        ids_no_special = tokenizer.encode(sent, add_special=False)
        tag = [a for a in SpeechTokenizer.ACCENTS if a in sent]
        tag_id = tokenizer.vocab[tag[0]] if tag else '-'
        rows.append((sent, len(ids_no_special), len(ids_with_special), tag_id))

    print('| Sentence | # Char tokens | # Tokens (with BOS/EOS) | Accent tag token ID |')
    print('|---|---|---|---|')
    for sent, n_char, n_tot, tag_id in rows:
        print(f'| {sent} | {n_char} | {n_tot} | {tag_id} |')


# ============================================================================
# PART B: CTC  (Exercise 2)
# ============================================================================

BLANK = '_'


def ctc_collapse(alignment):
    """Merge consecutive duplicates, then remove blanks."""
    merged = []
    for ch in alignment:
        if not merged or ch != merged[-1]:
            merged.append(ch)
    return ''.join(ch for ch in merged if ch != BLANK)


NEG_INF = -1e9


def log_add(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_forward_log_prob(log_probs, labels, blank=0):
    """CTC forward algorithm computing log P_CTC(labels | log_probs)."""
    T, V = log_probs.shape
    ext = [blank]
    for lab in labels:
        ext += [lab, blank]
    S = len(ext)

    alpha = np.full((T, S), NEG_INF)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            stay = alpha[t - 1, s]
            prev = alpha[t - 1, s - 1] if s - 1 >= 0 else NEG_INF
            skip = NEG_INF
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s - 2]:
                skip = alpha[t - 1, s - 2]
            best_prev = log_add(log_add(stay, prev), skip)
            alpha[t, s] = best_prev + log_probs[t, ext[s]]

    if S == 1:
        return alpha[T - 1, S - 1]
    return log_add(alpha[T - 1, S - 1], alpha[T - 1, S - 2])


def edit_distance(a, b):
    """Levenshtein distance between strings a and b."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


ALPHABET = list('helo wrd')
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 reserved for blank
IDX2CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 1
N_MELS = 20
WORDS = ['hello', 'world', 'hero', 'red', 'led', 'doer']


def synthesize_frames(word, frames_per_char=(3, 8)):
    frames = []
    for ch in word:
        n = random.randint(*frames_per_char)
        base = np.zeros(N_MELS)
        base[CHAR2IDX[ch] % N_MELS] = 3.0
        for _ in range(n):
            frames.append(base + np.random.randn(N_MELS) * 0.5)
    return np.stack(frames)


def build_ctc_model():
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyCTCModel(nn.Module):
        def __init__(self, in_dim=N_MELS, hidden=64, vocab=VOCAB_SIZE):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden * 2, vocab)

        def forward(self, x):
            h, _ = self.lstm(x)
            return F.log_softmax(self.fc(h), dim=-1)

    return TinyCTCModel()


def train_ctc(epochs=300, seed=0):
    """Exercise 2: train the toy CTC model, tracking CER, then run the
    frames_per_char=(1,2) greedy-decode robustness check."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = build_ctc_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    def greedy_decode(model, word, frames_per_char=(3, 8)):
        frames = synthesize_frames(word, frames_per_char)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            log_probs = model(x).squeeze(0)
        pred_ids = log_probs.argmax(dim=-1).tolist()
        pred_chars_raw = [IDX2CHAR.get(i, BLANK) if i != 0 else BLANK for i in pred_ids]
        return ctc_collapse(pred_chars_raw)

    losses, cers = [], []
    first_step_below_10 = None

    print(f'Training toy CTC model for {epochs} steps...')
    for step in range(epochs):
        word = random.choice(WORDS)
        frames = synthesize_frames(word)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
        targets = torch.tensor([CHAR2IDX[c] for c in word], dtype=torch.long)

        log_probs = model(x).transpose(0, 1)
        input_lengths = torch.tensor([log_probs.size(0)])
        target_lengths = torch.tensor([len(targets)])

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        model.eval()
        eval_word = random.choice(WORDS)
        decoded = greedy_decode(model, eval_word)
        cer = edit_distance(decoded, eval_word) / max(len(eval_word), 1)
        cers.append(cer)
        model.train()

        if first_step_below_10 is None and step > 20 and np.mean(cers[-20:]) < 0.10:
            first_step_below_10 = step + 1

        if (step + 1) % 50 == 0:
            print(f'Step {step + 1:3d} | CTC loss: {np.mean(losses[-50:]):.4f} | '
                  f'CER (last 20): {np.mean(cers[-20:]) * 100:.1f}%')

    print(f'\nCER first dropped below 10% at step: {first_step_below_10}')

    # Exercise 2a: collapsing sanity check
    print('\n-- Exercise 2a: ctc_collapse sanity check --')
    target_word = 'HELLO'
    my_alignments = [
        list('HHEE_LL_LOOO'),
        list('H_E_L_L_O'),
        list('HHHHEEEE_LLLL_LLLLOOOO'),
    ]
    for align in my_alignments:
        result = ctc_collapse(align)
        status = 'OK' if result == target_word else 'MISMATCH'
        print(f'  {"".join(align):28} -> "{result}"  [{status}]')

    # Exercise 2b: P_CTC comparison
    print('\n-- Exercise 2b: P_CTC("HEL") vs P_CTC("LEH") --')
    vocab = {0: BLANK, 1: 'H', 2: 'E', 3: 'L', 4: 'O'}
    char2id = {v: k for k, v in vocab.items()}
    np.random.seed(0)
    logits = np.random.randn(6, 5)
    log_probs_np = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    logp_hel = ctc_forward_log_prob(log_probs_np, [char2id[c] for c in 'HEL'])
    logp_leh = ctc_forward_log_prob(log_probs_np, [char2id[c] for c in 'LEH'])
    print(f'  log P_CTC("HEL") = {logp_hel:.4f}  ->  P = {np.exp(logp_hel):.6f}')
    print(f'  log P_CTC("LEH") = {logp_leh:.4f}  ->  P = {np.exp(logp_leh):.6f}')

    # Exercise 2d: shorter frames_per_char robustness check
    print('\n-- Exercise 2d: greedy decode with frames_per_char=(1,2) --')
    model.eval()
    n_correct = 0
    for word in WORDS:
        decoded = greedy_decode(model, word, frames_per_char=(1, 2))
        correct = decoded == word
        n_correct += correct
        print(f'  "{word}" -> "{decoded}"  [{"correct" if correct else "wrong"}]')
    print(f'\nAccuracy with frames_per_char=(1,2): {n_correct}/{len(WORDS)} words correct')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/ctc_model.pt')
    print('\nSaved model checkpoint to checkpoints/ctc_model.pt')


# ============================================================================
# PART C: wav2vec2 linear probe  (Exercise 3)
# ============================================================================

def train_linear_probe(X, y, n_classes, epochs=100):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=0.3, random_state=42, stratify=y.numpy())
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    probe = nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(epochs):
        logits = probe(X_train_t)
        loss = F.cross_entropy(logits, y_train_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc = (probe(X_test_t).argmax(1) == y_test_t).float().mean().item()
    return acc


def run_wav2vec2_probe(dataset, classes, n_per_class=40):
    """Exercise 3: raw mel-spectrogram baseline vs. frozen wav2vec2 linear probe."""
    import torch
    import torchaudio
    import torchaudio.transforms as T
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    if dataset != 'speechcommands':
        raise ValueError(f'Unsupported dataset: {dataset}')

    device = get_device()
    ensure_dirs()

    probe_words = classes
    print(f'Probing classes: {probe_words}')

    w2v_name = 'facebook/wav2vec2-base'
    w2v_extractor = Wav2Vec2FeatureExtractor.from_pretrained(w2v_name)
    w2v_model = Wav2Vec2Model.from_pretrained(w2v_name).to(device).eval()
    for p in w2v_model.parameters():
        p.requires_grad = False
    print(f'Loaded {w2v_name} ({sum(p.numel() for p in w2v_model.parameters()):,} frozen params)')

    sc_dataset = torchaudio.datasets.SPEECHCOMMANDS(root='data/speechcommands', download=True)

    by_label = {w: [] for w in probe_words}
    for i in range(len(sc_dataset)):
        wvf, sr, label, *_ = sc_dataset[i]
        if label in by_label and len(by_label[label]) < n_per_class:
            by_label[label].append(wvf)
        if all(len(v) >= n_per_class for v in by_label.values()):
            break

    mel_tf = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80)

    mel_feats, w2v_feats, labels_list = [], [], []
    with torch.no_grad():
        for label, clips in by_label.items():
            for wvf in clips:
                mel = mel_tf(wvf).squeeze()
                mel_feats.append(mel.mean(dim=-1))

                inputs = w2v_extractor(wvf.squeeze(0).numpy(), sampling_rate=16000,
                                        return_tensors='pt').to(device)
                out = w2v_model(**inputs).last_hidden_state
                w2v_feats.append(out.mean(dim=1).squeeze(0).cpu())

                labels_list.append(probe_words.index(label))

    X_mel = torch.stack(mel_feats)
    X_w2v = torch.stack(w2v_feats)
    y = torch.tensor(labels_list)

    print(f'Raw mel-spectrogram features: {X_mel.shape}')
    print(f'wav2vec2 features:            {X_w2v.shape}')

    acc_mel = train_linear_probe(X_mel, y, len(probe_words))
    acc_w2v = train_linear_probe(X_w2v, y, len(probe_words))
    random_baseline = 100 / len(probe_words)

    print(f'\n| Feature | Test Accuracy |')
    print(f'|---|---|')
    print(f'| Raw mel-spectrogram (mean-pooled) | {acc_mel * 100:.1f}% |')
    print(f'| wav2vec2 (frozen, mean-pooled)    | {acc_w2v * 100:.1f}% |')
    print(f'\nRandom baseline ({len(probe_words)}-way): {random_baseline:.1f}%')
    print(f'wav2vec2 gap over raw baseline: {(acc_w2v - acc_mel) * 100:.1f} points')


# ============================================================================
# PART D: Voice cloning with OpenVoice  (Exercise 4)
# ============================================================================

def load_openvoice(device_str):
    from huggingface_hub import snapshot_download
    from openvoice.api import ToneColorConverter

    ckpt_dir = snapshot_download(repo_id='myshell-ai/OpenVoiceV2')
    tone_color_converter = ToneColorConverter(f'{ckpt_dir}/converter/config.json', device=device_str)
    tone_color_converter.load_ckpt(f'{ckpt_dir}/converter/checkpoint.pth')
    print('OpenVoiceV2 loaded.')
    return ckpt_dir, tone_color_converter


def extract_se(reference, device_str):
    """Extract and save a tone color embedding from a reference clip."""
    import torch
    from openvoice import se_extractor

    ensure_dirs()
    ensure_ffmpeg_on_path()

    ref_path = reference
    if not ref_path.lower().endswith('.wav'):
        print(f'Converting {ref_path} to WAV...')
        ref_path = convert_to_wav(ref_path)

    ckpt_dir, tone_color_converter = load_openvoice(device_str)

    target_se, audio_name = se_extractor.get_se(
        ref_path, tone_color_converter, target_dir='data/voice_clone/processed', vad=True)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(target_se, 'checkpoints/target_se.pt')
    print(f'Tone color embedding shape: {target_se.shape}')
    print('Saved to checkpoints/target_se.pt')
    return target_se, ckpt_dir, tone_color_converter


def load_saved_se():
    import torch
    se_path = 'checkpoints/target_se.pt'
    if not os.path.exists(se_path):
        raise FileNotFoundError(
            f'{se_path} not found. Run --extract-se first:\n'
            f'  python3 run.py --model voice-clone --extract-se --reference my_voice.wav'
        )
    return torch.load(se_path)


def synthesize_accent(accent, text, device_str, target_se, ckpt_dir, tone_color_converter,
                       base_speaker_tts, speaker_ids):
    import torch
    import torchaudio
    import torchaudio.transforms as T

    style_to_se = {
        'us':    ('en-us.pth',    'EN-US'),
        'br':    ('en-br.pth',    'EN-BR'),
        'india': ('en-india.pth', 'EN_INDIA'),
        'au':    ('en-au.pth',    'EN-AU'),
    }
    se_file, spk_key = style_to_se[accent]
    spk_id = speaker_ids[spk_key]

    base_path = f'data/voice_clone/base_{accent}.wav'
    out_path = f'data/voice_clone/cloned_{accent}.wav'

    base_speaker_tts.tts_to_file(text, spk_id, base_path, speed=1.0)

    source_se = torch.load(f'{ckpt_dir}/base_speakers/ses/{se_file}', map_location=device_str)
    tone_color_converter.convert(
        audio_src_path=base_path, src_se=source_se, tgt_se=target_se,
        output_path=out_path, tau=0.3)

    # Acoustic measurements (Exercise 4a)
    wvf, sr = torchaudio.load(out_path)
    if sr != 22050:
        wvf = T.Resample(sr, 22050)(wvf)
        sr = 22050
    duration_s = wvf.shape[-1] / sr
    rms_energy = torch.sqrt((wvf ** 2).mean()).item()

    mel_tf = T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80)
    mel = mel_tf(wvf[0].unsqueeze(0)).squeeze()
    freqs = torch.linspace(0, sr / 2, mel.shape[0])
    mel_energy = mel.mean(dim=1)
    spectral_centroid = (freqs * mel_energy).sum() / mel_energy.sum()

    print(f'[{accent:6}] -> {out_path}  duration={duration_s:.2f}s  '
          f'RMS={rms_energy:.4f}  centroid={spectral_centroid.item():.1f} Hz')

    return out_path, duration_s, rms_energy, spectral_centroid.item()


def generate_voice_clone(accent, text, language, device_str):
    """Exercise 4: synthesize --text in the cloned voice, for one accent,
    all accents, or a different base-speaker language (cross-lingual)."""
    import nltk
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    ensure_dirs()
    ensure_ffmpeg_on_path()

    target_se = load_saved_se()
    ckpt_dir, tone_color_converter = load_openvoice(device_str)

    from melo.api import TTS as MeloTTS

    if language:
        # Cross-lingual cloning: same tone color, different base-speaker language
        base_tts = MeloTTS(language=language.upper(), device=device_str)
        spk_ids = base_tts.hps.data.spk2id
        spk_key = list(spk_ids.keys())[0]

        base_path = f'data/voice_clone/base_{language}.wav'
        out_path = f'data/voice_clone/cloned_{language}.wav'

        base_tts.tts_to_file(text, spk_ids[spk_key], base_path, speed=1.0)

        se_file_path = f'{ckpt_dir}/base_speakers/ses/{language.lower()}.pth'
        if not os.path.exists(se_file_path):
            print(f'[warn] No dedicated base-speaker SE for language "{language}", '
                  f'falling back to en-default.pth')
            se_file_path = f'{ckpt_dir}/base_speakers/ses/en-default.pth'

        import torch
        source_se = torch.load(se_file_path, map_location=device_str)
        tone_color_converter.convert(
            audio_src_path=base_path, src_se=source_se, tgt_se=target_se, output_path=out_path)

        print(f'[{language}] "{text}" -> {out_path}')
        return

    base_speaker_tts = MeloTTS(language='EN', device=device_str)
    speaker_ids = base_speaker_tts.hps.data.spk2id

    accents_to_run = ['us', 'br', 'india', 'au'] if accent == 'all' else [accent]

    results = []
    for acc in accents_to_run:
        result = synthesize_accent(acc, text, device_str, target_se, ckpt_dir,
                                    tone_color_converter, base_speaker_tts, speaker_ids)
        results.append((acc,) + result[1:])

    if len(results) > 1:
        print('\n| Accent | Duration (s) | RMS Energy | Mel Spectral Centroid |')
        print('|---|---|---|---|')
        for acc, d, r, c in results:
            print(f'| {acc} | {d:.2f} | {r:.4f} | {c:.1f} Hz |')


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='A6 Speech Processing — submission CLI')
    parser.add_argument('--model', required=True,
                         choices=['ctc', 'wav2vec2-probe', 'voice-clone'])

    # ctc
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train', action='store_true')

    # wav2vec2-probe
    parser.add_argument('--dataset', type=str, default='speechcommands')
    parser.add_argument('--classes', type=str, default='yes,no,stop,go',
                         help='Comma-separated SpeechCommands class names')

    # voice-clone
    parser.add_argument('--extract-se', action='store_true')
    parser.add_argument('--reference', type=str, default=None,
                         help='Path to your reference voice recording (wav/mp3/m4a)')
    parser.add_argument('--accent', type=str, default=None,
                         choices=['us', 'br', 'india', 'au', 'all'])
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--language', type=str, default=None,
                         help='Cross-lingual base-speaker language code, e.g. es, fr')
    parser.add_argument('--generate', action='store_true')

    args = parser.parse_args()
    ensure_dirs()

    if args.model == 'ctc':
        if not args.train:
            parser.error('--train is required for --model ctc')
        run_tokenizer_demo()
        print()
        train_ctc(epochs=args.epochs)

    elif args.model == 'wav2vec2-probe':
        if not args.train:
            parser.error('--train is required for --model wav2vec2-probe')
        classes = [c.strip() for c in args.classes.split(',')]
        run_wav2vec2_probe(args.dataset, classes)

    elif args.model == 'voice-clone':
        device = get_device()
        device_str = str(device)
        print(f'Using device: {device}')

        if args.extract_se:
            if not args.reference:
                parser.error('--reference is required with --extract-se')
            extract_se(args.reference, device_str)

        elif args.generate:
            if not args.text:
                parser.error('--text is required with --generate')
            if args.language:
                generate_voice_clone(None, args.text, args.language, device_str)
            else:
                if not args.accent:
                    parser.error('--accent (or --language) is required with --generate')
                generate_voice_clone(args.accent, args.text, None, device_str)

        else:
            parser.error('--model voice-clone requires --extract-se or --generate')


if __name__ == '__main__':
    main()
