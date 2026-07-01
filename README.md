# A6: Speech Processing — Submission

**Student:** Zwe Yu Ya Kyaw Zin Oo (st125990)

This repository contains the training/inference script (`run.py`), the full exercises notebook (`A6_Speech_Processing_Exercises.ipynb`), and this README with results, visualizations, and discussion for Exercises 1–4.

---

## Commands used

```bash
# Exercise 1 (Tokenization) + Exercise 2 (CTC): train the toy CTC model,
# which also runs the Ex1 tokenizer table and Ex2a/b/d checks
python3 run.py --model ctc --epochs 300 --train

# Exercise 3: linear-probe a pretrained wav2vec2 checkpoint on SpeechCommands
python3 run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train

# Exercise 4: extract tone color from my reference clip
python3 run.py --model voice-clone --extract-se --reference my_voice.wav

# Exercise 4: synthesize the test sentence in a single accent with my cloned voice
python3 run.py --model voice-clone --accent us --text "I got the job!" --generate

# Exercise 4: synthesize all four accents for comparison
python3 run.py --model voice-clone --accent all --text "I got the job!" --generate

# Exercise 4 (bonus): cross-lingual cloning
python3 run.py --model voice-clone --language es --text "Hola, como estas?" --generate
```

---

## Results table

| Task | Model / Method | Result | Notes |
|---|---|---|---|
| Tokenization (Ex 1) | SpeechTokenizer | — | char vs. token count table below |
| CTC character error rate (Ex 2) | Toy BiLSTM + CTC | CER < 10% by step 92; 0% by step 100 | error rate vs. training step |
| wav2vec2 vs. raw-feature probe (Ex 3) | Linear probe | 85.4% vs. 62.5% | wav2vec2 (frozen) vs. mel-spectrogram baseline |
| Voice cloning: accent + cross-lingual (Ex 4) | OpenVoice | mean cosine sim = 0.7134 (std 0.0477) | identity-vs-accent and language transfer |

### Exercise 1 — Tokenization detail

| Sentence | # Char tokens | # Tokens (with BOS/EOS) | Accent tag token ID |
|---|---|---|---|
| Hello, how are you? | 19 | 21 | — |
| Dr. Smith prescribed 10 tablets. | 36 | 38 | — |
| [EN-US] I got the job! | 15 | 17 | 36 |
| [EN-BR] I lost my wallet. | 18 | 20 | 37 |
| [EN-INDIA] This is completely unacceptable! | 33 | 35 | 38 |

### Exercise 2 — CTC detail

**a) Collapsing sanity check** — 3 hand-built alignments, all correctly collapse to `"HELLO"`:

| Alignment | Collapsed | Match |
|---|---|---|
| `HHEE_LL_LOOO` | `HELLO` | ✅ |
| `H_E_L_L_O` | `HELLO` | ✅ |
| `HHHHEEEE_LLLL_LLLLOOOO` | `HELLO` | ✅ |

**b) P_CTC comparison** — "HEL" vs. "LEH" on the same random 6-frame log-probs matrix:

| Word | log P_CTC | P_CTC |
|---|---|---|
| HEL | -5.5957 | 0.003714 |
| LEH | -7.0833 | 0.000839 |

Different orderings of the same three letters get different probabilities because the per-frame distribution isn't symmetric across symbols — an ordering matching the model's per-frame preferences scores higher.

**c) CER vs. training step:**

| Step | CTC Loss | CER (20-step rolling avg) |
|---|---|---|
| 50 | 2.2852 | 88.4% |
| 100 | 0.6212 | 0.0% |
| 150 | 0.0872 | 6.0% |
| 200 | 0.0436 | 4.0% |
| 250 | 0.0154 | 0.0% |
| 300 | 0.0136 | 0.0% |

CER first dropped below 10% at **step 92**.

**d) Shorter `frames_per_char=(1,2)` robustness check:** accuracy dropped to **3/6 words correct** (from near-100% at the trained `(3,8)` setting), confirming that fewer frames per character removes the redundancy the model and the collapsing function both rely on.

### Exercise 3 — wav2vec2 detail

**a) Raw mel-spectrogram vs. wav2vec2 (frozen) linear probe**, 4-way classification (`yes`/`no`/`stop`/`go`):

| Feature | Test Accuracy |
|---|---|
| Raw mel-spectrogram (mean-pooled) | 62.5% |
| wav2vec2 (frozen, mean-pooled) | 85.4% |

Random baseline (4-way): 25.0%. wav2vec2 improves **+22.9 points** over the raw-feature baseline.

**c) 6-class robustness check** (adding `up`/`down`):

| Classes | wav2vec2 Accuracy | Random Baseline | Margin over Random |
|---|---|---|---|
| 4 (yes/no/stop/go) | 85.4% | 25.0% | 60.4 pts |
| 6 (+up/down) | 73.6% | 16.7% | 56.9 pts |

Accuracy dropped 11.8 points, but **not proportionally** to the 50% increase in class count — the margin over the new random baseline stayed almost as strong (56.9 vs. 60.4 pts), indicating the frozen wav2vec2 representation is still largely linearly separable for the added classes.

### Exercise 4 — Voice cloning detail

**a) Per-accent synthesis + acoustic measurements** (test sentence: *"I got the job!"*):

| Accent | Duration (s) | RMS Energy | Mel Spectral Centroid |
|---|---|---|---|
| us | 1.92 | 0.0461 | 2488.3 Hz |
| br | 1.32 | 0.0805 | 2087.1 Hz |
| india | 1.72 | 0.0375 | 1880.8 Hz |
| au | 1.68 | 0.0670 | 2053.4 Hz |

**b) Identity check — cosine similarity between reference SE and each generated clip's SE:**

| Accent | Cosine Similarity to Reference |
|---|---|
| us | 0.6781 |
| br | 0.7435 |
| india | 0.7749 |
| au | 0.6571 |

Mean: **0.7134** (std: 0.0477)

**What should these similarities look like if disentanglement is working?** If OpenVoice's tone-color/style separation is doing its job, the four cosine similarities should be **high (close to 1.0) and roughly equal across all four accents** — the tone color converter is only supposed to change *how* the base-speaker output is re-rendered timbre-wise to match my reference, and accent/style is supposed to live in a completely separate part of the pipeline (the base speaker model + its prosody), not leak into the tone color embedding. If similarity is high for some accents but noticeably lower for others, that's evidence of "identity drift" — the disentanglement isn't perfect, and some base-speaker accents interact with the tone-color conversion in a way that pulls the cloned voice's *identity* (not just its accent) away from the original reference speaker.

In practice, my results fell short of that ideal: similarities clustered in the 0.66–0.77 range rather than close to 1.0, and varied by about 0.12 across accents rather than being roughly equal — India showed the closest match to my reference identity (0.775), while US and AU drifted furthest (0.678 and 0.657). This points to imperfect disentanglement: some accent-specific prosody appears to be leaking into the tone-color embedding rather than staying cleanly separated in the base-speaker stage. That said, part of this gap is likely a measurement artifact rather than a pure disentanglement failure — my generated clips were short single sentences (~1.3–1.9s) synthesizing "I got the job!", compared to the much longer ~17s reference recording used to extract the reference embedding; speaker embeddings extracted from short clips are inherently noisier, so the true disentanglement quality is probably somewhat better than these raw numbers suggest. Listening to the four clips back-to-back was consistent with the numbers — the US and AU renders sounded the least like my own voice, matching their lower similarity scores.

---

## Visualizations

- `ctc_greedy_decoding_and_cer.png` — CTC greedy decoding grid + character error rate curve (Part 3 / Exercise 2)
- `wav2vec2_vs_mel_probe.png` — wav2vec2 vs. mel-spectrogram linear probe comparison + t-SNE plot (Part 4 / Exercise 3)
- `voice_clone_mel_grid.png` — Mel spectrogram grid: same cloned voice across 4 accents (Part 5.4)
- `tokenization_comparison.png` — Tokenization comparison: NLP tokens vs. speech chars vs. accent tokens (Part 1)

*(All four are generated inline by the corresponding cells in `A6_Speech_Processing_Exercises.ipynb`; exported copies are included in this repo under `figures/`.)*

---

## Discussion

Working through speech tokenization and CTC alignment from scratch changes how I think about training a TTS or ASR model, because it makes explicit an assumption I'd otherwise take for granted from NLP: that input and output sequences line up token-for-token. In speech, they don't — a five-character word can stretch across 200 audio frames at a rate that varies with speaking speed, so an ASR model has to learn to output *something* at every frame while a separate collapsing mechanism (CTC's blank token and repeat-merging rule) turns that into the right text. This means model design for speech has to account for two axes of variability at once: what is being said, and how long each part of it takes to say — a problem NLP simply doesn't have, since text tokens are already discrete with explicit boundaries. It also reframes evaluation: a model can have a low training loss but a high character error rate at inference time if its frame-level predictions don't have enough redundancy to survive collapsing (as Exercise 2d showed directly), so alignment robustness is its own axis to test, not something that falls out of the loss curve automatically.

A tone color embedding is fundamentally a different kind of object than a text token or a CTC blank because of *what* it's derived from and *what* it selects among. A text token (or an accent tag like `[EN-US]`) is a member of a small, fixed, discrete vocabulary chosen in advance by the tokenizer's designer — it can only ever be one of a finite set of symbols, each with a dedicated embedding row learned during training. A CTC blank is even more constrained: it's a single reserved symbol whose entire meaning is structural ("no new label here yet"), not semantic. A tone color embedding, by contrast, is a continuous vector *extracted* from a specific audio sample at inference time — it isn't chosen from a predefined menu, and there's no upper bound on how many distinct tone colors it could represent, since it lives in a continuous space rather than a discrete one. All three condition the model's output, but a token or blank tells the model "produce this category of thing right now," while a tone color embedding tells the model "produce output that acoustically resembles this specific reference," which is why swapping in a different reference clip smoothly changes the output rather than jumping between a fixed set of options the way changing an accent tag does.
