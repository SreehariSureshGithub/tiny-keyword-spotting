# Tiny Keyword Spotting: "stop" detector in PyTorch

A lightweight wake-word / keyword-spotting model that listens to a 1-second audio clip and decides whether someone said **"stop"**. It's built from scratch in PyTorch: a 3-layer CNN with **~23k parameters (a 97 KB checkpoint)** running on 40-band log-mel spectrograms, small enough for a Raspberry Pi or other edge devices.

| | |
|---|---|
| Task | Binary keyword spotting ("stop" vs. any other word) |
| Data | [Google Speech Commands v0.01](https://arxiv.org/abs/1804.03209), official speaker-independent splits |
| Model | `TinyKWS`, 3 × Conv2d + global average pooling + 1 logit, 23,361 parameters |
| Test accuracy | **96.9%** on 900 held-out clips from unseen speakers (recall 98.4%, F1 0.946) |
| Inference | ~0.6 ms per clip for the model on a laptop CPU; ~3 ms including file loading and resampling |

## Key features
- **Tiny model:** 23k parameters and a 97 KB checkpoint, suitable for edge devices.
- **Speaker-independent evaluation** using the official Speech Commands split lists.
- **Robust audio input:** any sample rate, mono or stereo, WAV/FLAC/OGG.
- **Class-imbalance handling** (weighted BCE) and on-the-fly augmentation (shift, gain, noise).
- **Full pipeline as CLI scripts:** dataset preparation, training with early stopping, evaluation with a confusion matrix, inference.
- A **pre-trained model and sample recordings** are included, so the demo runs without downloading any data.

## Tech stack
Python · PyTorch · torchaudio (mel spectrograms, resampling) · soundfile · Jupyter

---

## How it works

```
 audio file ──► mono, 16 kHz ──► pad/trim to 1 s ──► log-mel spectrogram ──► TinyKWS ──► sigmoid ──► p("stop")
 (any rate,     (soundfile +      (16,000 samples)     40 mels × 101 frames     CNN                    > 0.5 → detected
  any channels)  resample)                             25 ms window, 10 ms hop
```

### 1. Audio preprocessing (`kws/audio.py`)
- Files are decoded with **soundfile** rather than `torchaudio.load`, which needs an FFmpeg/torchcodec backend that's unreliable on Windows.
- Stereo is averaged to mono, and any sample rate is resampled to **16 kHz**. For example, the included 96 kHz stereo sample works as is.
- Every clip is zero-padded or trimmed to exactly **1 second** (16,000 samples).
- Features: `MelSpectrogram(n_fft=400, hop_length=160, n_mels=40)` → `AmplitudeToDB`, giving a `1 × 40 × 101` image per clip.

### 2. Model (`kws/model.py`)

| Layer | Output shape | Params |
|---|---|---|
| Input log-mel | 1 × 40 × 101 | - |
| Conv 3×3, 16 + ReLU + MaxPool 2 | 16 × 20 × 50 | 160 |
| Conv 3×3, 32 + ReLU + MaxPool 2 | 32 × 10 × 25 | 4,640 |
| Conv 3×3, 64 + ReLU | 64 × 10 × 25 | 18,496 |
| Global average pool | 64 | - |
| Linear → 1 logit | 1 | 65 |
| **Total** | | **23,361** |

Global average pooling instead of a large fully-connected layer keeps the model tiny and makes it tolerant to small shifts in where the word appears within the second.

### 3. Training (`train.py`, `kws/dataset.py`)
- **Loss:** `BCEWithLogitsLoss` with `pos_weight = #negatives / #positives` (≈ 2.7). "stop" is the minority class, and without this weight the model learns to always answer "no".
- **Augmentation** (training only; each is applied with 50% probability): random time shift of ±100 ms, random gain of 0.7 to 1.3×, and Gaussian noise (σ = 0.005).
- **Optimiser:** Adam (lr 1e-3) with `ReduceLROnPlateau` (factor 0.5, patience 2) on validation loss.
- **Early stopping:** stops after 5 epochs without validation improvement and keeps the best checkpoint. The shipped model stopped at epoch 37 (best at epoch 32). The full training log is in [`notebooks/01_train.ipynb`](notebooks/01_train.ipynb).

### 4. Dataset (`prepare_dataset.py`)
Speech Commands ships `validation_list.txt` and `testing_list.txt`, which assign each *speaker* to one split. The prep script uses these lists, so **no speaker in the test set was heard during training**. Clips of `stop` become class `1`; clips of the other selected words become class `0`.

Split used for the shipped model:

| Split | "stop" (1) | other words (0) | Total |
|---|---|---|---|
| train | 1,886 | 5,075 | 6,961 |
| val | 246 | 649 | 895 |
| test | 249 | 651 | 900 |

---

## Results

Results for the shipped checkpoint `models/best_model.pt` on the 900-clip test split (threshold 0.5), from `python evaluate.py`:

| Metric | Test (900 clips) | Validation (895 clips) |
|---|---|---|
| Accuracy | **96.89%** | 96.54% |
| Precision | 91.08% | 92.16% |
| Recall (keyword caught) | **98.39%** | 95.53% |
| F1 | 0.946 | 0.938 |
| False-alarm rate | 3.69% | 3.08% |

Test confusion matrix:

| true \ predicted | other | stop |
|---|---|---|
| **other** | 627 | 24 |
| **stop** | 4 | 245 |

The model rarely misses a real "stop" (4 of 249). Most errors are false alarms on other words. Because the positive class is weighted during training, the model leans toward recall. Raising `--threshold` trades some recall for fewer false alarms.

On the two included real-world recordings (recorded separately at 96 kHz stereo, so quite different from the 16 kHz mono training data):

```
$ python predict.py samples/test-stop.wav samples/test-step.wav
samples/test-stop.wav: STOP detected (p=1.000)
samples/test-step.wav: no keyword (p=0.427)
```

"step" is a deliberately hard negative because it sounds very close to "stop". It's correctly rejected, but only just (p = 0.43). Near-miss words like this are where a larger negative set or a tuned threshold would help most.

---

## Quick start

```bash
git clone https://github.com/SreehariSureshGithub/tiny-keyword-spotting.git
cd tiny-keyword-spotting
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### Run the pre-trained model on your own audio
```bash
python predict.py samples/test-stop.wav samples/test-step.wav
python predict.py path/to/your_clip.wav --threshold 0.7
```
Any WAV, FLAC or OGG file works. Only the first second is analysed, so trim your recording so the word falls inside it.

From Python:
```python
from predict import KeywordDetector

detector = KeywordDetector()                    # loads models/best_model.pt
detector.probability("samples/test-stop.wav")   # -> float in [0, 1]
detector.detect("samples/test-step.wav")        # -> False
```

### Re-train from scratch
1. Download **Speech Commands v0.01** (about 1.4 GB) and extract it into `data/`:
   ```bash
   mkdir -p data/speech_commands_v0.01
   curl -L http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -o data/sc.tar.gz
   tar -xzf data/sc.tar.gz -C data/speech_commands_v0.01
   ```
2. Build the binary dataset:
   ```bash
   python prepare_dataset.py --source data/speech_commands_v0.01                     # all other words as negatives
   python prepare_dataset.py --source data/speech_commands_v0.01 --negatives go no up  # or pick a subset
   ```
   You can also train a detector for a different word with `--keyword yes`.
3. Train. The best checkpoint is saved to `models/best_model.pt`, and test metrics are printed at the end:
   ```bash
   python train.py
   ```
4. Evaluate any checkpoint or split:
   ```bash
   python evaluate.py --split test --threshold 0.5
   ```

> Using *all* 29 other words as negatives gives a much larger and more imbalanced dataset than the one above. `pos_weight` handles the imbalance automatically. The shipped model was trained on a smaller subset of 6,375 non-keyword clips, so exact numbers will differ.

A GPU is used automatically if available. Training on CPU also works because the model is tiny.

---

## Project structure

```
tiny-keyword-spotting/
├── kws/                    # reusable package
│   ├── config.py           # sample rate, feature + training hyper-parameters, paths
│   ├── audio.py            # robust loading, resampling, pad/trim, log-mel features
│   ├── dataset.py          # PyTorch Dataset + augmentation
│   └── model.py            # TinyKWS network + checkpoint loader
├── prepare_dataset.py      # Speech Commands -> Dataset/{train,val,test}/{0,1}
├── train.py                # training loop, class weighting, LR schedule, early stopping
├── evaluate.py             # accuracy, precision, recall, F1, false-alarm rate, confusion matrix
├── predict.py              # CLI + KeywordDetector class for inference
├── models/best_model.pt    # trained weights (97 KB)
├── samples/                # two real recordings: "stop" and "step"
├── notebooks/
│   ├── 01_train.ipynb      # original exploratory notebook, with the saved training log
│   └── 02_predict.ipynb    # original inference notebook
├── requirements.txt
└── LICENSE
```

The scripts are a cleaned-up version of the notebooks. The model, features and preprocessing are identical, and the shipped checkpoint gives the same predictions through both.

---

## Limitations and next steps
- **Single clip, single word.** It classifies one fixed 1-second window. It doesn't stream yet. A real-time version would slide a 1 s window over microphone input every ~100 ms and smooth the predictions.
- **No explicit noise/silence class.** Background-noise clips from Speech Commands aren't used. Adding them as negatives would cut false alarms in noisy rooms.
- **Fixed threshold.** 0.5 is a default. For a wake word you'd usually tune the threshold on the validation set for a target false-alarm rate.
- **Deployment.** The next step is exporting to ONNX / TorchScript and quantising to int8 for microcontrollers or a Raspberry Pi.

## Acknowledgements
- Dataset: P. Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, 2018 ([arXiv:1804.03209](https://arxiv.org/abs/1804.03209)), licensed CC BY 4.0. The dataset isn't redistributed in this repository.

## License
[MIT](LICENSE)
