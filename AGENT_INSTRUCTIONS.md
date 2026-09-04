# F170 R&D — Local Agent Experiment Guide

**Target hardware:** RTX-class GPU + Ryzen-class CPU on your local machine.
**Goal:** Push the F170 frozen-backbone + tiny-head + FedAvg loop toward a
production-ready, robust, general-purpose foundation.

---

## 0. Setup

```bash
# Pull the F170 working tree
git clone https://github.com/SuperInstance/federated-tinyml-vessel.git
cd federated-tinyml-vessel

# Create a venv (don't pollute the global site-packages)
python3.11 -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA (RTX 30/40 series)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# F170 deps
pip install numpy requests sounddevice soundfile

# Optional but recommended: real audio testing
pip install librosa pyserial paho-mqtt
```

Verify:
```bash
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'devs:', torch.cuda.device_count())"
# expect: cuda: True devs: 1
nvidia-smi
```

---

## 1. The 5 Breakthrough Experiments

These are the experiments that will move F170 from "R&D demo" to "general-purpose foundation". Run them in order; each builds on the last.

### 1.1 Stress Test: 100 simulated devices, real connectivity drops

The current F170 paper shows 5 devices in a clean lab. Real vessel fleets have
100 vessels, intermittent satellite/VHF links, stragglers, drop-outs.

**Build:** `experiments/exp1_100_devices.py`
- 100 devices simulated in parallel with `concurrent.futures.ProcessPoolExecutor` (Ryzen will love this)
- Each device has a random drop-out probability 0.1-0.3 per round
- Each device has a random data volume 100-1000 samples per round
- Aggregator: FedAvg + staleness-weighted averaging (older updates have lower weight)
- 200 rounds
- Measure: convergence speed, peak accuracy, byte budget per round, total wall-clock

**Target:** Achieve 90%+ accuracy by round 200 with the 1.3 KB head, byte budget <500 KB per device for the full run.

### 1.2 Robust Aggregation: Byzatine, Median, Trimmed Mean

Real vessel networks are hostile. A single malicious or broken device shouldn't
poison the global head.

**Build:** `experiments/exp2_robust_agg.py`
- 10 devices, 1 of which is a malicious device flipping head signs randomly
- Compare 4 aggregators: FedAvg (baseline, fails), Coordinate-wise Median, Trimmed Mean (top/bottom 20% removed), Krum (nearest neighbor)
- Measure: accuracy drop vs. clean baseline, convergence stability

**Target:** Robust aggregators maintain >85% accuracy under 1/10 adversary; FedAvg degrades to <60%.

### 1.3 Real Backbone: PANNs CNN14 (or YAMNet) on ESC-50

The hand-crafted backbone hits 47.5% on real audio. State-of-the-art audio
backbones hit 80%+ on ESC-50.

**Build:** `experiments/exp3_panns_backbone.py`
- Use `PANNs_inference.py` or load a pretrained YAMNet from `torchaudio.models.yamnet`
- Extract 1024-dim or 2048-dim embeddings (vs our 64-dim hand-crafted)
- Project down to 64-dim with a learned random projection (frozen, seeded)
- Train the 1.3 KB head via FedAvg
- 5 vessel-relevant ESC-50 categories

**Target:** 80%+ test accuracy on ESC-50 with the same 1.3 KB head, proving the head is robust to backbone choice.

### 1.4 MCU Deployment: Compile to TFLite Micro, Flash to ESP32-S3

The C port exists. The 173-byte int4 head is the killer artifact. Flash it to a real ESP32-S3 (N16R8: 16MB flash, 8MB PSRAM, 240 MHz Xtensa LX7).

**Build:** `experiments/exp4_esp32_deploy/`
- Convert the int4 head to a TFLite Micro `.tflite` file
- Add a quantize-aware-training loop (QAT) that learns the int4 weights directly
- Write a `main.cpp` for ESP32-S3: reads mic, runs backbone, runs head, reports via WiFi/MQTT
- Measure: inference latency, peak RAM, power consumption (INA219 current sensor)

**Target:** <1ms inference latency, <50KB peak RAM, <50mA peak current. The head must run on bare metal.

### 1.5 Multi-Task Heads: Anomaly + Fuel + Tangle on one Backbone

The architecture is backbone-agnostic. Prove it by training 3 tiny heads on the
same backbone, each for a different vessel task.

**Build:** `experiments/exp5_multitask.py`
- Backbone: shared frozen CNN (vibration or acoustic features)
- Head A (1.3 KB): anomaly detection on engine vibration
- Head B (1.3 KB): fuel consumption regression
- Head C (1.3 KB): line_tangle classifier (the killer app)
- Each task has its own labels but shares the backbone
- 3 separate FedAvg aggregators, or one multi-head aggregator

**Target:** All 3 heads converge independently, sharing the same backbone, no interference.

---

## 2. The Long-Form Investigation: Compression

The head is 1.3 KB. Can we push it lower without losing accuracy?

**Build:** `experiments/compression_sweep.py`
- Sweep: int4 → int2 → binary → sparsity
- For each: 100 federated runs, measure final acc, byte budget, latency
- Plot: accuracy vs. bytes

**Targets to explore:**
- int2 (4 values): 0.5x further compression = 87 bytes
- ternary (-1, 0, +1): 1.58 bits/weight = 65 bytes
- binary: 1 bit/weight = 41 bytes
- 90% sparse + int4: 27 bytes

**Open question:** is there a *theoretical* minimum? Probably 32 bytes (8 uint32 weights + 5 biases at int4). Can we train a model that fits in 32 bytes?

---

## 3. The Open Questions to Attack

1. **Optimal backbone for vessel audio**: PANNs vs. YAMNet vs. hand-crafted MFCC+PCA. Each has different accuracy / size / latency trade-offs.

2. **Adaptive quantization**: A head that's fp32 for early rounds (when accuracy matters) and int4 for late rounds (when bytes matter).

3. **Heterogeneous head sizes**: Some devices might support 5 KB heads, others only 500 bytes. Can we mix in one federation?

4. **Continual learning**: The head must keep learning without forgetting. Add a tiny EWC penalty to the loss.

5. **Personalization**: Keep a local copy of the head. Mix with the global head via a learned gate.

6. **Decentralized aggregation**: No central server. Vessels gossip with neighbors via LoRa / VHF. Same byte budget, different topology.

7. **Differential privacy**: Add Gaussian noise to the head before upload. Measure accuracy vs. ε (privacy budget).

---

## 4. The Benchmark: ESP32-S3 Reference Implementation

To make F170 a real foundation, ship a complete reference that anyone can fork.

**`f170-reference-esp32/`** (new repo):
- `backbone/` — the frozen backbone compiled to TFLite Micro (or hand-crafted C)
- `head/` — the 173-byte int4 head as a `.tflite` (or a C struct)
- `aggregator/` — the Python aggregator, runnable on a Raspberry Pi
- `wire/` — MQTT or HTTP client
- `flash.sh` — one-command flash to ESP32-S3
- `tests/` — full test suite, including a hardware-in-the-loop test

**Target:** A captain with a $20 ESP32 + a $5 mic + a $5 IMU can run a real federated learning experiment at sea, contributing to a global model.

---

## 5. The Papers to Write

After each successful experiment, write a short paper (4-8 pages):

| # | Title | Key result |
|---|-------|-----------|
| F171 | Real PANNs backbone on ESC-50 | 80%+ test acc with 1.3 KB head |
| F172 | Robust aggregation under 30% adversaries | Trimmed Mean wins |
| F173 | ESP32-S3 reference deployment | <1ms inference, <50mA |
| F174 | 100-device fleet simulation | Converges under stragglers |
| F175 | Multi-task heads on one backbone | 3 tasks, 1 backbone, no interference |
| F176 | Compression sweep: int4 → binary | Minimum head size is 32 bytes |

---

## 6. The Tokens to Use (Local Agent)

- **GROQ_TOKEN** — fast iterative ideation, Q&A, paper drafting. 14 models including gpt-oss-120b.
- **DeepInfra** — fallback for heavier reasoning. 189 models.
- **PyPI / npm** — publish the libraries.
- **GitHub** — push everything.

Don't waste local CPU/GPU on ideation. Use GROQ. Save the RTX for actual training and benchmarking.

---

## 7. The Token Economy (Your Constraint)

- **Your local agent tokens are precious.** Don't burn them on planning.
- **GROQ is fast and cheap for ideation.** Use it for paper drafting, experimental design, literature review.
- **DeepInfra for synthesis.** Use it when GROQ isn't enough.
- **Local RTX for compute.** Use it for actual training, not for talking to LLMs.

A 1-hour local experiment that uses 100 API calls on GROQ (~$0.05) and 4 hours of GPU time (free) is way better than 1 hour of agent time burning tokens.

---

## 8. The Reporting Cadence

After each experiment:
1. Run, measure, plot
2. Write 1 page of findings
3. Commit + push
4. Update live canon if novel
5. Tell the director (me) what you found

I'll integrate your findings into the F170 series as new papers (F171-F176+).

---

## 9. The KILLER App: Line Tangle Classifier

Always keep this in mind: the captain is the witness. The wrist-mounted
classifier that fires a haptic ~3 seconds before hooks pull out of the shoot
saves $20K of gear and prevents a 6-hour retrieval mission. That's the reason
this exists. Everything else is in service of that.

---

## 10. The Iron Rules

1. **The 1.3 KB head is the contract.** Don't grow it without a measured reason.
2. **The backbone is frozen.** Don't unfreeze it unless the experiment explicitly tests that.
3. **The hash is the address.** Every head has an FNV-1a 64-bit identity. Verify it.
4. **The wire is the only thing on the network.** No raw audio, no raw features.
5. **The substrate is the porthole.** Same head, byte-exact in Python/C/Rust/JS.
6. **The captain is the witness.** Production = wearable + haptic + 6-second lead time.

Get to work.
