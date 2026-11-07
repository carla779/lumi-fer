# LUMI — Real-time Facial Expression Recognition

Facial Expression Recognition (FER) module for **LUMI**, a paediatric social robot designed to help reduce children's anxiety before clinical procedures. The system detects a face from the robot's eye camera, classifies it into one of five emotions (**happy, sad, angry, surprised, neutral**) and smooths the output over time so the robot can react to how the child is feeling.

Built for my 3rd-year Integrated Design Project (EMS690U, BEng Biomedical Engineering, Queen Mary University of London, 2025–26).

## Pipeline

```
camera frame
   └─► BlazeFace face detection (MediaPipe)
         └─► crop + resize to 224×224
               └─► CNN classifier ──┐  EfficientNet-B0 (v9)
                                    ├─ average softmax (ensemble)
               └─► CNN classifier ──┘  Swin-Tiny (v10)
                     └─► temporal smoothing (rolling window)
                           └─► emotion + confidence
```

Training ideas that went into the final model (v9):

- **Multi-dataset training** on RAF-DB, FERPlus, ExpW and FER2013 combined into one unified CSV
- **Action-unit (AU) auxiliary loss**: nine geometric face features (eye openness, mouth opening/width, brow raise…) extracted with MediaPipe Face Mesh and predicted as a side task
- **FERPlus soft labels** (crowd-vote distributions instead of a single hard label)
- **Dataset × class balanced sampling**, MixUp (α = 0.2), label smoothing and targeted augmentation for minority classes
- ImageNet-pretrained backbone, frozen for the first 5 epochs then fine-tuned

## Results

**Benchmark test sets** (pooled across the four datasets):

| Model | Backbone | Params | Test macro-F1 |
|---|---|---|---|
| v9 | EfficientNet-B0 | 4.0 M | **0.62** |

**Ablation** (one component removed at a time, change in macro-F1 vs v9):

| Change | Δ macro-F1 |
|---|---|
| RAF-DB only (no multi-dataset) | −0.09 |
| ResNet-18 instead of EfficientNet-B0 | −0.04 |
| No AU loss / no balanced sampling / hard labels / no targeted augmentation | −0.02 to −0.03 each |

**Leave-one-dataset-out** generalisation (train on three, test on the held-out one): RAF-DB 0.53 · FERPlus 0.55 · ExpW 0.51 · FER2013 0.48.

**Latency** (laptop CPU, 4 threads, 224×224, batch 1): EfficientNet-B0 28.5 ms median, Swin-Tiny 196 ms median. Projected Raspberry Pi 5 INT8 latency for EfficientNet-B0 is about 23 ms. See [`evaluation/deployment_proxy_summary.txt`](evaluation/deployment_proxy_summary.txt).

**Human validation study**: 27 participants (Fitzpatrick skin types I–VI) watched 50 emotion-eliciting clips while the system classified their faces live (1,348 valid trials). Spontaneous, real-world expressions turned out much harder than benchmark photos: the smoothed output was biased towards *neutral* (macro-F1 0.27 against participants' self-reported emotion), with *happy* the only class above F1 0.5. Closing this gap between posed benchmark data and natural expressions is the main area for future work.

## Repository layout

| Path | Contents |
|---|---|
| `build_multidataset_fer7_csv_v2.py` | Builds the unified CSV from RAF-DB, FERPlus, ExpW (+ `fer2013_appender.py` for FER2013) |
| `preprocess_faces_from_csv_v3.py` | Detects, crops and saves 224×224 faces for every image in the CSV |
| `extract_auproxy_facemesh_from_csv.py` | Extracts the 9 AU-proxy features with MediaPipe Face Mesh |
| `cnn_train_multidataset_fer7_soft_au.py` | Main training script (EfficientNet-B0 / Swin-Tiny / ResNet, AU loss, soft labels, MixUp) |
| `cnn_webcam_ensemble.py` | **Live demo**: ensemble of v9 + v10 on a webcam |
| `cnn_webcam_fer7_multi.py` | Live demo with a single model |
| `cnn_webcam_experiment.py` | Live FER logger used during the participant study |
| `eye_tracking_facemesh.py` | Gaze tracking used alongside FER |
| `models_cnn_multi/` | Model config (`*_meta.json`); put the `*_best.pt` weights here |
| `evaluation/` | Laptop/Raspberry Pi benchmarks, ablation and leave-one-dataset-out kits |
| `training_kits/v9_retrain/` | Scripts to retrain and evaluate v9 end to end on Windows |
| `experiment/` | Participant study: stimulus preparation, PsychoPy experiment, Pi ↔ PsychoPy bridge |
| `robot/` | LUMI eye/gaze control on the Raspberry Pi |
| `notebooks/` | Google Colab training notebook |
| `legacy/` | Earlier iterations: per-emotion logistic-regression models on hand-crafted landmark features, FER2013 / RAF-DB-only CNNs |

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the live demo

Trained weights are not stored in the repo (they're too large and derived from research-only datasets). Place `fer7_v9_5class_mixup_best.pt` (and optionally `fer7_v10_swin_5class_best.pt`) in `models_cnn_multi/`, then:

```bash
python cnn_webcam_ensemble.py                                   # v9 + v10 ensemble
python cnn_webcam_ensemble.py --run_names fer7_v9_5class_mixup  # v9 only (fast)
python cnn_webcam_ensemble.py --cam 1 --smooth 5                # other camera, shorter smoothing
```

### Train from scratch

The datasets are not included and must be requested from their owners under their research licences:
[RAF-DB](http://www.whdeng.cn/raf/model1.html) · [FERPlus](https://github.com/microsoft/FERPlus) · [ExpW](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html) · [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

```bash
python build_multidataset_fer7_csv_v2.py --datasets_root DATASETS --out_csv unified_fer7.csv
python preprocess_faces_from_csv_v3.py --in_csv unified_fer7.csv --out_dir processed_faces --out_csv unified_fer7_processed.csv
python extract_auproxy_facemesh_from_csv.py --in_csv unified_fer7_processed.csv --out_csv unified_fer7_processed_au.csv
python cnn_train_multidataset_fer7_soft_au.py --csv unified_fer7_processed_au.csv \
    --arch efficientnet_b0 --pretrained --img_size 224 --epochs 50 --lr 1e-4 \
    --lambda_au 0.05 --mixup_alpha 0.2 --run_name fer7_v9_5class_mixup
```

Training v9 takes about 4–6 hours on a single consumer GPU.

## Data and ethics

No participant data (video, demographics, FER logs or self-reports) is included in this repository. The validation study was run with informed consent, and its data stays private under the study's ethics terms. The emotion-eliciting stimulus clips are not redistributed either; `experiment/stimulus_list.csv` only describes them.

## Acknowledgements

MediaPipe face detection and face landmarker models © Google (Apache 2.0). Pretrained backbones from `torchvision` and `timm`.
