# Kuala Lumpur Road Dataset Anonymizer (`KL-Road-Anonymizer`)

[![Dataset Status](https://img.shields.io/badge/Dataset_Status-Work_In_Progress-orange.svg)](#dataset-publication--roadmap)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-Grounding_DINO_ViT-green.svg)](https://huggingface.co/IDEA-Research/grounding-dino-base)

An automated, open-source privacy anonymization pipeline designed specifically for urban Southeast Asian road footage (Kuala Lumpur, Malaysia). 

This pipeline uses a **Vision-Language Transformer (Grounding DINO)** coupled with a **Spatial Vehicle ROI Engine** to automatically detect, filter, and obfuscate **vehicle registration plates** and **human faces/heads** (including helmeted motorcycle riders) prior to dataset release.

> **Note on Dataset Availability:** This dataset is currently a **work in progress**. The code and anonymization pipeline are being shared now for community review. Once data curation and final quality checks are finalized, the dataset will be published, and the download link will be updated here.

## Technical Highlights

- Built with PyTorch + Hugging Face Transformers
- Uses Grounding DINO Vision Transformer
- Designed a custom spatial ROI filtering algorithm to reduce false positives
- Hardware acceleration with CUDA and Apple Silicon (MPS)
- Reproducible research pipeline
---

## File Structure

```
KL-Road-Anonymizer/
│
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
│
├── docs/
│   └── sample comparison.png          # Side-by-side comparison of original vs anonymized frames
│
└── Notebooks/
    └── v2-0-anonymising-dataset.ipynb # Main processing notebook

```

---

## Dataset Collection & Preprocessing

The underlying dataset captures real-world driving conditions across diverse Kuala Lumpur roadways (highways, dense city intersections, motorcycle lanes, and suburban streets):

1. **Hardware Capture:** Video was recorded using an **iPhone 14 Pro** mounted while riding a bicycle through Kuala Lumpur traffic.
2. **Frame Extraction:** High-definition video streams were sampled and converted into individual image frames at **2 frames per second (2 FPS)**. Sampling at 2 FPS preserves temporal continuity for sequence tracking while reducing redundant near-identical frames.
3. **Anonymization Stage:** All extracted 2 FPS frames are processed through this pipeline to remove personally identifiable information (PII) before publication, aligning with global data privacy compliance standards (such as Malaysia's PDPA).

---

## The Technical Journey: Why Vision Transformers?

Building a privacy filter for Kuala Lumpur road imagery presents unique challenges: high motorcycle density, varied plate formats (black acrylic, square, slanted, custom fonts), extreme camera angles, and high-contrast ambient lighting.

![Sample Comparison](docs/sample%20comparison.png)

*Figure: Side-by-side comparison showing original extracted frames (left) and anonymized outputs (right) with targeted Gaussian blur applied to license plates and human faces/heads.*

### Phase 1: OpenCV Haar Cascades (Failed)
* **Approach:** Classical classifiers (`haarcascade_frontalface_default.xml` and `haarcascade_russian_plate_number.xml`).
* **Failure Modes:** Severe false-positive rates on car grilles, rims, asphalt patterns, and road shadows. Failed to detect black Malaysian license plates and motorcycle riders.

### Phase 2: YOLOv8 Convolutional Neural Networks (Inconsistent)
* **Approach:** CNN-based bounding box detection fine-tuned on faces and license plates.
* **Failure Modes:** CNNs rely on fixed local grid anchors. Highly angled license plates (e.g., turning vehicles or overtaking motorcycles) and tilted heads resulted in partial crops (blurring only half a plate) or total detection dropouts.

### Phase 3: Grounding DINO Vision Transformer + Spatial ROI Filtering (Current Solution)
* **Approach:** Deployed **Grounding DINO** (`IDEA-Research/grounding-dino-base`), a Vision Transformer architecture that uses cross-modal attention between text prompts and visual features across the entire image space.
* **Result:** High recall across arbitrary angles, extreme lighting variations, and helmeted heads.

---

## Technical Innovation: Spatial Vehicle ROI Containment

Vision Transformers are highly sensitive; low confidence thresholds can occasionally pick up high-contrast asphalt textures or speed bump patterns as "plates." 

To solve this, the pipeline enforces a **Spatial Context Containment Check**:

1. The script calculates the exact center point (centroid) of every candidate license plate detection:
   $$\text{Centroid} = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$$
2. A candidate license plate is **only blurred** if its centroid lies within the bounding box of a detected vehicle (`car`, `motorcycle`, `bus`, `truck`) plus a 10% safety margin.
3. Detections found on bare road asphalt or background structures are automatically discarded as noise.


```
            Raw Frame Input (2 FPS)
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │ Grounding DINO Vision Transformer   │
    │ Prompts: "vehicle, car, motorcycle, │
    │  license plate, human face, head"   │
    └──────────────────┬──────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐       ┌──────────────────┐
│  Vehicle Boxes   │       │ Candidate Plates │
│  & Face/Heads    │       │     & Faces      │
└────────┬─────────┘       └────────┬─────────┘
         │                          │
         └─────────────┬────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │   Spatial Containment Check         │
    │   Is Plate Center inside a Vehicle? │
    └──────────────────┬──────────────────┘
             ┌─────────┴─────────┐
             │                   │
          [ YES ]             [ NO ]
             │                   │
             ▼                   ▼
    ┌────────────────┐  ┌────────────────┐
    │ Apply Expanded │  │ Reject Blur    │
    │ Gaussian Blur  │  │ (Asphalt/Noise)│
    └────────────────┘  └────────────────┘

```


---

## Features

* **Zero-Shot Language Grounding:** Uses `IDEA-Research/grounding-dino-base` to catch targets independent of size, rotation, or viewing angle.
* **Asphalt False-Positive Suppression:** Filters candidate plates by verifying their spatial placement on vehicle ROIs.
* **Helmet & Head Coverage:** Prompts for `human face` and `head` to catch motorcycle riders wearing helmets or turned away from the camera.
* **Asymmetric Expansion Padding:** Applies 25% horizontal and 15% vertical bounding box expansion so plate characters along borders do not leak.
* **Hardware Accelerated:** Automatic detection for CUDA GPUs and Apple Silicon Metal Performance Shaders (`mps`).

---

## Installation & Usage

### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/your-username/KL-Road-Anonymizer.git](https://github.com/your-username/KL-Road-Anonymizer.git)
cd KL-Road-Anonymizer
pip install -r requirements.txt

```

`requirements.txt`:

```text
torch
torchvision
transformers
opencv-python
pillow
huggingface_hub

```

### 2. Run Processing Script

Set your frame input directory and desired output directory in `main.py` (or run directly in Kaggle/Jupyter):

```python
INPUT_DIR = "/path/to/extracted_frames"
OUTPUT_DIR = "/path/to/anonymized_frames"

```

Execute the pipeline:

```bash
python main.py

```

---

## Dataset Publication & Roadmap

* [x] Record raw Kuala Lumpur road video footage (iPhone 14 Pro)
* [x] Extract frame sequences at 2 FPS
* [x] Build Vision-Transformer anonymization pipeline
* [ ] Complete full-dataset anonymization pass
* [ ] Run manual Quality Control (QC) validation
* [ ] Publish dataset on public repository (Kaggle / Hugging Face / IEEE DataPort)
* [ ] Update dataset URL in this README

---

## License & Citation

Distributed under the MIT License. See `LICENSE` for details.

If you use this pipeline or tool in your work, please cite:

```bibtex
@dataset{kl_road_dataset_in_progress,
  author    = {Mohammed Abdul Al Arafat Tanzin},
  title     = {Kuala Lumpur Urban Road Video Dataset for Autonomous Perception},
  year      = {2026},
  note      = {Work in Progress - Dataset Link Coming Soon},
  url       = {[https://github.com/TanzinAbdul/Kuala-Lumpur-Road-Dataset-Anonymizee](https://github.com/TanzinAbdul/Kuala-Lumpur-Road-Dataset-Anonymize)}
}


```
