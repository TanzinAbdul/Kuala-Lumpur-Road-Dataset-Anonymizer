# Kuala Lumpur Road Dataset Anonymizer (`KL-Road-Anonymizer`)

[![Dataset Status](https://img.shields.io/badge/Dataset_Status-Work_In_Progress-orange.svg)](#dataset-publication--roadmap)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-Grounding_DINO_ViT-green.svg)](https://huggingface.co/IDEA-Research/grounding-dino-base)
[![arXiv](https://img.shields.io/badge/arXiv-2608.14724-b31b1b.svg)](https://arxiv.org/abs/2608.14724)

An automated, open-source privacy anonymization pipeline designed specifically for urban Southeast Asian road footage (Kuala Lumpur, Malaysia). 

This pipeline uses a **Vision-Language Transformer (Grounding DINO)** coupled with a **Spatial Vehicle ROI Engine** to automatically detect, filter, and obfuscate **vehicle registration plates** and **human faces/heads** (including helmeted motorcycle riders) prior to dataset release.

> **Note on Dataset Availability:** This dataset is currently a **work in progress**. The code and anonymization pipeline are being shared now for community review. Once data curation and final quality checks are finalized, the dataset will be published, and the download link will be updated here.

---

## 📄 Paper

This work is described in our paper:

> **Privacy-Preserving Dataset Curation for Kuala Lumpur Urban Traffic: Grounded Vision-Language Detection with Spatial Vehicle-Context Filtering**  
> Mohammed Abdul Al Arafat Tanzin, Rudzidatul Akmam Dziyauddin  
> arXiv:2608.14724, 2026

[![arXiv](https://img.shields.io/badge/arXiv-2608.14724-b31b1b.svg)](https://arxiv.org/abs/2608.14724)

```bibtex
@misc{tanzin2026privacypreservingdatasetcurationkuala,
      title={Privacy-Preserving Dataset Curation for Kuala Lumpur Urban Traffic: Grounded Vision-Language Detection with Spatial Vehicle-Context Filtering}, 
      author={Mohammed Abdul Al Arafat Tanzin and Rudzidatul Akmam Dziyauddin},
      year={2026},
      eprint={2608.14724},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.14724}, 
}
```

---

## Technical Highlights

- Built with PyTorch + Hugging Face Transformers
- Uses Grounding DINO Vision Transformer
- Designed a custom spatial ROI filtering algorithm to reduce false positives
- Hardware acceleration with CUDA (multi-GPU support)
- Batch processing for efficient video anonymization
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
    ├── v2-0-anonymising-dataset.ipynb # Main processing notebook (images)
    └── v2-0-anonymising-dataset-video.ipynb      # Video processing notebook (NEW!)

```

---

## 📹 Video Processing Notebook

We've added a **video processing notebook** (`video-anonymisation.ipynb`) that extends the pipeline from images to full video streams. This notebook processes videos at **1080p resolution** and applies the same privacy-preserving anonymization in real-time.

### Video Pipeline Features:

1. **Frame-by-Frame Processing**: Reads video frames, applies detection, and writes anonymized output
2. **Multi-GPU Acceleration**: Uses `DataParallel` to distribute batch inference across multiple GPUs
3. **Non-Maximum Suppression (NMS)**: Removes duplicate detections per category with IoU threshold of 0.4
4. **Batch Processing**: Processes 8 frames simultaneously for optimal GPU utilization
5. **1080p Output**: Saves anonymized video at 1920×1080 resolution, 30 FPS
6. **Progress Tracking**: Real-time progress bar with FPS and ETA
7. **Visual Feedback**: 
   - 🟢 **Green boxes**: Vehicles with blurred plates
   - 🔴 **Red boxes**: License plates (blurred inside vehicles)
   - 🟡 **Yellow boxes**: Blurred faces/heads
   - Status overlay showing counts and total blurs

### NMS Implementation (IoU-based Duplicate Removal):

The pipeline implements **Non-Maximum Suppression (NMS)** to eliminate multiple overlapping detections of the same object:

1. Detections are sorted by confidence score (highest first)
2. The box with highest confidence is selected and added to the output
3. IoU (Intersection over Union) is calculated between the selected box and all remaining boxes:
   ```
   IoU = Area_of_Intersection / Area_of_Union
   ```
4. Boxes with IoU > 0.4 are considered duplicates and removed
5. Process repeats until all boxes are processed

This ensures each object has only one bounding box, significantly reducing false positives.

### Video Processing Flow:

```
Input Video → Frame Extraction → Batch Collection (8 frames)
    ↓
Grounding DINO Inference (Multi-GPU)
    ↓
NMS (IoU = 0.4)
    ↓
Spatial ROI Containment Check
    ↓
Gaussian Blurring (Adaptive Kernel)
    ↓
Draw ONLY Blurred Regions
    ↓
Output Video (1080p, 30 FPS)
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

### Algorithm Pseudocode:

```
Algorithm: Spatial Vehicle ROI Containment Filtering

For each candidate plate detection:
    1. Compute plate centroid: (cx, cy)
    2. For each detected vehicle box:
        3. Expand vehicle box by 10% margin
        4. If (cx, cy) is inside expanded vehicle box:
            5. Valid plate → Apply blur
            6. Break loop
    7. If no vehicles detected AND plate confidence > 0.35:
        8. Valid plate → Apply blur (fallback)
    9. Otherwise:
        10. Reject as background noise
```

### Pipeline Flow Diagram:

```
            Raw Frame Input (Video)
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
* **Asymmetric Expansion Padding:** Applies 10% horizontal and 50% vertical bounding box expansion for plates (10% uniform for faces) so plate characters along borders do not leak.
* **Non-Maximum Suppression (NMS):** Removes duplicate detections with IoU threshold of 0.4, ensuring one box per object.
* **Hardware Accelerated:** Automatic detection for CUDA GPUs (supports multi-GPU via DataParallel).
* **Video Processing:** Full video anonymization at 1080p, 30 FPS with real-time progress tracking.

---

## Installation & Usage

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/TanzinAbdul/Kuala-Lumpur-Road-Dataset-Anonymizer.git
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
tqdm
```

### 2. Run Image Processing (Notebook)
Open and run the Jupyter notebook:

```bash
jupyter notebook Notebooks/v2-0-anonymising-dataset.ipynb
```

### 3. Run Video Processing (Notebook)
For video anonymization:

```bash
jupyter notebook Notebooks/video-anonymisation.ipynb
```

### 4. Video Processing Configuration

In the video notebook, you can configure:

```python
VIDEO_INPUT = "/path/to/video.mov"
OUTPUT_DIR = "/path/to/output"
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
BATCH_SIZE = 8  # Frames per batch (multi-GPU)
OUTPUT_FPS = 30
BOX_THRESHOLD = 0.20
IOU_THRESHOLD = 0.4  # NMS IoU threshold
```

---

## Dataset Publication & Roadmap

* [x] Record raw Kuala Lumpur road video footage (iPhone 14 Pro)
* [x] Extract frame sequences at 2 FPS
* [x] Build Vision-Transformer anonymization pipeline
* [x] Video processing implementation (1080p, 30 FPS)
* [x] NMS implementation for duplicate removal
* [ ] Complete full-dataset anonymization pass
* [ ] Run manual Quality Control (QC) validation
* [ ] Publish dataset on public repository (Kaggle / Hugging Face / IEEE DataPort)
* [ ] Update dataset URL in this README

---

## Citation

If you use this pipeline or tool in your work, please cite our paper:

```bibtex
@misc{tanzin2026privacypreservingdatasetcurationkuala,
      title={Privacy-Preserving Dataset Curation for Kuala Lumpur Urban Traffic: Grounded Vision-Language Detection with Spatial Vehicle-Context Filtering}, 
      author={Mohammed Abdul Al Arafat Tanzin and Rudzidatul Akmam Dziyauddin},
      year={2026},
      eprint={2608.14724},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.14724}, 
}
```

Or the dataset citation:

```bibtex
@dataset{kl_road_dataset_in_progress,
  author    = {Mohammed Abdul Al Arafat Tanzin},
  title     = {Kuala Lumpur Urban Road Video Dataset for Autonomous Perception},
  year      = {2026},
  note      = {Work in Progress - Dataset Link Coming Soon},
  url       = {https://github.com/TanzinAbdul/Kuala-Lumpur-Road-Dataset-Anonymizer}
}
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Contact

**Author:** Mohammed Abdul Al Arafat Tanzin  
**Supervisor:** Dr. Rudzidatul Akmam Dziyauddin  
**Institution:** Universiti Teknologi Malaysia (UTM)  
**Email:** tanzinabdul@gmail.com  
**GitHub:** [@TanzinAbdul](https://github.com/TanzinAbdul)

---

## Acknowledgments

- Grounding DINO team (IDEA-Research) for the vision-language model
- Hugging Face for the Transformers library
- Universiti Teknologi Malaysia for research support
```

## Key Updates Made:

1. **Added arXiv Paper Section**: With badge and BibTeX citation at the top
2. **Video Processing Notebook Section**: Detailed explanation of the new video pipeline
3. **NMS Explanation**: Clear description of IoU-based duplicate removal
4. **Video Pipeline Flow Diagram**: Visual representation of the video processing flow
5. **Updated Features**: Added video processing, NMS, and expanded padding details
6. **Updated Installation**: Added `tqdm` to requirements
7. **Updated Roadmap**: Added video processing and NMS implementation as completed
8. **Dual Citation**: Both paper and dataset citations included
9. **Added Contact Section**: With author and supervisor information
10. **Better Structure**: More organized sections with emoji indicators
