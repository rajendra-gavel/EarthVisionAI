# EarthVision AI --- POC Development Log

## Project

**EarthVision AI --- Satellite Image Analysis & Change Detection with an
AI Assistant**

**POC constraint:** Maximum 15-day demonstrable POC. Focus on a working
AI/product demonstration rather than remote-sensing research or
infrastructure.

## Scope

### P0

-   FastAPI + AI inference/backend
-   React dashboard
-   Before/after satellite image change detection
-   Object/feature detection

### P1

-   Building segmentation
-   EarthVision AI Assistant

### P2

-   Land-cover classification
-   Advanced map capabilities only if time permits

### Out of scope

-   Kubernetes
-   Kafka
-   Database/distributed infrastructure
-   Excessive dataset plumbing
-   Resurrecting old research environments
-   Jupyter as a required workflow; Python scripts/backend are preferred

## Project directory

`/mnt/d/pocs/EarthVisionAI`

## Environment

-   Python 3.12.3
-   `.venv`: `/mnt/d/pocs/EarthVisionAI/.venv/bin/python`
-   PyTorch 2.13.0+cu130
-   Torchvision 0.28.0+cu130
-   CPU-only runtime (`torch.cuda.is_available() == False`)
-   Ultralytics 8.4.138
-   Rasterio 1.5.1
-   OpenCV 5.0.0
-   NumPy 2.5.2
-   FastAPI 0.141.1
-   TorchGeo 0.10.0
-   Node.js 22.23.2
-   Vite 8.2.2
-   Frontend packages: axios, lucide-react

## Data

### OSCD

Downloaded: `data/raw/oscd/OSCD_Images.zip`

Extracted:
`data/raw/oscd/Onera Satellite Change Detection dataset - Images/`

Abu Dhabi 13-band Sentinel-2 imagery was inspected and visualizations
were created under `results/oscd/abudhabi/`.

Train labels: `data/raw/oscd/OSCD_Train_Labels.zip`

### ICCD --- India-specific demo data

Repository: `https://github.com/noopurs018/ICCD_Dataset`

Sample: `data/raw/india/iccd/ICCD_Sample.zip`

Extracted: `data/raw/india/iccd/sample/ICCD_Sample/`

Current Agra pair: - `Agra/labeled/im1/Agra_0_2022_r00_c01.png` -
`Agra/labeled/im2/Agra_0_2023_r00_c01.png` -
`Agra/labeled/label/Agra_0_2023_r00_c01.png`

Three labeled Agra pairs are 1024x1024. Ground-truth change percentages
are approximately c01 0.55%, c02 0.92%, c03 1.24%.

**Important:** the earlier 25.41% value resulted from incorrect RGBA
channel aggregation and is not actual change.

## Change detection

### Current baseline

``` text
before + after
    -> cv2.absdiff
    -> grayscale
    -> threshold = 30
    -> binary change mask
```

Agra c01: - Size: 1024 × 1024 - Total pixels: 1,048,576 - Changed
pixels: 526,042 - Baseline change: 50.17%

Baseline evaluation:

``` text
c01: Precision 0.0082, Recall 0.7489, F1 0.0163, IoU 0.0082
c02: Precision 0.0180, Recall 0.9416, F1 0.0354, IoU 0.0180
c03: Precision 0.0187, Recall 0.7721, F1 0.0366, IoU 0.0186

Average IoU ~0.015
Average F1  ~0.029
```

Conclusion: raw differencing severely over-detects change. The 50.17%
result is a **temporary baseline demonstration**, not the final AI
result.

### ChangeFormer

Repo cloned to `external/ChangeFormer`. Old Python 3.8 / PyTorch
1.10-era dependencies were inspected. No suitable bundled checkpoint was
available.

**Decision:** do not install the old environment and do not spend POC
time resurrecting it.

### TorchGeo

TorchGeo 0.10.0 was installed. Inspected change architectures include
FCSiamConc, FCSiamDiff, ChangeStar, ChangeViT and BTC.

**Decision:** no suitable ready pretrained end-to-end model was selected
for the current ICCD demo, and we will not train a deep model on only
three labeled scenes. Model hunting is stopped for now.

## Deferred change-map refinement

After the major P0/P1 flows are demonstrable, return to change detection
for a dedicated polish pass: 1. Better learned/semantic change detector.
2. Image registration/alignment. 3. Noise reduction, morphology and
connected components. 4. Meaningful-change filtering. 5. Overlay changes
on the after image. 6. Confidence/severity and region-level statistics.
7. Evaluation against ICCD ground truth. 8. Select the most compelling
before/after pair for the final demo.

The final UI must clearly distinguish the baseline from a learned AI
detector.

## Object detection

Ultralytics inference is validated with `yolo26n-obb.pt`.

Supported classes include:
`plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool`.

**Important:** this model does not detect buildings.

Standalone RGB: `results/oscd/abudhabi/abudhabi_date1_rgb.png`

YOLO OBB successfully detected 2 roundabouts. Do not claim accuracy from
this test. Sentinel-2 resolution/domain mismatch means a more suitable
high-resolution aerial image may be selected for the final
object-detection demo.

## Backend --- working state

FastAPI: `http://localhost:9090`

Health: `GET /api/v1/health`

Change detection: `POST /api/v1/change-detection`

Service: `backend/app/services/change_service.py`

The service: - reads and validates before/after images - computes
`cv2.absdiff` - converts to grayscale - thresholds at 30 - calculates
change statistics - writes a binary change-map PNG - returns change-map
metadata

Generated map:
`data/results/change_maps/Agra_0_2022_r00_c01_to_Agra_0_2023_r00_c01_change.png`

Confirmed size: 134134 bytes.

Current response includes:

``` json
{
  "width": 1024,
  "height": 1024,
  "changed_pixels": 526042,
  "total_pixels": 1048576,
  "change_percentage": 50.17,
  "method": "pixel_difference_baseline",
  "threshold": 30,
  "change_map": "/data/results/change_maps/Agra_0_2022_r00_c01_to_Agra_0_2023_r00_c01_change.png"
}
```

Static serving: `/data -> /mnt/d/pocs/EarthVisionAI/data`

CORS: - `http://localhost:5173` - `http://127.0.0.1:5173`

WSL has had unrelated Uvicorn instances on ports 8000 and 8002.
EarthVision uses 9090; do not kill the unrelated processes when managing
EarthVision.

For clean backend testing:

``` bash
cd /mnt/d/pocs/EarthVisionAI
PYTHONPATH=/mnt/d/pocs/EarthVisionAI/backend /mnt/d/pocs/EarthVisionAI/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 9090
```

Clean runs without `--reload` are preferred during debugging because
reloader/process handling on `/mnt/d` caused confusion.

## Frontend --- working state

React/Vite app: `frontend/`

Run:

``` bash
cd /mnt/d/pocs/EarthVisionAI/frontend
npm run dev -- --host 0.0.0.0
```

Frontend: `http://localhost:5173`

Dashboard currently contains: - EarthVision AI header - AI Engine
Online - Agra 2022 before image - Agra 2023 after image - Analyze
Changes button - Analysis metrics - Baseline Analysis warning -
Generated Change Map - EarthVision AI Assistant placeholder

Current displayed baseline: - 1024 × 1024 - 526,042 changed pixels -
50.17% - Pixel Difference

The black/white change map is visible in React: - White = pixels
identified as changed by the baseline detector - Black = no detected
change

## Current architecture

``` text
                    EarthVision AI
                          |
             +------------+------------+
             |                         |
        React / Vite               FastAPI
        localhost:5173            localhost:9090
             |                         |
             | HTTP / Axios            |
             +------------>------------+
                                       |
                            ChangeDetectionService
                                       |
                         +-------------+-------------+
                         |                           |
                   Statistics                 Change Map PNG
                         |                           |
                         +-------------+-------------+
                                       |
                                       v
                                React Dashboard
```

## Completed milestones

-   [x] AI/Python environment validated
-   [x] OSCD downloaded and inspected
-   [x] ICCD India sample downloaded and inspected
-   [x] Baseline change detection implemented
-   [x] Baseline evaluated and shown to be noisy
-   [x] ChangeFormer investigated and old stack rejected for POC
-   [x] TorchGeo change models investigated; checkpoint hunting stopped
-   [x] YOLO OBB inference validated
-   [x] FastAPI backend created
-   [x] Health endpoint working
-   [x] Change-detection API working
-   [x] CORS configured
-   [x] Satellite image static serving working
-   [x] React/Vite frontend created
-   [x] Before/after images displayed
-   [x] React -\> FastAPI integration working
-   [x] Change-map generation implemented
-   [x] Change map displayed in React

## Next work

### Immediate

1.  Object/feature detection API using validated Ultralytics inference.
2.  Detection overlays and feature counts in React.
3.  EarthVision AI Assistant using actual analysis context.

### Later

Return to change detection for the dedicated refinement/polish pass.

### Final POC flow

``` text
Satellite imagery
      -> Before/After comparison
      -> Change analysis
      -> Visual change map
      -> Object/feature detection
      -> EarthVision AI Assistant
         explaining observed results
```

## Decision register

  Decision                                     Status
  -------------------------------------------- ---------------------------------
  15-day demonstrable POC                      Fixed
  React + FastAPI                              Adopted
  ICCD Agra sample                             Adopted
  Pixel difference as temporary baseline       Adopted
  ChangeFormer old environment                 Rejected
  Deep training on 3 labeled scenes            Rejected
  Continued TorchGeo checkpoint hunting        Stopped/deferred
  Jupyter as required workflow                 Rejected
  Change-map polish                            Deferred until major flows work
  YOLO OBB supported feature classes           Adopted for next milestone
  Building detection with current YOLO26 OBB   Not supported
  Kubernetes/Kafka/DB                          Out of scope

## Working principle

**Ship the demonstrable POC first; refine the highest-value
change-detection capability after the full workflow exists.**

The current pixel-difference map is a temporary baseline and must not be
presented as a high-quality semantic AI result.
