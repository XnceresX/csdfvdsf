# SAFEPASS: Multi-Camera Strategic Topology Tracker & Multi-Modal Person Re-ID
## Comprehensive Technical Architecture, Model Pipeline, and System Explainer

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Challenge in Multi-Camera CCTV Surveillance
In real-world surveillance (e.g., apartment complexes, corporate campuses, hospitals, transit hubs), tracking an individual across multiple CCTV cameras presents three fundamental challenges:

1. **Non-Overlapping Fields of View (Blind Spots):** Cameras do not have 100% continuous coverage. A person walking from a parking lot into a stairwell disappears from one camera and only appears in another after a transit delay of several seconds. Traditional continuous computer vision tracking algorithms (e.g., DeepSORT, ByteTrack) fail because tracklets break across blind spots.
2. **Severe Computational Bottleneck (The $N$-Camera GPU Explosion):** Running deep object detectors (YOLO) and multi-task feature extraction networks (OSNet, ArcFace, OpenVINO attributes) continuously at 15–30 FPS across 8, 16, or 64 cameras requires massive multi-GPU hardware clusters. In standard surveillance, 85–95% of camera feeds at any given second show empty hallways or static scenes. Running full deep inference on all feeds continuously is computationally wasteful.
3. **Severe Visual Degradation & Domain Shift:** CCTV cameras are mounted near ceilings or high walls (top-down pitch angles), producing severe perspective distortion. Faces are frequently occluded, blurred, low-resolution ($<30\times30$ pixels), or pointing away from the camera. Lighting shifts dramatically between outdoor sunlight and indoor incandescent stairwells, rendering RGB color matching unreliable.

### 1.2 The SAFEPASS Solution
SAFEPASS solves these challenges through a unified **Multi-Modal Feature Fusion Engine** paired with a **Strategic Topology Tracker**:

* **Strategic 1-Camera Tracking (87.5% Compute / GPU Savings):** Instead of running AI inference on all 8 cameras simultaneously, the system runs deep detection and feature extraction **only on the currently occupied camera**. The other 7 cameras run at **0% GPU** in cached idle mode. When the subject approaches an exit zone or leaves the frame, the system awakens **only topological neighbor cameras** during a mathematically constrained transit time window $[T_{\min}, T_{\max}]$.
* **Multi-Modal Person Re-ID Fusion:** Combines four complementary feature extractors:
  1. **Body Embedding:** OSNet (Omni-Scale Network, Market-1501 pretrained, 512-d).
  2. **Perceptual Color:** CIELAB $\Delta E_{ab}^*$ sampling on human anatomical segments (immune to lighting shifts).
  3. **Demographic / Visual Attributes:** OpenVINO Crossroad multi-attribute classifier (backpack, bag, hat, jacket, pants).
  4. **Facial Biometrics:** ArcFace / MobileFaceNet (512-d) with an **autonomous CCTV quality gate** ($w_{\text{face}} = 0.0$ when quality $< 0.35$).
* **Automated Final Destination Resolution:** Maps physical transit to an architectural topology graph, automatically inferring terminal destinations (e.g., "Flat 1A", "Flat 2B", "Main Gate Exit").

---

## 2. System Architecture & Component Diagram

```
                                      +---------------------------------------------+
                                      |            OPERATOR INTAKE WIZARD           |
                                      |  (Target Appearance Query & Last Seen Cam)  |
                                      +---------------------------------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------------------------------------------+
|                                             TOPOLOGY TRACKER STATE MACHINE                                             |
|                                                                                                                       |
|    [ STANDBY ] --------> [ ACQUIRING ] --------> [ TRACKING ] --------> [ TRANSITIONING ] --------> [ RESOLVED ]      |
|    (0% Compute)          (Scan Entry Cams)      (Active Cam Only)        (Awaken Neighbors)       (Terminal Zone)     |
+-----------------------------------------------------------------------------------------------------------------------+
            |                                      |                                |
            |                                      v                                v
            |                      +-------------------------------+  +-------------------------------+
            |                      |   ACTIVE CAMERA CV PIPELINE   |  |   NEIGHBOR PREDICTIVE SEARCH  |
            |                      |   (YOLOv8 + ByteTrack)        |  |   (Transit Bound Check)       |
            |                      +-------------------------------+  +-------------------------------+
            |                                      |
            |                                      v
            |                      +----------------------------------------------------------------+
            |                      |                   MULTI-MODAL FEATURE EXTRACTION                |
            |                      |                                                                |
            |                      |  1. Body Re-ID:     OSNet Market-1501 (512-d embedding)        |
            |                      |  2. Perceptual Col: CIELAB ΔE (Torso & Leg Anatomical Zones)  |
            |                      |  3. Attributes:     OpenVINO Crossroad (Backpack, Hat, etc.)   |
            |                      |  4. Face ArcFace:   Quality-Gated (w_face = 0 if Q < 0.35)     |
            |                      +----------------------------------------------------------------+
            |                                      |
            |                                      v
            |                      +----------------------------------------------------------------+
            |                      |                ADAPTIVE MULTI-MODAL FUSION ENGINE              |
            |                      |       S_fuse = (w_os*S_os + w_col*S_col + w_att*S_att) / W     |
            |                      +----------------------------------------------------------------+
            |                                      |
            |                                      v
            |                      +----------------------------------------------------------------+
            |                      |            ROLLING TARGET EMBEDDING GALLERY (N <= 10)          |
            |                      +----------------------------------------------------------------+
            |
            v
+-----------------------------------------------------------------------------------------------------------------------+
|                                           FASTAPI VMS SURVEILLANCE DASHBOARD                                          |
|                                                                                                                       |
|  * 8-Stream MJPEG Multiplexing with Surveillance HUD        * Interactive SVG Blueprint with Dynamic FOV Cones        |
|  * Sub-Feed Auto-Routing for Top 2 Neighbors                * VMS Timeline Scrubber Bar with Event Markers            |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 3. Deep Learning Models & Feature Extractors

### 3.1 Object Detection & Tracklet Generation (YOLOv8 + ByteTrack)
* **Model:** YOLOv8 Nano (`yolov8n.pt` / `yolov8n-seg.pt`).
* **Input Resolution:** Dynamic ($960 \times 1080$ portrait CCTV feeds resized to $640 \times 640$ letterbox for inference).
* **Role:** Detects bounding boxes $[x_1, y_1, x_2, y_2]$ classified as `person` (Class 0) with confidence threshold $\tau_{\text{conf}} \ge 0.35$ and IoU threshold $\tau_{\text{iou}} = 0.45$.
* **Tracking:** ByteTrack associates detections across consecutive frames inside the same camera using Kalman filter state prediction and two-stage Hungarian matching on detection bounding boxes:
  $$\text{IoU}(B_1, B_2) = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}$$
  Assigns persistent local track IDs (`track_id`) within the active camera.

---

### 3.2 Deep Body Re-Identification (OSNet)
* **Architecture:** Omni-Scale Network (`osnet_x1_0`).
* **Weights:** `models/osnet_x1_0_market1501.pth` (10.4 MB, trained on Market-1501 dataset).
* **Execution:** FP16 GPU inference with CPU fallback (`src/embed.py`).
* **Design Philosophy:** Standard CNNs (ResNet) operate at a fixed convolutional receptive field scale. In CCTV surveillance, discriminative person features exist across vastly different scales:
  * **Fine Scale:** Shoe logos, watch, necklace, glasses.
  * **Medium Scale:** Pockets, sleeve stripes, waist pouch, bag strap.
  * **Coarse Scale:** Jacket color, silhouette, long coat pattern.
  OSNet achieves omni-scale learning using lightweight bottleneck depthwise separable streams with multiple receptive fields, combining them via dynamic channel-wise gate weights:
  $$\mathbf{y} = \sum_{t=1}^T \mathbf{g}_t(\mathbf{x}) \odot \mathbf{f}_t(\mathbf{x})$$
  where $\mathbf{f}_t(\mathbf{x})$ is stream $t$ with receptive field $k_t$, and $\mathbf{g}_t(\mathbf{x})$ is a channel-attention gating vector.
* **Output:** 512-dimensional vector $\mathbf{v}_{\text{raw}} \in \mathbb{R}^{512}$, normalized to the unit hypersphere:
  $$\mathbf{e} = \frac{\mathbf{v}_{\text{raw}}}{\|\mathbf{v}_{\text{raw}}\|_2}, \quad \|\mathbf{e}\|_2 = 1.0$$
* **Similarity Metric:** Cosine similarity:
  $$S_{\text{osnet}}(\mathbf{e}_1, \mathbf{e}_2) = \mathbf{e}_1 \cdot \mathbf{e}_2 = \cos(\theta)$$

---

### 3.3 Perceptual CIELAB Color Space Extraction
* **Module:** `src/color.py`.
* **The Pitfall of RGB/HSV in CCTV:** RGB and HSV are severely distorted by ambient lighting shifts, shadows, and camera white-balance differences. Two photos of the same navy jacket can produce completely distinct HSV hues under outdoor sunlight vs indoor fluorescent corridors.
* **CIELAB Standard:** Uses the CIE $L^*a^*b^*$ color space, mathematically modeled after human visual perception:
  * $L^*$: Perceptual lightness ($0 = \text{black}$, $100 = \text{diffuse white}$).
  * $a^*$: Red-green opponent axis (negative = green, positive = red).
  * $b^*$: Yellow-blue opponent axis (negative = blue, positive = yellow).
* **Anatomical Landmark Sampling:**
  Rather than averaging the entire crop (which includes background pixels and walkways), the crop is divided into anatomical zones:
  * **Upper Garment Zone:** $y \in [0.15 \cdot H, 0.45 \cdot H]$, $x \in [0.20 \cdot W, 0.80 \cdot W]$.
  * **Lower Garment Zone:** $y \in [0.55 \cdot H, 0.85 \cdot H]$, $x \in [0.20 \cdot W, 0.80 \cdot W]$.
* **Color Distance Metric (CIE76 $\Delta E_{ab}^*$):**
  $$\Delta E_{ab}^* = \sqrt{(L_1^* - L_2^*)^2 + (a_1^* - a_2^*)^2 + (b_1^* - b_2^*)^2}$$
  A distance of $\Delta E < 2.3$ is imperceptible to the human eye. The distance is mapped to similarity:
  $$S_{\text{color}} = \exp\left(-\frac{\Delta E_{ab}^*}{25.0}\right) \in [0.0, 1.0]$$

---

### 3.4 OpenVINO Demographic & Clothing Attributes
* **Model:** `person-attributes-recognition-crossroad-0230` (`.xml` + `.bin`, 2.9 MB).
* **Inference Engine:** OpenVINO Runtime CPU/iGPU (`src/attributes.py`).
* **Output Attributes:** Soft probability distribution over 8 visual attributes:
  * `has_backpack` ($p \in [0, 1]$)
  * `has_bag` / `has_handbag` ($p \in [0, 1]$)
  * `has_hat` / `has_cap` ($p \in [0, 1]$)
  * `has_coat_jacket` ($p \in [0, 1]$)
  * `has_longpants` vs shorts ($p \in [0, 1]$)
  * `is_male` ($p \in [0, 1]$)
* **Attribute Similarity Metric:**
  For any binary attribute check against target criteria with expected value $y \in \{0, 1\}$:
  $$S_{\text{attr}} = \frac{1}{|A|} \sum_{a \in A} \left[ y_a \cdot p_a + (1 - y_a) \cdot (1 - p_a) \right]$$

---

### 3.5 Facial Biometrics (ArcFace) & Quality Gating
* **Model:** ArcFace / MobileFaceNet (`data/models/arcface.onnx`, 136.6 MB, `face/extractor.py`).
* **Output:** 512-dimensional normalized facial embedding $\mathbf{f} \in \mathbb{R}^{512}$.
* **The Top-Down CCTV Problem:** In top-down CCTV, faces are frequently captured at steep angles ($>45^\circ$ downward pitch), occluded by caps, or heavily motion-blurred. Feeding a corrupted facial crop into Re-ID drops accuracy significantly.
* **Autonomous CCTV Quality Gate (`face/quality.py`):**
  Calculates an composite facial quality score $Q_{\text{face}} \in [0.0, 1.0]$ based on:
  1. **Sharpness (Laplacian Variance):** $V_{\text{lap}} = \text{Var}(\nabla^2 I_{\text{gray}})$.
  2. **Resolution:** Minimum face box dimension $\min(W_{\text{face}}, H_{\text{face}})$.
  3. **Head Pose Angles:** Yaw and pitch deviation from frontal.
* **Strict Suppression Rule:**
  $$\text{If } Q_{\text{face}} < 0.35 \text{ or face not detected} \implies w_{\text{face}} = 0.0$$
  This ensures that noisy, distorted top-down CCTV face detections **never** corrupt body Re-ID and attribute matching.

---

### 3.6 Crop Quality Assessment
* **Module:** `src/quality.py`.
* Evaluates pedestrian crop quality before adding to the target gallery:
  * Resolution score: penalizes crops smaller than $128 \times 64$ pixels.
  * Aspect ratio score: penalizes non-pedestrian aspect ratios (optimal $H/W \approx 2.0$ to $3.0$).
  * Edge density & contrast.
  Low-quality crops ($Q_{\text{crop}} < 0.40$) are discarded to prevent gallery contamination.

---

## 4. Multi-Modal Fusion Engine

The fusion engine (`src/fusion.py`) combines the individual modality similarities into a single global matching score $S_{\text{fuse}} \in [0.0, 1.0]$.

### 4.1 Adaptive Rule-Based Fusion Formula
$$S_{\text{fuse}} = \frac{w_{\text{osnet}} \cdot S_{\text{osnet}} + w_{\text{color}} \cdot S_{\text{color}} + w_{\text{attr}} \cdot S_{\text{attr}} + w_{\text{face}} \cdot S_{\text{face}}}{w_{\text{osnet}} + w_{\text{color}} + w_{\text{attr}} + w_{\text{face}}}$$

* **Default Baseline Weights:**
  * $w_{\text{osnet}} = 0.50$ (primary global identity carrier)
  * $w_{\text{color}} = 0.25$ (CIELAB clothing color)
  * $w_{\text{attr}} = 0.25$ (backpack, hat, jacket, pants)
  * $w_{\text{face}} = 0.00$ (suppressed on CCTV; promoted to $0.35$ only when high-res frontal face is present)
* **Dynamic Modality Dropout:** If an attribute or face cannot be computed, its weight is automatically zeroed out, and the remaining weights are renormalized to sum to $1.0$.

### 4.2 Target Embedding Gallery (Rolling Bank)
* Maintains a rolling bank $\mathcal{G} = \{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_N\}$ of verified target embeddings (maximum size $N = 10$).
* When matching a candidate crop with embedding $\mathbf{e}_{\text{cand}}$:
  $$S_{\text{osnet}} = \max_{\mathbf{e}_i \in \mathcal{G}} (\mathbf{e}_{\text{cand}} \cdot \mathbf{e}_i)$$
* When a high-confidence match ($S_{\text{fuse}} \ge 0.55$) with acceptable quality ($Q \ge 0.40$) is detected, $\mathbf{e}_{\text{cand}}$ is pushed into the gallery $\mathcal{G}$, enabling the system to adapt to perspective and lighting changes as the person moves through the building.

---

## 5. Strategic Topology Tracker & State Machine

The core intelligence governing camera scheduling and GPU savings is the **Strategic Topology Tracker** (`src/topology_tracker.py`).

### 5.1 Topology Graph Representation
The Mayfair Apartment complex consists of 8 CCTV cameras across 4 vertical levels, modeled as an adjacency graph $G = (V, E)$:

```
[TERRACE / ROOFTOP]                                     CAM 8
                                                          |
                                           [2.0s - 25.0s] | (Stairs to Terrace)
                                                          |
[2ND FLOOR]                                             CAM 7
                                                          |
                                           [2.0s - 20.0s] | (Stairs 1F <-> 2F)
                                                          |
[1ST FLOOR]                                             CAM 6
                                                         / \
                                        [2.0s - 25.0s]  /   \  [2.0s - 25.0s]
                                     (Stairs to Ground)/     \ (Stairs to Ground)
                                                      /       \
[GROUND FLOOR]             CAM 4 (Entrance Left) ----+         +---- CAM 3 (Entrance Right)
                            |         \                             /         |
             [2.0s - 20.0s] |          \------ [1.0s - 12.0s] -----/          | [1.5s - 18.0s]
              (NW Passage)  |                                                 | (SE Approach)
                            v                                                 v
                       CAM 5 (North Walkway) ----------- CAM 1 (East Walkway) -+ CAM 2 (SE Driveway)
                                              [2.0s - 20.0s]
                                               (NE Corner)
```

#### Detailed Camera Node Map
| Camera ID | Location Name | Floor | FOV Angle | Topological Neighbors | Connected Exit Destinations |
|:---|:---|:---:|:---:|:---|:---|
| **CAM 1** | Ground Floor East Walkway | 0 (GF) | $180^\circ$ | `CAM 2`, `CAM 5` | SE Corner Gate, North Walkway |
| **CAM 2** | Ground Floor SE Driveway / Side Walkway | 0 (GF) | $225^\circ$ | `CAM 1`, `CAM 3` | **Main Gate Exit**, East Walkway, Building Entrance |
| **CAM 3** | Ground Floor Building Entrance Right | 0 (GF) | $210^\circ$ | `CAM 2`, `CAM 4`, `CAM 6` | **Main Gate Exit**, Ground Stairwell to 1F, Entrance Left, SE Driveway |
| **CAM 4** | Ground Floor Building Entrance Left | 0 (GF) | $330^\circ$ | `CAM 3`, `CAM 5`, `CAM 6` | **Main Gate Exit**, **Room/Store**, Ground Stairwell to 1F, North Walkway, Entrance Right |
| **CAM 5** | Ground Floor North Back Walkway | 0 (GF) | $180^\circ$ | `CAM 1`, `CAM 4` | East Walkway, Entrance Left |
| **CAM 6** | First Floor Stairwell Landing | 1 (1F) | $90^\circ$ | `CAM 3`, `CAM 4`, `CAM 7` | **Flat 1A**, **Flat 1B**, Stairs Down to Ground, Stairs Up to 2F |
| **CAM 7** | Second Floor Stairwell Landing | 2 (2F) | $90^\circ$ | `CAM 6`, `CAM 8` | **Flat 2A**, **Flat 2B**, Stairs Down to 1F, Stairs Up to Terrace |
| **CAM 8** | Top of Building / Terrace | 3 (TR) | $180^\circ$ | `CAM 7` | **Terrace Door**, **Terrace Area**, Stairs Down to 2F |

---

### 5.2 The 87.5% Compute Savings Mechanism
In an 8-camera setup:
* **Naive Approach:** 8 streams $\times$ YOLOv8 $\times$ OSNet $= 8.0 \times$ GPU load continuously.
* **SAFEPASS Strategic Tracking:**
  $$\text{GPU Load} = \begin{cases} 
  1.0 \times \text{ (only the active camera running inference)} & \text{during tracking} \\ 
  2.0 \times \text{ (active + neighbor being tested)} & \text{during handoff moments} \\ 
  0.0 \times & \text{when standby} 
  \end{cases}$$
  $$\text{Average Compute Savings} = \frac{8 - 1}{8} \times 100\% = \mathbf{87.5\% \text{ GPU Savings}}$$

---

### 5.3 Global State Machine Lifecycle

```
       +--------------------+
       |   UNINITIALIZED    | <------------------------------------+
       +--------------------+                                      |
                 |                                                 |
         [ Operator Query ]                                        |
                 v                                                 |
       +--------------------+                                      |
       |     ACQUIRING      |                                      |
       | (Scan Entry Cams)  |                                      |
       +--------------------+                                      |
                 |                                                 |
       [ Match Score >= 0.65 ]                                     |
                 v                                                 |
       +--------------------+                                      |
+----> |      TRACKING      |                                      |
|      | (Active Cam Only)  |                                      |
|      +--------------------+                                      |
|                |                                                 |
|     [ Target Enters Exit Zone OR Disappears ]                    |
|                v                                                 |
|      +--------------------+                                      |
|      |   TRANSITIONING    |                                      |
|      | (Awaken Neighbors) |                                      |
|      +--------------------+                                      |
|             /        \                                           |
|   [ Found in         [ Reached Terminal                          |
|     Neighbor ]         Exit Zone ]                               |
+---------+                    \                                   |
                                v                                  |
                      +--------------------+                       |
                      |      RESOLVED      | ----------------------+
                      | (Final Destination)|     [ Operator Reset ]
                      +--------------------+
```

1. **`UNINITIALIZED` / `STANDBY`:**
   * All 8 camera nodes are idle ($0\%$ GPU).
   * Frame 0 of each video is cached in RAM to serve instant snapshot/stream thumbnails without decoding overhead.
2. **`ACQUIRING` (Cold Start):**
   * Triggered when operator submits an appearance query (e.g., *Black jacket, Blue trousers, CAM 2*).
   * Activates only the designated entry camera(s).
   * YOLOv8 detects candidates; attributes and colors are scored against query criteria.
   * If a candidate scores $S \ge 0.65$ (`acquisition_threshold`), the system **automatically locks** onto the subject, initializes the embedding gallery $\mathcal{G}$, and transitions to `TRACKING`.
3. **`TRACKING` (Single-Camera Active Tracking):**
   * The active camera runs YOLOv8 + ByteTrack + OSNet feature extraction.
   * All other 7 cameras run at $0\%$ GPU.
   * If target position intersects an exit zone or tracklet is lost for $>1.5$ seconds:
     * System records time of departure $t_{\text{depart}}$ and exit trajectory.
     * Transitions to `TRANSITIONING`.
4. **`TRANSITIONING` (Predictive Neighbor Handoff):**
   * Only topological neighbors defined in `configs/topology.yaml` are awakened to state `SEARCHING_NEIGHBOR`.
   * **Transit Time Gate:** A candidate in neighbor camera $C_j$ is only accepted if the elapsed time $\Delta t = t_{\text{arrive}} - t_{\text{depart}}$ satisfies the physical transit bounds:
     $$T_{\min}(C_i, C_j) \le \Delta t \le T_{\max}(C_i, C_j)$$
   * If matched ($S_{\text{fuse}} \ge 0.55$), the target is handed off to $C_j$, which becomes the new active camera. Previous camera drops to $0\%$ GPU idle.
5. **`RESOLVED` (Case Destination Closure):**
   * If the target enters a designated terminal zone (e.g., enters the door of "Flat 1A", "Flat 1B", "Flat 2A", "Flat 2B", or departs through "Main Gate Exit"), tracking halts.
   * System emits a final audit verdict: inferred destination, total elapsed transit time, confidence score, and complete camera trajectory path:
     $$\text{CAM 2} \longrightarrow \text{CAM 3} \longrightarrow \text{CAM 5} \longrightarrow \mathbf{\text{Flat 1A}}$$

---

## 6. End-to-End Walkthrough: Step-by-Step Execution

Here is the exact step-by-step lifecycle of tracking a person from building entry to apartment arrival:

### Scenario: Individual in Dark Jacket & Blue Pants Enters Main Gate
```
Time: 00:00.0 | Global State: UNINITIALIZED
- Operator opens http://127.0.0.1:8000
- Dashboard launches centered "Target Acquisition Wizard"
- Operator inputs:
    * Last seen: CAM 2 (SE Corner Gate)
    * Upper color: Black
    * Lower color: Blue
    * Pants: Long Pants / Trousers
    * Outerwear: Wearing Jacket
- Operator clicks "START TRACKING TARGET"
```

```
Time: 00:00.1 | State Transition -> ACQUIRING
- Wizard auto-dismisses completely.
- CAM 2 is assigned ComputeState.ACTIVE; CAM 1,3,4,5,6,7,8 remain ComputeState.IDLE (0% GPU).
- YOLOv8 runs on CAM 2 at t = 0.0s.
- Candidate Track #1 detected at coordinates [810, 520, 930, 780].
- CIELAB extractor measures: Upper = Black (ΔE = 1.2), Lower = Blue (ΔE = 1.8).
- Attribute classifier outputs: has_jacket = 0.88, has_longpants = 0.94.
- Fusion score S_fuse = 0.82 >= 0.65 threshold.
- Auto-lock triggered! Target registered.
- Embedding vector e_0 (512-d) saved to Gallery Bank G.
- State Transition -> TRACKING (CAM 2)
```

```
Time: 00:00.1 - 00:08.5 | State: TRACKING (CAM 2)
- Person walks across courtyard in CAM 2.
- ByteTrack updates bounding box.
- OSNet extracts embeddings; cosine similarity to bank G stays at 0.89 - 0.94.
- SVG Map displays pulsing mint FOV cone on CAM 2.
- Primary Video Pane displays high-res CAM 2 feed with bounding box HUD.
- At t = 00:08.5, person steps into Exit Zone: "To Parking Right / Stairs" [x in 0.00..0.35, y in 0.20..0.70].
```

```
Time: 00:08.6 | State Transition -> TRANSITIONING
- Person leaves CAM 2 frame. Tracklet lost in CAM 2.
- Departure recorded: from_cam = "cam2", t_depart = 8.6s.
- Topology graph checks neighbors of CAM 2:
    * CAM 1 (East Walkway): Transit window [1.0s, 15.0s]
    * CAM 3 (Parking Right): Transit window [1.5s, 18.0s]
- CAM 1 and CAM 3 awaken to ComputeState.SEARCHING_NEIGHBOR.
- CAM 4, 5, 6, 7, 8 remain at 0% GPU.
- Sub-Feed 1 automatically streams CAM 1; Sub-Feed 2 automatically streams CAM 3.
```

```
Time: 00:11.2 | Re-Acquisition in CAM 3
- At t = 11.2s (elapsed Δt = 2.6s, within [1.5s, 18.0s] bound):
- Person appears in CAM 3 frame.
- Candidate crop extracted by YOLOv8 on CAM 3.
- OSNet computes embedding e_cand.
- Similarity against Bank G: max(e_cand · e_i) = 0.78 >= 0.55 reacquisition threshold.
- Re-acquisition confirmed!
- State Transition -> TRACKING (CAM 3)
- CAM 1 drops back to IDLE (0% GPU).
- Audit event logged: "CAM 2 ➔ CAM 3 handoff confirmed (Score: 78%)".
- Trajectory polyline draws green dashed vector from CAM 2 to CAM 3 on Ground Floor map.
```

```
Time: 00:15.0 - 00:22.0 | CAM 3 -> CAM 5 (Stairwell Transition to 1F)
- Target walks across Parking Right towards stairwell door.
- Enters Exit Zone "Ground Stairwell" at t = 00:22.0.
- Target departs CAM 3. Neighbors awakened: CAM 5 (1F Landing, transit [2.0s, 25.0s]).
- At t = 00:27.4 (Δt = 5.4s):
- Person appears on CAM 5 (First Floor Stairwell Landing).
- OSNet match against Gallery Bank: 0.81.
- State Transition -> TRACKING (CAM 5).
- Map automatically shifts active tab to "1st Floor".
- SVG shows pulsing mint cone on CAM 5 (First Floor).
```

```
Time: 00:31.0 | Destination Resolution -> Flat 1A
- Person walks up to door on the left of 1F landing.
- Person's bounding box intersects Terminal Exit Zone: "Flat 1A" [x in 0.00..0.28, y in 0.30..0.75].
- Target enters Flat 1A door and is no longer visible.
- State Transition -> RESOLVED
- System issues Final Audit Verdict:
    * Destination: "Flat 1A"
    * Last Camera: CAM 5 (1st Floor Landing)
    * Camera Path: CAM 2 -> CAM 3 -> CAM 5 -> Flat 1A
    * Confidence: 95.0%
- Status badge switches to RESOLVED; red beacon marks Flat 1A on 1st Floor blueprint.
- All cameras return to 0% GPU standby.
```

---

## 7. Web Dashboard Architecture ("Modern Dark" VMS)

The frontend is an enterprise-grade surveillance operations dashboard inspired by **Synology Surveillance Station ("Monitor Center")** built with vanilla ES6+ JS, SVG, and HTML5, powered by FastAPI (`web/app.py`).

### 7.1 Visual Layout Components

```
+-------------------------------------------------------------------------------------------------------+
| TOP NAVBAR: Brand Badge | Live Mint Status Signal | Clock | NEW SEARCH | RESET | CALIBRATION          |
+-------------------+---------------------------------------------------+-------------------------------+
| LEFT NAVIGATION   | CENTER-LEFT COLUMN                                | RIGHT MONITOR COLUMN          |
|                   |                                                   |                               |
| [CAMERAS & NODES] | [MAPS SERVICE: Architectural Floor Plan]          | [PRIMARY CCTV FEED: CAM X]    |
| * CAM 1 [GF]      | * Floor Tabs: Ground | 1F | 2F | Terrace          | * Live MJPEG Feed             |
| * CAM 2 [GF] (ACT)| * SVG Canvas with Embedded Blueprint              | * Surveillance HUD Overlay    |
| * CAM 3 [GF]      | * Dynamic FOV Sector Cones (Mint/Amber/Slate)     | * OSNet Bank Badge N=...      |
| * CAM 4 [GF]      | * Breadcrumb Trajectory Polylines                 |                               |
| * CAM 5 [1F]      | * Target Detection Ripple Beacon                  +-------------------------------+
| * CAM 6 [GF]      +---------------------------------------------------+ [TOPOLOGICAL NEIGHBOR EXITS]  |
| * CAM 7 [2F]      | [BOTTOM-LEFT OPERATIONAL WIDGETS]                 | * Sub-Feed 1 (Top Neighbor)   |
| * CAM 8 [TR]      | * Target Localization Status Table                | * Sub-Feed 2 (2nd Neighbor)   |
|                   | * Topology Audit Log (Timestamped Transitions)    | * Quick Switcher: CAM 1 .. 8  |
+-------------------+---------------------------------------------------+-------------------------------+
| BOTTOM DOCK: Digital Timecode (00:00 / 15:00) | Transport Controls | 15-Min Scrubber with Mint Ticks  |
+-------------------------------------------------------------------------------------------------------+
```

### 7.2 Design Tokens ("Modern Dark")
* **Background Surface:** `--dp-color-ground: #050505` (deep near-black).
* **Panels & Cards:** `--dp-color-panel: #0e0e0e` (opaque dark graphite with `--dp-radius-panel: 6px`).
* **Hairline Borders:** `--dp-color-hairline: #222222` and `--dp-color-hairline-2: #303030`.
* **Live Mint Signal:** `--dp-color-live: #3fd6c6` (active tracking status, live camera indicator, and timeline event markers).
* **Amber Transit Signal:** `--dp-color-amber: #e5a93c` (neighbor search and pending handoffs).
* **Typography:**
  * Display / Headers: Geist / Modern Dark Grotesk (`"Modern Dark Grotesk", "Geist", Arial, sans-serif`).
  * Numerical Values, Timestamps, Codes: Geist Mono (`"Modern Dark Mono", "Geist Mono", monospace`).

### 7.3 Real-Time Communication Protocols
* **MJPEG Multipart Video Streaming (`/api/stream/{camera_id}`):**
  Serves multipart JPEG frames (`multipart/x-mixed-replace; boundary=frame`) directly from memory buffers. Camera frames are rendered with surveillance HUD overlays (timestamp, active tracklet ID, bounding boxes).
* **WebSocket Live Telemetry (`/ws`):**
  Broadcasts JSON state updates at 150ms intervals containing current simulation time, compute load states across all 8 cameras, active candidates, target gallery size, and event audit logs.
* **REST API:**
  * `POST /api/search`: Initiates cold-start acquisition.
  * `POST /api/playback`: Controls simulation play/pause and speeds ($1\times, 2\times, 4\times$).
  * `POST /api/seek`: Moves global timeline to exact timestamp $t$.
  * `POST /api/offsets`: Calibrates per-camera time synchronization offsets live.
  * `POST /api/remap`: Remaps camera node to an alternate `.mp4` file path without server restart.
  * `POST /api/reset`: Resets state machine to uninitialized standby.

---

## 8. Directory Structure & Key Files Reference

```
project_zainside/
├── configs/
│   ├── default.yaml              # Detection, Re-ID, and fusion model thresholds
│   └── topology.yaml             # 8-camera graph, exit zones, transit bounds, map coordinates
├── data/
│   ├── models/
│   │   ├── arcface.onnx          # 512-d ArcFace facial feature extractor (136.6 MB)
│   │   ├── osnet_x1_0_market1501.pth # Market-1501 trained body Re-ID weights (10.4 MB)
│   │   ├── yolov8n.pt            # YOLOv8 pedestrian detector
│   │   └── attributes/           # OpenVINO person attributes crossroad model
│   ├── videos/                   # Video feeds: CAM 1.mp4 through CAM 8.mp4
│   └── mayfair/floorplans/       # Architectural blueprints: ground_floor, floor1, floor2, terrace
├── face/
│   ├── detector.py               # Face detection & bounding box crop extraction
│   ├── extractor.py              # ArcFace ONNX 512-d embedding extractor
│   └── quality.py                # CCTV face quality filter (w_face = 0 if Q < 0.35)
├── models/
│   ├── fusion_lr.pkl             # Pre-trained logistic regression fusion model
│   └── osnet_x1_0_market1501.pth # OSNet weights backup
├── src/
│   ├── attributes.py             # OpenVINO 8-attribute recognition
│   ├── color.py                  # CIELAB perceptual color extractor & Delta E metric
│   ├── detect_track.py           # YOLOv8 + ByteTrack pedestrian tracking
│   ├── embed.py                  # OSNet 512-d embedding extractor with FP16 GPU support
│   ├── fusion.py                 # Multi-modal fusion engine (adaptive_rule)
│   ├── gallery.py                # Rolling target gallery & embedding bank
│   ├── identity.py               # Long-term identity assignment & multi-camera clustering
│   ├── quality.py                # Pedestrian crop quality & sharpness estimator
│   ├── pipeline.py               # Combined detection, attribute, color, embedding pipeline
│   └── topology_tracker.py       # Strategic 8-camera topology graph tracker & state machine
├── tests/
│   ├── test_color.py             # CIELAB accuracy & color difference unit tests
│   ├── test_detect_and_pipeline.py# Detection & pipeline initialization tests
│   ├── test_fusion.py            # Multi-modal fusion logic tests
│   ├── test_identity.py          # Identity manager tests
│   ├── test_topology_tracker.py  # Topology graph, transitions, and handoff tests
│   └── test_web_api.py           # FastAPI REST & WebSocket endpoint tests
├── web/
│   ├── app.py                    # FastAPI server & background playback orchestrator
│   ├── static/
│   │   ├── css/style.css         # Modern Dark VMS stylesheet
│   │   ├── js/app.js             # VMS client, SVG map controller, WebSocket receiver
│   │   └── maps/                 # Static blueprint PNGs
│   └── templates/
│       └── index.html            # Single-page surveillance operator console
├── pytest.ini                    # Pytest configuration
├── run_safepass_web.bat          # 1-click startup launcher for Windows
└── yolov8n-seg.pt                # YOLOv8 segmentation weights
```

---

## 9. Summary of Key Hyperparameters & Thresholds

| Parameter | Value | Description | File Location |
|:---|:---:|:---|:---|
| `duration_sec` | `900.0s` | Simulation duration (15 minutes per camera) | `configs/topology.yaml` |
| `fps` | `15.0` | Surveillance video framerate | `configs/topology.yaml` |
| `acquisition_threshold` | `0.65` | Minimum multi-modal match score for auto-lock on target | `configs/topology.yaml` |
| `reacquisition_fusion_threshold` | `0.55` | Minimum fusion score to confirm handoff in neighbor camera | `configs/topology.yaml` |
| `lost_timeout_sec` | `1.5s` | Time target must be absent before triggering neighbor search | `configs/topology.yaml` |
| `neighbor_search_timeout_sec` | `30.0s` | Max duration neighbor cameras stay in search mode before declaring lost | `configs/topology.yaml` |
| `detection_conf_threshold` | `0.35` | Minimum confidence threshold for YOLOv8 pedestrian detections | `configs/default.yaml` |
| `gallery_update_quality_threshold` | `0.40` | Minimum quality required to store a new embedding in target bank | `configs/default.yaml` |
| `face_quality_threshold` | `0.35` | Minimum quality for facial biometrics; if lower, $w_{\text{face}} = 0.0$ | `face/quality.py` |
| `gallery_max_size` | `10` | Max number of rolling embeddings stored in target bank | `src/topology_tracker.py` |
| `osnet_embedding_dim` | `512` | Dimensionality of OSNet body embedding vector | `src/embed.py` |
| `arcface_embedding_dim` | `512` | Dimensionality of ArcFace facial embedding vector | `face/extractor.py` |
| `cielab_sigma` | `25.0` | Scaling factor for mapping CIELAB $\Delta E$ to similarity | `src/color.py` |

---

## 10. How to Run & Verify

1. **Run Full Test Suite:**
   ```bash
   C:\project\safepass_appearance\venv\Scripts\python.exe -m pytest -q
   # Result: 81 passed in ~20 seconds
   ```
2. **Start Web Server:**
   ```bash
   C:\project\safepass_appearance\venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000
   ```
3. **Access Dashboard:** Open your browser to `http://127.0.0.1:8000`.
4. **Deploy Target Search:** The Target Acquisition Wizard pops up automatically. Select an entry camera (e.g., `CAM 2`), choose clothing colors/attributes, and click **START TRACKING TARGET**. The wizard dismisses, and you can watch the strategic topology tracker lock onto the person and follow them through the building.

