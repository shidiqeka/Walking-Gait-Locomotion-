# Walking-Gait-Locomotion: Humanoid Bipedal Locomotion for BarelangS

[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%203.x-blue)](https://mujoco.org)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%202.x-EE4C2C)](https://pytorch.org)
[![Robotics](https://img.shields.io/badge/Humanoid-BarelangS%2021DOF-brightgreen)](https://github.com/shidiqeka/Walking-Gait-Locomotion-)

Sistem lokomosi bipedal humanoid lengkap untuk robot **BarelangS** (21 DOF, 20 aktuator) menggunakan:
1. **SuperMimic & Bio-mechanical Human Motion Capture (MoCap)**:
   - Differentiable Neural World Model (*Backpropagation Through Time*).
   - Kendali kemudi interaktif real-time (*WASD Teleoperation*).
   - Refleks penyeimbang dinamis (*Capture-Point & Stepping Recovery*) tahan dorongan $\pm 26\text{ N}$.
2. **Procedural 3D Terrain PPO Reinforcement Learning**:
   - Kurikulum medan bergelombang, lereng, dan bebatuan (*Heightfield terrain*).
   - Refleks keseimbangan IMU torso aktif (*Active PD balance compensation*).

---

## 1. Quick Start (Cara Menjalankan)

### A. Kemudi Interaktif & Autonomous Striker AI ([T] RoboCup Mode)
Kendalikan robot layaknya karakter game dengan gerak jalan manusia asli, atau aktifkan kecerdasan buatan penendang bola otonom (*Autonomous Striker AI*):
```bash
python steer_barelang_supermimic.py
```
* **`[T]`**: **Toggle Autonomous Striker AI (RoboCup AI Mode)** $\to$ Robot otomatis mencari bola, lari kencang (*sprint*), mengunci posisi di belakang bola menghadap gawang, dan menendang keras ke gawang!
* **`[W]` / `[S]`**: Maju / Mundur (*Walk* hingga *Sprint* $1.62\text{ m/s}$)
* **`[A]` / `[D]`**: Belok Kiri / Kanan (*Steering*)
* **`[Q]` / `[E]`**: Geser menyamping (*Strafe Left/Right*)
* **`[K]` / `[SPACE]`**: Dynamic Soccer Kick (Tendangan kaki kanan berkecepatan tinggi)
* **`[SPACE]`**: Berhenti (*auto-blend* ke posisi berdiri stabil)
* **`[1]` / `[2]` / `[3]`**: Kecepatan preset (Jalan Santai $0.50\text{ m/s}$, Jalan Cepat $0.80\text{ m/s}$, *Sprint* $1.55\text{ m/s}$)
* **`[B]`**: *Respawn* / Acak posisi bola sepak di lapangan
* **`[V]`**: Ganti kamera (*Chase Cam* $\to$ *TV Broadcast Pitch Cam* $\to$ *Goal Net Cam*)
* **`[R]`**: Reset posisi robot dan bola sepak ke titik awal `(0, 0)`

---

### B. Simulasi Pertandingan Sepak Bola Otonom (Autonomous Striker Match)
Simulasikan robot BarelangS bertanding di arena RoboCup berstandar regulasi:
```bash
# Mode Interaktif GUI 3D:
python sim_soccer_striker_ai.py

# Render Video Demonstrasi Otonom (Approach -> Sprint -> Align -> Strike -> Goal -> Celebrate):
python sim_soccer_striker_ai.py --render-demo data/mocap/barelang_soccer_striker_match.mp4
```
Kecepatan luncur bola mencapai **$17.6\text{ km/h}$** menjebol jaring gawang, diikuti selebrasi kemenangan robot mengangkat kedua lengan (*V-Pose Victory Celebration*)!

---

### C. Simulasi Fisika Murni Tendangan Bola Sepak (Newton Physics)
Uji impak tabrakan dan momentum tendangan kaki robot terhadap bola sepak dengan kontak dinamis tertutup:
```bash
python sim_soccer_kick_physics.py
```
* **`[SPACE]` / `[K]`**: Luncurkan tendangan dinamis ($11-14\text{ km/h}$, jarak $>5\text{ meter}$)
* **`[R]`**: Reset robot dan bola secara instan dan *thread-safe*
* **`[C]`**: Toggle sudut pandang kamera (*Cinematic Side-View* vs *Tracking*)

---

### D. Uji Ketahanan Dorongan & Stepping Recovery
Uji ketahanan fisik robot terhadap dorongan/benturan keras:
```bash
python test_push_recovery.py
```
* **`[I]`**: Dorong dari Belakang ($+26\text{ N}$) $\to$ Langkah penyeimbang maju
* **`[K]`**: Dorong dari Depan ($-24\text{ N}$) $\to$ Langkah penyeimbang mundur
* **`[J]` / `[L]`**: Dorong dari Kanan / Kiri ($\pm 22\text{ N}$) $\to$ Melebarkan pijakan kaki
* **`[P]`**: Kejutan impuls acak ($30\text{ N}$)
* **`[SPACE]`**: Toggle jalan $\leftrightarrow$ berdiri diam

---

### C. Visualisasi Motion Capture Manusia Asli
Putar trajektori MoCap manusia yang telah di-retarget ke sendi BarelangS:
```bash
python view_mocap_barelang.py --mode loop
```
* **`[SPACE]`**: Pause / Play
* **`[1]` / `[2]`**: Slow-motion ($0.25\times, 0.5\times$) atau percepat ($1.5\times, 2.0\times$)
* **`[C]`**: Toggle siklus langkah kontinu (*seamless stride loop*)

---

## 2. Struktur Repositori

```
Walking-Gait-Locomotion-/
├── supermimic/                      # Framework SuperMimic (Differentiable World Model)
│   ├── mocap_dataset.py             # Loader trajektori referensi MoCap & horizon slicer
│   ├── world_model.py               # PyTorch Neural World Model (MLP s_t, a_t -> s_{t+1})
│   ├── policy.py                    # Jaringan Actor Tracking Policy
│   ├── collect_dataset.py           # Pengumpul 50k transisi fisika MuJoCo (RSI sampling)
│   ├── train_world_model.py         # Pelatihan Differentiable World Model (MSE < 0.032)
│   ├── train_policy_bptt.py         # Pelatihan BPTT Horizon H=8 (error sendi < 10°)
│   ├── push_recovery_controller.py  # Capture Point & Stepping Recovery controller
│   └── evaluate_policy.py           # Evaluasi closed-loop fisika di MuJoCo
│
├── data/mocap/                      # Dataset & Video Rekaman Hasil
│   ├── barelang_mocap_walk.npz      # Trajektori jalan manusia yang di-retarget ke BarelangS
│   ├── amp_humanoid_walk.npy        # Raw dataset CMU/NVIDIA AMP walking
│   ├── amp_humanoid_run.npy         # Raw dataset CMU running
│   ├── barelang_steering_demo.mp4   # Video demonstrasi kemudi S-curve
│   └── barelang_push_recovery_demo.mp4 # Video demonstrasi tahan dorongan 26N
│
├── models/                          # Model Neural Network Terlatih
│   ├── supermimic_world_model.pt    # World Model SuperMimic terlatih
│   ├── supermimic_policy.pt         # Tracking Policy SuperMimic terlatih
│   └── ppo_barelangS_v2_best.zip    # Model PPO Terrain Navigation terbaik
│
├── meshes/                          # File CAD/3D STL robot BarelangS
├── textures/                        # Tekstur photorealistic procedural terrain
├── barelangS_terrain_mujoco.xml     # Model deskripsi fisik MuJoCo 3D
├── barelang_terrain_env_v2.py       # Gym Environment Terrain & IMU Torso
├── steer_barelang_supermimic.py     # Controller kemudi interaktif real-time
├── test_push_recovery.py            # Simulator uji tabrakan & dorongan
├── view_mocap_barelang.py           # Visualizer gerak MoCap
└── retarget_mocap_barelang.py       # Skrip retargeting kinematik MoCap
```

---

## 3. Spesifikasi Aktuator Robot BarelangS

Robot memiliki 20 sendi aktif:
* **Tungkai Kaki (10 Sendi)**: Dynamixel **MX-106** ($10.0\text{ N}\cdot\text{m}$, $K_p = 35.0$)
  * `hip_roll`, `hip_pitch`, `knee`, `ankle_pitch`, `ankle_roll` (kiri & kanan)
* **Panggul Yaw & Lengan (8 Sendi)**: Dynamixel **MX-64** ($6.0\text{ N}\cdot\text{m}$, $K_p = 20.0$)
  * `hip_yaw`, `shoulder_pitch`, `shoulder_roll`, `elbow` (kiri & kanan)
* **Leher & Kepala (2 Sendi)**: Dynamixel **MX-28** ($2.5\text{ N}\cdot\text{m}$, $K_p = 12.5$)
  * `head_yaw`, `head_pitch`

---

## 4. Persyaratan Sistem & Instalasi

```bash
git clone git@github.com:shidiqeka/Walking-Gait-Locomotion-.git
cd Walking-Gait-Locomotion-

# Pastikan environment Python (>= 3.9) aktif:
pip install numpy scipy mujoco torch opencv-python gymnasium stable-baselines3
```

---

## 5. Lisensi
Dikembangkan untuk Riset Tim Humanoid Soccer Robot **BarelangFC**.
Didistribusikan di bawah lisensi MIT / Academic Research.
