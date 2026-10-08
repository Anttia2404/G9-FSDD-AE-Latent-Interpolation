# G9 — Tăng cường dữ liệu nhận dạng chữ số nói bằng nội suy không gian tiềm ẩn của Autoencoder (FSDD)

**Sinh viên:** Võ Tấn Tài — 23110150  
**Giảng viên hướng dẫn:** Hồ Nhựt Minh  
**Học phần:** Trí tuệ nhân tạo cho IoT — 2026–2027  
**Mã đề tài:** G9 — Tạo sinh (Generative)

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Yêu cầu môi trường](#2-yêu-cầu-môi-trường)
3. [Dữ liệu](#3-dữ-liệu)
4. [Quy trình chạy](#4-quy-trình-chạy)
5. [Cấu trúc mã nguồn](#5-cấu-trúc-mã-nguồn)
6. [Cấu hình thí nghiệm](#6-cấu-hình-thí-nghiệm)
7. [Kết quả thực nghiệm](#7-kết-quả-thực-nghiệm-đã-chạy-ngày-05102026)
8. [Các điểm đã khớp đề cương](#8-các-điểm-đã-khớp-đề-cương)
9. [Giới hạn và phần chưa làm](#9-giới-hạn-và-phần-chưa-làm)
10. [Lưu ý khi báo cáo](#10-lưu-ý-khi-báo-cáo)
11. [Trích dẫn](#11-trích-dẫn)
12. [Liên hệ](#12-liên-hệ)

---

## 1. Tổng quan

Mã nguồn cài đặt đúng theo đề cương `DeCuong_G9_Vo_Tan_Tai_Da_Chinh_Sua.docx`. Nghiên cứu khảo sát hiệu quả của việc **tăng cường dữ liệu** nhận dạng chữ số nói bằng **nội suy không gian tiềm ẩn (latent space interpolation)** của Autoencoder (AE) trên bộ dữ liệu **FSDD** (Free Spoken Digit Dataset).

**Câu hỏi nghiên cứu:**

- Nội suy latent (Linear, SLERP) có cải thiện Accuracy, Macro-F1 so với baseline và SpecAugment không?
- AE chất lượng đến đâu thì nội suy latent có ích?

**Kết quả chính:** Trên 90 runs (5 điều kiện × 6 folds × 3 seeds), không phương pháp tăng cường nào vượt baseline B0. Tuy nhiên, **P1_AE_Linear** là phương pháp tăng cường tốt nhất, vượt B2_AE_Rec ở 5/6 speakers và 14/18 runs (+1,78 pp).

---

## 2. Yêu cầu môi trường

- **Python:** 3.10+
- **GPU:** Khuyến nghị (đã kiểm tra trên Google Colab GPU Tesla T4)
- **RAM:** ≥ 8 GB
- **Thời gian chạy đầy đủ:** ~2 giờ (6 folds × 3 seeds × 5 conditions)

### Phiên bản thư viện (đã kiểm tra)

Xem `requirements.txt` để biết phiên bản cụ thể. Cài đặt:

```bash
pip install -r requirements.txt
```

Nếu không có `requirements.txt` (ví dụ khi clone mới), cài thủ công:

```bash
pip install "torch>=2.0" librosa==0.11.0 soundfile numpy pandas scikit-learn matplotlib scipy tqdm
```

---

## 3. Dữ liệu

Bộ dữ liệu FSDD — Free Spoken Digit Dataset:

- 3.000 bản ghi WAV, 8 kHz, mono
- 6 người nói: george, jackson, lucas, nicolas, theo, yweweler
- Mỗi người đọc mỗi chữ số 0–9 năm mươi lần
- Tên file: `digit_speaker_index.wav`

### Tải dữ liệu

```bash
git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git

# Khóa phiên bản (đề cương yêu cầu):
git -C free-spoken-digit-dataset rev-parse HEAD
# Output mong đợi: 26eb9aaf76e81b692f806f9140c2d2777410d7a1
```

---

## 4. Quy trình chạy

### Bước 1: Chuẩn bị

```bash
# Cài đặt môi trường
pip install -r requirements.txt

# Tải FSDD (nếu chưa có)
git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git
```

### Bước 2: Tiền xử lý (`prep`)

```bash
python g9_pipeline.py prep --data free-spoken-digit-dataset/recordings --out out
```

Output mong đợi:

```text
3000 ban ghi | 6 nguoi noi: ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler']
thoi luong: min 0.143s, max 2.283s | bi cat: 0.7% | bo loc mel rong (rong): 0/64
phan bo lop: [300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
ban ghi rong: 0 | trung sha256: 0
```

Kết quả tạo ra:

- `out/manifest.csv` — Metadata 3.000 bản ghi
- `out/features_raw.npy` — Phổ log-mel, shape `(3000, 64, 63)`

### Bước 3: Chạy thí nghiệm (`run`)

Chạy nhanh để kiểm tra môi trường (~1 phút):

```bash
python g9_pipeline.py run --out out --quick
```

Chạy đầy đủ — TN1 + TN2 (6 folds × 3 seeds, ~2 giờ):

```bash
python g9_pipeline.py run --out out
```

Chạy rút gọn (6 folds × 1 seed, ~40 phút):

```bash
python g9_pipeline.py run --out out --seeds 0
```

> **Lưu ý quan trọng:** Script tự động bỏ qua cặp (fold, seed) đã có trong `results.csv`. Nếu Colab bị ngắt, chỉ cần chạy lại cùng lệnh — script sẽ tiếp tục từ chỗ dừng, không chạy lại từ đầu.

### Bước 4: Tổng hợp kết quả (`aggregate`)

```bash
python g9_pipeline.py aggregate --out out
```

Output mong đợi:

- Bảng 1: Mean ± SD giữa 6 người nói
- Bảng 2: Accuracy theo từng người nói
- Bảng 3: Chênh lệch ghép cặp vs B0/B1/B2

Lưu thành: `summary_methods.csv`, `summary_per_speaker.csv`, `summary_paired.csv`.

### Thí nghiệm mở rộng (tùy chọn)

```bash
# TN3: Số mẫu tăng cường M=500
python g9_pipeline.py run --out out_M500 --M 500

# TN5: d_z=32
python g9_pipeline.py run --out out_dz32 --dz 32

# TN5: λ=0
python g9_pipeline.py run --out out_lam0 --lam 0
```

---

## 5. Cấu trúc mã nguồn

```text
.
├── g9_pipeline.py              # Mã nguồn chính
├── requirements.txt            # Phiên bản thư viện
├── README.md                   # File này
└── out/                        # Thư mục output (tạo bởi lệnh)
    ├── manifest.csv            # Metadata 3.000 bản ghi
    ├── splits.csv              # Bảng LOSO 6 folds
    ├── features_raw.npy        # Phổ log-mel 3000×64×63
    ├── norm_fold1.npz ... norm_fold6.npz   # μ, σ chuẩn hóa theo fold
    ├── pairs/                  # Danh sách cặp sinh (truy vết)
    │   └── pairs_f*_s*.csv
    ├── ckpt/                   # Checkpoint AE + Classifier
    │   ├── ae_f*_s*.pt
    │   └── clf_<method>_f*_s*.pt
    ├── results.csv             # Kết quả 90 runs (16 cột)
    ├── predictions.csv         # Dự đoán từng mẫu
    ├── summary_methods.csv     # Bảng 1: mean ± SD
    ├── summary_per_speaker.csv # Bảng 2: theo người nói
    ├── summary_paired.csv      # Bảng 3: chênh lệch ghép cặp
    ├── fig_interp_fold1.png    # Hình: phổ gốc/tái tạo/nội suy
    ├── fig_boxplot.png         # Hình: boxplot Accuracy
    ├── fig_bar.png             # Hình: bar plot mean ± SD
    ├── fig_per_speaker.png     # Hình: Accuracy theo người nói
    ├── fig_cm_P1.png           # Hình: confusion matrix P1
    └── fig_cm_B0.png           # Hình: confusion matrix B0
```

### CLI

```text
python g9_pipeline.py <command> [options]

Commands:
  prep        Tiền xử lý FSDD → manifest.csv, features_raw.npy
  run         Huấn luyện AE + Classifier, sinh dữ liệu tăng cường
  aggregate   Tổng hợp mean ± SD, delta ghép cặp

Options cho "run":
  --out PATH        Thư mục output (mặc định: out)
  --folds [1 2 3]   Chỉ chạy một số folds (mặc định: tất cả)
  --seeds [0 1 2]   Chỉ chạy một số seeds (mặc định: 0, 1, 2)
  --M INT           Số mẫu sinh/fold (mặc định: 1000)
  --dz INT          Chiều không gian tiềm ẩn (mặc định: 64)
  --lam FLOAT       Hệ số λ cho L1 loss (mặc định: 0.1)
  --ae-epochs INT   Số epoch tối đa của AE (mặc định: 200)
  --clf-steps INT   Số bước tối đa của classifier (mặc định: 6000)
  --quick           Chạy nhanh để kiểm tra (3 epochs, 300 steps)
```

---

## 6. Cấu hình thí nghiệm

### 6.1. Tiền xử lý (mục 2.2 đề cương)

| Bước                 | Cấu hình                                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------- | ------- | --- |
| Âm thanh đầu vào     | WAV float32, 8.000 Hz, mono                                                                          |
| DC và pre-emphasis   | Trừ trung bình; a = 0,97: `v[n] = u[n] − 0,97·u[n−1]`                                                |
| Chuẩn hóa thời lượng | Zero-pad cuối nếu < 8.000 mẫu; nếu dài hơn, chọn đoạn 8.000 mẫu có tổng bình phương biên độ lớn nhất |
| STFT                 | Hann, `n_fft = win_length = 256`, `hop_length = 128`, `center=True`, `pad_mode='constant'`           |
| Phổ mel              | 64 bộ lọc; f_min = 50 Hz, f_max = 4.000 Hz; `power = 2`; `htk = False`; `norm = 'slaney'`            |
| Log-power            | `S = ln(1 + P)` với `P = B_mel ·                                                                     | STFT(v) | ²`  |
| Chuẩn hóa thống kê   | μ_f, σ_f trên Train của fold (63 khung hợp lệ)                                                       |
| Tensor cho mạng      | 64 × 63 + 1 cột đệm 0 → 1 × 64 × 64                                                                  |

### 6.2. Chia dữ liệu LOSO (mục 2.3)

| Fold | Test     | Val      | Train                            |
| ---- | -------- | -------- | -------------------------------- |
| 1    | george   | jackson  | lucas, nicolas, theo, yweweler   |
| 2    | jackson  | lucas    | george, nicolas, theo, yweweler  |
| 3    | lucas    | nicolas  | george, jackson, theo, yweweler  |
| 4    | nicolas  | theo     | george, jackson, lucas, yweweler |
| 5    | theo     | yweweler | george, jackson, lucas, nicolas  |
| 6    | yweweler | george   | jackson, lucas, nicolas, theo    |

### 6.3. Kiến trúc mô hình

**Autoencoder (276.001 tham số):**

- Encoder: Conv2d 1→8→16→32 (stride 2, kernel 3, ReLU)
- Bottleneck: Flatten 2048 → Linear 64
- Decoder: Linear 64 → 2048 → ConvTranspose2d 32→16→8→1

**Classifier CNN (64.266 tham số):**

- Conv2d 1→16→32→64 (stride 2, kernel 3, ReLU)
- Flatten 4096 → Dropout 0.2 → Linear 10

### 6.4. Huấn luyện

| Thành phần       | AE           | Classifier   |
| ---------------- | ------------ | ------------ |
| Optimizer        | AdamW        | AdamW        |
| Learning rate    | 1e-3         | 1e-3         |
| β                | (0.9, 0.999) | (0.9, 0.999) |
| Weight decay     | 1e-4         | 1e-4         |
| Batch size       | 32           | 32           |
| Seed             | {0, 1, 2}    | {0, 1, 2}    |
| Cosine schedule  | Theo epoch   | Theo step    |
| Max epochs/steps | 200          | 6000         |
| Early stopping   | patience 20  | patience 20  |

### 6.5. Điều kiện so sánh (mục 3.6)

| Mã           | Dữ liệu huấn luyện classifier               | Vai trò               |
| ------------ | ------------------------------------------- | --------------------- |
| B0_Real      | 2.000 mẫu Train thật; không tăng cường      | Đối chứng TRTR        |
| B1_SpecMask  | Train thật + 1.000 mẫu che thời gian/tần số | So sánh với masking   |
| B2_AE_Rec    | Train thật + 1.000 phổ D(E(X))              | Tách tác dụng tái tạo |
| P1_AE_Linear | Train thật + 1.000 phổ nội suy tuyến tính   | Phương pháp chính     |
| P2_AE_SLERP  | Train thật + 1.000 phổ SLERP                | So sánh nội suy       |

---

## 7. Kết quả thực nghiệm (đã chạy ngày 05/10/2026)

### 7.1. Kết quả chính (mean ± SD giữa 6 người nói)

| Method       | Accuracy (%)  | Macro-F1 (%)  | ΔAcc vs B0 (pp) |
| ------------ | ------------- | ------------- | --------------- |
| B0_Real      | 44,69 ± 14,78 | 41,40 ± 15,33 | —               |
| B1_SpecMask  | 41,83 ± 14,48 | 38,84 ± 13,73 | −2,86           |
| B2_AE_Rec    | 40,97 ± 13,22 | 38,02 ± 12,69 | −3,72           |
| P1_AE_Linear | 42,74 ± 13,52 | 39,93 ± 13,23 | −1,94           |
| P2_AE_SLERP  | 41,97 ± 12,10 | 38,53 ± 12,10 | −2,72           |

### 7.2. Accuracy theo người nói (%)

| Speaker  | B0    | B1    | B2    | P1    | P2    |
| -------- | ----- | ----- | ----- | ----- | ----- |
| george   | 51,20 | 47,27 | 45,53 | 44,60 | 43,13 |
| jackson  | 60,13 | 55,13 | 53,80 | 55,20 | 52,67 |
| lucas    | 31,27 | 32,07 | 28,40 | 29,33 | 30,87 |
| nicolas  | 57,80 | 52,20 | 53,07 | 58,13 | 56,33 |
| theo     | 23,27 | 17,13 | 21,53 | 24,47 | 25,13 |
| yweweler | 44,47 | 47,20 | 43,47 | 44,73 | 43,67 |

### 7.3. Chênh lệch ghép cặp

| So sánh | ΔAcc (pp) | ΔAcc SD | Speakers improved | Runs improved |
| ------- | --------- | ------- | ----------------- | ------------- |
| P1 − B0 | −1,94     | 3,18    | 3/6               | 7/18          |
| P1 − B1 | +0,91     | 4,58    | 3/6               | 10/18         |
| P1 − B2 | +1,78     | 2,03    | 5/6               | 14/18         |
| P2 − B2 | +1,00     | 2,48    | 4/6               | 11/18         |

**Điểm đáng chú ý:** P1 vượt B2 ở 5/6 speakers và 14/18 runs (+1,78 pp, SD nhỏ 2,03) → tín hiệu tích cực của phương pháp nội suy.

### 7.4. Thông tin chạy

- Tổng runs: 90 (5 conditions × 6 folds × 3 seeds) + 18 AE
- Tổng thời gian: ~2 giờ trên Google Colab GPU Tesla T4
- SLERP fallbacks: 0/1.000 (không có vấn đề số học)
- Ngày chạy: 05/10/2026
- Commit FSDD: `26eb9aaf76e81b692f806f9140c2d2777410d7a1`

---

## 8. Các điểm đã khớp đề cương

| Mục | Nội dung                                                                                                         | Cách kiểm tra                          |
| --- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 2.2 | DC → pre-emphasis 0,97 → đệm/cắt 8.000 mẫu → STFT 256/128 → mel 64 → ln(1+P) → z-score theo Train → thêm 1 cột 0 | `prep` in thống kê; tensor 1×64×64     |
| 2.3 | LOSO 6 fold, Val = người kế tiếp theo vòng tròn                                                                  | `splits.csv` trùng bảng trong đề cương |
| 3.2 | AE 276.001 tham số                                                                                               | cột `ae_params` trong `results.csv`    |
| 3.3 | Loss = MSE trung bình + 0,1 × L1 trung bình, chỉ trên 63 khung                                                   | hàm `rec_loss`                         |
| 3.4 | Linear/SLERP dùng chung danh sách cặp, α ∈ {0,25; 0,5; 0,75}, fallback khi sin θ ≈ 0                             | `pairs/*.csv`, cột `slerp_fallbacks`   |
| 3.5 | CNN 64.266 tham số, khởi tạo giống nhau giữa các điều kiện theo seed                                             | cột `clf_params`                       |
| 3.6 | B0, B1 (mask thời gian + tần số), B2 (tái tạo), P1, P2                                                           | cột `method`                           |
| 4.5 | Trung bình 3 seed theo người nói, SD mẫu số 5 giữa 6 người nói                                                   | `aggregate`                            |

---

## 9. Giới hạn và phần chưa làm

Các phần sau thuộc mức ưu tiên 2 / mở rộng trong đề cương và chưa được cài đặt trong mã này:

- **TN4** — đánh giá nhiễu Gaussian 20/10/5 dB trên checkpoint đã khóa (mục 4.6)
- **TSTR** — đánh giá classifier huấn luyện chỉ trên dữ liệu sinh (mục 4.2)
- **Kiểm tra sao chép phổ** — khoảng cách phổ giữa mẫu sinh và Train thật (mục 4.3)
- **Đo độ trễ và bộ nhớ trên thiết bị** (mục 4.7) — hiện chỉ đo trên GPU T4
- **Kiểm định thống kê (Wilcoxon)** — mean ± SD không tự chứng minh ý nghĩa thống kê

---

## 10. Lưu ý khi báo cáo

- **Phiên bản thư viện:** Ghi rõ phiên bản thực tế (`pip freeze`). Mã đã kiểm tra với librosa 0.11.0 (khớp với đề cương).
- Kết quả `--quick` (3 epoch AE, 300 bước) chỉ để kiểm tra mã chạy được, **KHÔNG** đưa vào báo cáo.
- Nếu P1/P2 không vượt B0/B1/B2, vẫn báo cáo đúng như vậy (đề cương mục 5.3). Đây là kết quả hợp lệ của nghiên cứu.
- Confusion matrix đã được vẽ trong `fig_cm_P1.png` và `fig_cm_B0.png`.
- Không sửa số liệu để đạt kết quả đẹp. Trung thực với kết quả thực nghiệm.

---

## 11. Trích dẫn

Nếu sử dụng mã này, vui lòng trích dẫn:

```text
Võ Tấn Tài. (2026). Tăng cường dữ liệu nhận dạng chữ số nói bằng nội suy không gian
tiềm ẩn của Autoencoder trên bộ dữ liệu FSDD. Báo cáo học phần Trí tuệ nhân tạo cho IoT,
Trường Đại học Sư phạm Kỹ thuật TP.HCM.
```

---

## 12. Liên hệ

- **Sinh viên:** Võ Tấn Tài
- **MSSV:** 23110150
- **Email:** 23110150@student.hcmute.edu.vn
- **Ngày hoàn thành:** 05/10/2026
