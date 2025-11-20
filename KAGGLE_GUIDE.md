# 📘 Kaggle Notebook 학습 가이드

## 🎯 목표

7개 CausalMamba 디노이저 모델을 Kaggle GPU 환경에서 학습

---

## 📋 준비물

### 1. Kaggle 계정
- 무료 GPU 사용 가능 (주 30시간)
- T4 또는 P100 권장

### 2. 데이터셋 업로드

**Option A: Kaggle Dataset으로 업로드**
```
1. Kaggle → Your Work → Datasets → New Dataset
2. 업로드:
   - FinancialDenoising/ (전체 폴더)
   - train_only.csv (TRMwithQuant에서)
3. Title: "financial-denoising-causal-mamba"
```

**Option B: 직접 업로드**
```
Kaggle Notebook에서:
- Add Data → Upload → Select Files
```

---

## 🚀 실행 방법

### Step 1: Kaggle Notebook 생성

1. Kaggle → Code → New Notebook
2. Settings:
   - **Accelerator**: GPU T4 (또는 P100)
   - **Internet**: On (pip install 필요 시)
   - **Persistence**: Files only

### Step 2: 코드 셀 실행

**셀 1: 환경 설정**
```python
# 프로젝트 복사 (데이터셋으로 업로드한 경우)
!cp -r /kaggle/input/financial-denoising-causal-mamba/FinancialDenoising /kaggle/working/
%cd /kaggle/working/FinancialDenoising

# 또는 직접 업로드한 경우
# !ls /kaggle/working/  # 확인
```

**셀 2: 필요 라이브러리 설치 (uv 없으므로)**
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas scikit-learn tqdm matplotlib
```

**셀 3: GPU 확인**
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**셀 4: 학습 시작**
```python
# 방법 1: 준비된 스크립트 사용
!python kaggle_train_all_clusters.py

# 방법 2: 수동으로 하나씩
# for i in range(7):
#     !python training/train_denoiser.py \
#         --cluster_id {i} \
#         --data_path /kaggle/input/your-data/train_only.csv \
#         --epochs 100 \
#         --device cuda
```

### Step 3: 모델 다운로드

```python
# 학습 완료 후 압축
!zip -r trained_models.zip trained_models/

# Kaggle Notebook 화면 우측:
# Data → Output → trained_models.zip 다운로드
```

---

## ⏱️ 예상 소요 시간

| GPU | 1 Cluster | 7 Clusters | 여유 시간 |
|-----|-----------|------------|----------|
| T4 | ~50분 | ~6시간 | 3시간 OK |
| P100 | ~30분 | ~4시간 | 5시간 OK |

**Kaggle 제한**: 9시간/세션 → **충분함** ✅

---

## 🎯 학습 중 모니터링

### 확인할 지표

```python
# 각 클러스터 학습 시 출력:
Epoch 1/100: loss=0.1234
Epoch 10/100: loss=0.0856  # 감소 확인
...
Epoch 100/100: loss=0.0234
✓ Saved checkpoint to trained_models/cluster_0_best.pt
```

**정상 패턴**:
- Loss 초기: 0.1-0.2
- Loss 중반: 0.05-0.08
- Loss 최종: 0.02-0.04

**비정상 패턴**:
- Loss가 안 떨어짐 (0.1 고정) → 학습 안됨
- Loss가 NaN → 버그
- Loss가 발산 (증가) → learning rate 문제

---

## 🐛 문제 해결

### 1. GPU 메모리 부족
```
RuntimeError: CUDA out of memory
```
**해결**: `train_denoiser.py` 수정
```python
# Line 239 근처
batch_size = 16  # 기본 32 → 16으로 감소
```

### 2. 데이터 경로 오류
```
FileNotFoundError: train_only.csv
```
**해결**: `kaggle_train_all_clusters.py` 수정
```python
# Line 18
DATA_PATH = "/kaggle/input/실제경로/train_only.csv"  # 수정!
```

확인:
```python
!ls /kaggle/input/  # 업로드된 데이터셋 목록
!ls /kaggle/input/your-dataset-name/  # 파일 확인
```

### 3. Import 오류
```
ModuleNotFoundError: No module named 'models'
```
**해결**:
```python
import sys
sys.path.append('/kaggle/working/FinancialDenoising')
```

---

## 📥 학습 완료 후 로컬 적용

### 1. 모델 다운로드
```
Kaggle → Output → trained_models.zip
```

### 2. 로컬에 배치
```bash
# Windows
cd C:\Users\jrjin\Desktop\FinancialDenoising
unzip trained_models.zip
mv trained_models_old_bimamba trained_models_backup  # 백업
```

### 3. 추론 실행
```bash
uv run python inference/denoise_causal.py \
    --input_csv ../TRMwithQuant/TinyRecursiveModels/CSVs/val_only.csv \
    --output_csv val_denoised_causal_v2.csv \
    --device cpu  # 로컬은 CPU
```

### 4. 검증
```bash
uv run python Common/evaluation/validate_trading_signals.py \
    --train_original ../TRMwithQuant/TinyRecursiveModels/CSVs/train_only.csv \
    --train_denoised train_denoised_causal_v2.csv \
    --val_original ../TRMwithQuant/TinyRecursiveModels/CSVs/val_only.csv \
    --val_denoised val_denoised_causal_v2.csv
```

---

## ✅ 체크리스트

**학습 전**:
- [ ] Kaggle GPU 활성화 확인
- [ ] 데이터셋 경로 확인
- [ ] 충분한 시간 여유 (6-9시간)

**학습 중**:
- [ ] Loss 감소 확인
- [ ] GPU 메모리 사용량 모니터링
- [ ] 체크포인트 저장 확인

**학습 후**:
- [ ] 7개 클러스터 모두 완료
- [ ] trained_models.zip 다운로드
- [ ] 로컬 추론 테스트

---

## 📞 도움말

**Kaggle 공식 문서**:
- GPU 사용: https://www.kaggle.com/docs/notebooks#gpu
- 데이터셋: https://www.kaggle.com/docs/datasets

**프로젝트 이슈**:
- GitHub (있다면) 또는 개발자에게 문의
