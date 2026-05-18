# ShotGuide: Short-form Shot Classification

숏폼 영상 링크를 입력하면 화면 전환 단위로 장면을 나누고, 각 장면의 shot type과 화면 텍스트 여부를 예측한 뒤 원본 영상 위에 분석 결과를 오버레이하는 프로젝트입니다.

## 프로젝트 목표

틱톡, 인스타그램 릴스, 유튜브 쇼츠처럼 세로형 숏폼 콘텐츠는 빠른 컷 전환, 자막/텍스트 오버레이, 제품/얼굴 클로즈업 등 기존 영화/드라마 기반 shot classification과 다른 패턴을 가집니다.

이 프로젝트는 숏폼 영상을 입력받아 다음 결과를 만드는 것을 목표로 합니다.

- 장면 전환 단위 scene 분할
- scene별 대표 프레임 추출
- `close-up`, `medium`, `wide`, `object`, `space` shot type 분류
- 화면 텍스트 포함 여부(`has_text`) 예측
- shot type과 confidence를 원본 영상 위에 표시한 overlay mp4 생성

## 현재까지 진행상황

### 1. 데이터셋 구축

라벨링된 컷 이미지 데이터셋을 구성했습니다.

- 전체 라벨 이미지 수: `1188장`
- 영상 수: `129개`
- split 기준: video-level split
- train: `721장`
- val: `208장`
- test: `259장`

shot type 분포는 다음과 같습니다.

| Shot type | Count |
|---|---:|
| medium | 405 |
| object | 296 |
| wide | 216 |
| close-up | 147 |
| space | 124 |

텍스트 포함 여부 분포는 다음과 같습니다.

| has_text | Count |
|---|---:|
| text | 945 |
| no text | 243 |

### 2. ResNet18 Baseline 실험

먼저 ResNet18 pretrained backbone을 사용한 multi-task baseline을 학습했습니다.

예측 대상은 두 가지입니다.

- `shot_type`
- `has_text`

관련 결과는 `outputs_baseline/`에 남아 있습니다.

주요 결과:

| Model | Shot Acc | Shot Macro F1 | Text Acc | Text F1 | Joint Acc |
|---|---:|---:|---:|---:|---:|
| ResNet18 frozen backbone baseline | 0.710 | 0.707 | 0.660 | 0.762 | 0.498 |

### 3. CLIP Embedding 기반 모델 학습

ResNet baseline 이후, CLIP ViT-B/32를 frozen backbone으로 사용하고 이미지별 512차원 CLIP embedding을 추출했습니다. 그 위에 작은 multi-task classification head를 학습했습니다.

학습 결과 CLIP 기반 모델이 ResNet baseline보다 더 좋은 성능을 보였습니다.

| Model | Shot Acc | Shot Macro F1 | Text Acc | Text F1 | Joint Acc |
|---|---:|---:|---:|---:|---:|
| ResNet18 frozen backbone baseline | 0.710 | 0.707 | 0.660 | 0.762 | 0.498 |
| CLIP ViT-B/32 frozen embeddings | 0.803 | 0.776 | 0.865 | 0.917 | 0.703 |

따라서 현재 최종 추론 파이프라인은 ResNet이 아니라 `CLIP ViT-B/32 + multi-task head` 구조를 사용합니다.

### 4. 인스타그램 링크 기반 Overlay 파이프라인 구현

새 인스타그램 Reels 링크를 넣으면 다음 흐름이 실행되도록 구성했습니다.

```text
Instagram Reels URL
↓
yt-dlp로 mp4 다운로드
↓
CLIP 인접 임베딩 거리 기반 scene detection
↓
scene별 대표 프레임 추출
↓
CLIP embedding + 학습된 head로 shot_type / has_text 예측
↓
원본 영상 위에 예측 결과 overlay
↓
*_overlay.mp4, scene_predictions.csv, scene_metadata.csv 저장
```

## 현재 파일 구조와 역할

### 실행 및 추론 코드

#### `shotguide_instagram_overlay_pipeline.py`

새 인스타그램 링크를 처리하는 메인 실행 스크립트입니다.

`INSTAGRAM_LINKS` 리스트에 링크를 넣고 실행하면 다운로드부터 overlay 생성까지 한 번에 수행합니다.

```powershell
python shotguide_instagram_overlay_pipeline.py
```

#### `shotguide_instagram_overlay_pipeline.ipynb`

위 `.py` 파일의 노트북 버전입니다.

발표, 시연, 단계별 확인에 사용하기 좋습니다. 링크를 바꿀 때는 노트북 상단의 `INSTAGRAM_LINKS` 리스트를 수정하면 됩니다.

#### `shotguide_batch_video_inference.py`

공통 추론 함수 파일입니다.

주요 역할:

- CLIP ViT-B/32 backbone 로드
- `clip_vit_b32_multitask_head.pt` 로드
- 영상 정보 읽기
- scene별 대표 프레임 추출
- scene embedding 생성
- `shot_type`, `has_text` 예측
- guide text 생성
- overlay mp4 렌더링

`shotguide_instagram_overlay_pipeline.py`와 `shotguide_instagram_overlay_pipeline.ipynb`가 이 파일을 `base`로 import해서 사용합니다.

### 학습 코드

#### `shotguide_clip_embedding_baseline.ipynb`

CLIP 기반 모델을 학습한 노트북입니다.

주요 역할:

- `dataset_index_with_splits.csv` 로드
- CLIP ViT-B/32 frozen backbone 로드
- 라벨 이미지별 512차원 CLIP embedding 추출
- embedding cache 저장
- `shot_type`, `has_text`를 동시에 예측하는 multi-task head 학습
- 최종 checkpoint `clip_vit_b32_multitask_head.pt` 저장
- ResNet baseline과 CLIP 성능 비교

### 데이터 및 결과 폴더

#### `labeled_dataset/`

직접 라벨링한 컷 이미지 데이터셋입니다.

모델 재학습에 필요합니다. 현재 `.gitignore`에 포함되어 있어 Git에는 올라가지 않습니다.

#### `outputs_baseline/`

ResNet18 baseline 실험 결과와 데이터 split 정보가 들어 있습니다.

중요 파일:

- `dataset_index_with_splits.csv`: 학습/검증/테스트 split 정보
- `training_history.csv`: ResNet baseline 학습 로그
- `test_predictions.csv`: ResNet baseline 테스트 예측 결과
- `best_multitask_resnet18.pt`, `best_multitask_resnet18_final.pt`: ResNet baseline checkpoint

#### `outputs_clip_embeddings/`

CLIP 기반 모델 학습 결과 폴더입니다.

중요 파일:

- `clip_vit_b32_multitask_head.pt`: 현재 최종 추론 파이프라인에서 사용하는 학습된 head
- `clip_vit_b32_openai_embeddings.npz`: 라벨 이미지들의 CLIP embedding cache
- `clip_embedding_metadata.csv`: embedding과 원본 이미지 메타데이터 연결 정보
- `clip_test_metrics.json`: CLIP 모델 테스트 성능
- `clip_test_predictions.csv`: CLIP 모델 테스트 예측 결과
- `resnet_vs_clip_comparison.csv`: ResNet baseline과 CLIP 비교 결과

#### `videos_new_links/`

인스타그램 링크에서 다운로드된 원본 mp4가 저장되는 폴더입니다.

#### `outputs_instagram_overlay/`

새 링크 추론 결과가 저장되는 폴더입니다.

영상별로 다음 결과가 생성됩니다.

- `*_overlay.mp4`
- `scene_metadata.csv`
- `scene_predictions.csv`
- `scene_clip_embeddings.npz`
- `clip_distance_profile.csv`
- `scene_frames/`

## 실행 방법

### 1. 새 인스타그램 링크 분석

`shotguide_instagram_overlay_pipeline.py` 또는 `shotguide_instagram_overlay_pipeline.ipynb`에서 아래 리스트를 수정합니다.

```python
INSTAGRAM_LINKS = [
    "https://www.instagram.com/reels/SHORTCODE/",
]
```

스크립트로 실행할 경우:

```powershell
python shotguide_instagram_overlay_pipeline.py
```

결과는 `outputs_instagram_overlay/`에 저장됩니다.

### 2. CLIP head 재학습

`shotguide_clip_embedding_baseline.ipynb`를 실행합니다.

필요한 입력:

- `labeled_dataset/`
- `outputs_baseline/dataset_index_with_splits.csv`

생성되는 주요 결과:

- `outputs_clip_embeddings/clip_vit_b32_multitask_head.pt`
- `outputs_clip_embeddings/clip_test_metrics.json`
- `outputs_clip_embeddings/clip_test_predictions.csv`

## 현재 확인된 한계

### close-up과 medium 혼동

현재 테스트 결과에서 `close-up`을 `medium`으로 잘못 분류하는 경우가 있습니다.

CLIP 테스트셋 기준:

```text
실제 close-up 30장 중
- close-up으로 맞춘 것: 16장
- medium으로 틀린 것: 10장
- object로 틀린 것: 4장

실제 medium 85장 중
- medium으로 맞춘 것: 75장
- close-up으로 틀린 것: 3장
```

즉, 주된 문제는 `medium`을 못 맞추는 것이 아니라 `close-up`을 `medium`으로 밀어내는 방향입니다.

가능한 원인:

- train 데이터에서 `close-up`이 `medium`보다 적음
  - close-up train: `100장`
  - medium train: `234장`
- 숏폼에서는 얼굴/상반신/제품 컷의 경계가 모호함
- CLIP embedding만으로 피사체 화면 점유율을 직접 반영하기 어려움

## 앞으로 진행할 사항

### 1. close-up / medium 라벨 기준 명확화

라벨링 기준을 더 구체적으로 정의해야 합니다.

예시:

```text
close-up:
핵심 피사체가 화면 높이 또는 너비의 약 50% 이상을 차지하거나,
얼굴/제품/음식/손 등 디테일 정보가 주된 컷

medium:
피사체가 분명히 보이지만 상반신, 주변 맥락, 배경 정보도 함께 중요한 컷
```

### 2. close-up hard sample 보강

단순히 close-up 수를 늘리는 것보다, medium과 헷갈릴 만한 close-up을 집중적으로 추가하는 것이 중요합니다.

우선 보강할 예시:

- 얼굴이 화면을 크게 차지하지만 어깨/상반신이 일부 보이는 장면
- 음식/제품이 화면 중심에 크지만 배경도 조금 보이는 장면
- 손, 화장품, 액세서리 등 디테일 중심 장면
- 모델이 기존에 `medium`으로 틀린 close-up과 비슷한 장면

목표는 close-up train 데이터를 현재 `100장`에서 `180~220장` 수준으로 늘리는 것입니다.

### 3. 라벨 재검수 후 CLIP head 재학습

close-up/medium 기준을 정리한 뒤 기존 라벨 일부를 재검수하고, `shotguide_clip_embedding_baseline.ipynb`를 다시 실행해 head를 재학습합니다.

재학습 후 확인할 지표:

- close-up recall
- close-up → medium 오분류 수
- shot macro F1
- joint accuracy

### 4. CV feature 추가 검토

close-up과 medium의 차이는 의미 정보보다 피사체 크기와 화면 점유율의 영향이 큽니다.

추가해볼 수 있는 feature:

- 얼굴 bounding box의 화면 점유율
- 사람 bounding box의 화면 점유율
- 중앙 피사체 크기
- OCR/text 영역 비율

향후 구조 예시:

```text
CLIP 512-dim embedding
+
face/person/text area ratio features
↓
multi-task classification head
```

### 5. 결과물 정리

최종 제출/발표용으로 다음 자료를 정리할 예정입니다.

- 최종 overlay mp4 예시
- scene별 예측 CSV 예시
- ResNet vs CLIP 비교 표
- confusion matrix
- close-up/medium 개선 전후 비교

