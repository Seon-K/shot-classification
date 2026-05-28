# ShotGuide 모델 설계 기준

## 목적

- 인스타그램 릴스 또는 숏폼 영상의 컷 전환 탐지
- 컷별 `shot_type` 분류
- 컷별 `has_text` 여부 판정
- 예측 결과를 confidence와 함께 overlay guide로 변환

## 라벨 체계

```text
shot_type = close-up, medium, wide, object, space
has_text = 0, 1
```

처음부터 결합 라벨 대신 multi-task 구조를 사용했습니다.

```text
Input image
↓
Image encoder
↓
Shared representation
├─ shot_head → shot_type
└─ text_head → has_text
```

이유:

- `shot_type`과 `text`는 서로 다른 시각적 단서를 사용함
- 결합 라벨을 만들면 class 수가 증가하고 데이터 불균형이 커짐
- text 존재 여부가 shot-type 판단을 과도하게 방해하는 것을 줄일 수 있음

## 핵심 설계 가설

```text
shot-type classification과 text presence detection은 서로 다른 visual representation을 요구한다.
```

Shot-type에는 피사체 크기, 배경 비중, spatial layout, object boundary가 중요합니다. Text presence에는 자막, overlay text, UI-like pattern, high-frequency visual cue가 중요합니다.

## 모델별 역할

| 모델 | 역할 | 적용 이유 |
|---|---|---|
| CLIP ViT-B/32 | semantic context | 이미지와 텍스트를 함께 학습한 vision-language baseline |
| DINOv2 | structural visual feature | 구도, 피사체 배치, object boundary 보완 |
| DINOv3 | 최신 DINO 대체 후보 | DINOv2보다 최신 self-supervised backbone 검증 |
| SigLIP2 | text-sensitive vision-language feature | text presence에서 높은 성능 |
| C-RADIOv4 | distilled single backbone | 여러 teacher representation을 하나로 압축한 대안 |

## 주요 구조

### CLIP Stronger Head

```text
CLIP ViT-B/32 frozen image encoder
↓
512-dim embedding
↓
shared MLP: 512 → 512 → 256
├─ shot_head
└─ text_head
```

장점은 단순성과 pipeline 적용 용이성입니다. 한계는 shot-type에서 spatial composition을 충분히 반영하지 못한다는 점입니다.

### CLIP + DINOv2 Shot Ensemble

```text
CLIP shot probability
DINOv2 shot probability
↓
weighted ensemble
```

CLIP은 semantic cue에 강하고 DINOv2는 visual structure에 강하기 때문에, 구도 판단이 중요한 shot-type task에서 상호보완적입니다.

최적 설정:

```text
alpha_clip = 0.46
loss = focal loss
```

### SigLIP2 Text Branch

SigLIP2는 CLIP 계열의 최신 vision-language encoder 중 하나입니다. OCR처럼 글자를 읽는 모델은 아니지만, text/overlay 존재 여부를 감지하는 representation이 강하게 나타났습니다.

```text
SigLIP2 single text accuracy: 0.9455
SigLIP2 single text macro F1: 0.8995
```

### CLIP+DINOv2 Shot + SigLIP2 Text

현재 최고 성능 구조입니다.

```text
shot: CLIP + DINOv2 focal ensemble
text: SigLIP2 text head
```

성능:

```text
shot_accuracy: 0.8618
text_accuracy: 0.9527
joint_accuracy: 0.8218
```

단점은 encoder 3개를 사용한다는 점입니다.

### C-RADIOv4-SO400M

C-RADIOv4는 DINOv3, SigLIP2, SAM 계열 teacher의 representation을 하나의 backbone으로 distillation한 모델입니다.

실험 목적:

```text
3-model 구조를 단일 backbone으로 대체할 수 있는지 확인
```

결과:

```text
joint_accuracy: 0.8000
text_accuracy: 0.9745
shot_accuracy: 0.8255
```

3-model 최고 성능보다는 낮지만 단일 backbone이라는 점에서 성능-복잡도 trade-off가 좋습니다.

## 라벨 Ambiguity 대응

shot-type 라벨은 객관적인 객체 분류가 아니라 구도 판단에 가깝습니다. 특히 `medium`, `wide`, `close-up`, `object` 사이의 경계가 애매합니다.

이 문제를 다음 방식으로 다뤘습니다.

1. macro F1 / balanced accuracy 사용
2. DINOv2 structural representation 추가
3. focal loss 적용
4. top-2 accuracy 분석
5. confidence 기반 reliability 적용

최종 best 모델의 shot top-2 accuracy는 `0.9745`입니다. 이는 모델이 완전히 엉뚱하게 틀리는 것이 아니라 가까운 shot 후보 사이에서 혼동한다는 것을 보여줍니다.

## 발표용 해석

```text
Shot-type과 text 여부는 서로 다른 visual cue를 요구한다.
따라서 각 subtask에 적합한 representation을 분리해 실험했고,
semantic feature, structural feature, text-sensitive feature를 결합했을 때 가장 높은 joint reliability를 얻었다.
```
