# 진행 상황 요약

## 완료된 작업

- 인스타그램 릴스 링크 입력 기반 shot guide task 정의
- `wide`, `medium`, `close-up`, `object`, `space` shot-type 라벨 체계 정리
- `text` / `notext` binary text presence 라벨 적용
- `deep/labeled_dataset` 기준 데이터 경로 정리
- CLIP ViT-B/32 embedding 기반 multi-task head 학습
- CLIP head 구조 개선 및 validation threshold tuning
- DINOv2 shot-only 실험
- CLIP+DINOv2 ensemble 실험
- focal loss 기반 hyperparameter tuning
- CLIP ViT-L/14, ViT-H/14 대형 backbone 실험
- SigLIP, SigLIP2 backbone 실험
- DINOv3 단독 및 CLIP+DINOv3 ensemble 실험
- DINOv3+SigLIP2 ensemble 실험
- CLIP+DINOv2 shot branch + SigLIP2 text branch 조합 실험
- C-RADIOv4-SO400M 단일 backbone 실험
- 모델별 comprehensive metrics 산출
- shot-type별 per-class 분석
- calibration, confidence threshold trade-off 분석
- 공통 오분류 이미지 분석
- 인스타그램 overlay pipeline 개선
- scene detection frame encoding 속도 개선
- text threshold validation tuning 반영

## 실험 흐름과 이유

| 순서 | 실험 | 진행 이유 | 관찰 |
|---:|---|---|---|
| 1 | CLIP ViT-B/32 baseline | image-text pretraining 기반 semantic feature가 task에 적합한지 확인 | 기본 성능 확보 |
| 2 | CLIP head 개선 | 모델을 늘리기 전에 head 구조와 threshold만으로 개선 가능한지 확인 | stronger multi-task head 성능 개선 |
| 3 | CLIP+DINOv2 | shot-type은 구도와 공간 구조가 중요하므로 DINOv2 structural feature 추가 | shot 성능 크게 개선 |
| 4 | hyperparameter tuning | alpha, dropout, class weight, focal loss 등 최적화 | focal loss가 가장 효과적 |
| 5 | CLIP-L/H, SigLIP, SigLIP2 | 더 큰/최신 vision-language backbone 비교 | SigLIP2가 text에 강함 |
| 6 | DINOv3 | DINOv2의 최신 대체 후보 검증 | 현재 데이터셋에서는 DINOv2보다 낮음 |
| 7 | CLIP+DINOv3 | DINOv3도 CLIP과 결합하면 보완되는지 확인 | CLIP+DINOv2보다 낮음 |
| 8 | CLIP+DINOv2 shot + SigLIP2 text | shot과 text를 각 모델이 잘하는 branch로 분리 | 현재 최고 joint acc |
| 9 | C-RADIOv4 | 3-model 구조를 단일 distilled backbone으로 대체 가능한지 확인 | 단일 모델로 joint acc 0.8000 |
| 10 | DINOv3+SigLIP2 | 최신 DINO 계열과 SigLIP2 조합 확인 | joint acc 0.8000 |

## 주요 모델 결과

| 모델 | 모델 수 | shot acc | shot macro F1 | text acc | text macro F1 | joint acc |
|---|---:|---:|---:|---:|---:|---:|
| CLIP stronger head | 1 | 0.8073 | 0.8081 | 0.8945 | 0.7982 | 0.7273 |
| CLIP+DINOv2 focal | 2 | 0.8618 | 0.8603 | 0.8945 | 0.7943 | 0.7600 |
| DINOv3 single | 1 | 0.8255 | 0.8038 | 0.9018 | 0.7706 | 0.7455 |
| CLIP+DINOv3 | 2 | 0.8400 | 0.8425 | 0.8945 | 0.7943 | 0.7527 |
| SigLIP2 single | 1 | 0.8182 | 0.8312 | 0.9455 | 0.8995 | 0.7745 |
| DINOv3+SigLIP2 | 2 | 0.8436 | 0.8518 | 0.9455 | 0.8995 | 0.8000 |
| C-RADIOv4-SO400M | 1 | 0.8255 | 0.8216 | 0.9745 | 0.9493 | 0.8000 |
| **CLIP+DINOv2 shot + SigLIP2 text** | **3** | **0.8618** | **0.8603** | **0.9527** | **0.9078** | **0.8218** |

## 현재 최고 성능 모델

```text
shot branch: CLIP ViT-B/32 + DINOv2 ViT-S/14 focal ensemble
text branch: SigLIP2 ViT-B/16-256 text head
alpha_clip: 0.46
text_threshold: 0.46
joint_accuracy: 0.8218
```

## 핵심 문제점

현재 병목은 text보다 shot-type입니다.

```text
CLIP+DINOv2 shot + SigLIP2 text
shot-only error ratio: 0.1309
text-only error ratio: 0.0400
```

주요 혼동쌍:

| 실제 라벨 | 잘못 예측 |
|---|---|
| medium | wide |
| medium | close-up |
| close-up | object |
| close-up | medium |

즉 라벨 경계가 애매한 구도형 클래스가 프로젝트의 핵심 어려움입니다.

## 향후 개선점

- `medium`, `close-up`, `wide`, `object` 라벨 기준 재정의
- 공통 오분류 샘플 재검수
- 컷 내부 단일 프레임이 아니라 multi-frame representation 사용
- temporal context 기반 shot classification
- confidence 낮은 컷은 guide 출력을 제한하는 reliability-aware pipeline
- scene detection threshold 라벨 기반 정량 평가
- annotator agreement 측정
- human evaluation 추가
