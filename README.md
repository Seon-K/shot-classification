# ShotGuide

인스타그램 릴스 링크 또는 숏폼 영상을 입력하면 컷 전환을 탐지하고, 각 컷의 `shot_type`과 화면 내 `text` 여부를 예측한 뒤 overlay 촬영 가이드 영상으로 생성하는 프로젝트입니다.

## Task

- Cut / scene transition detection
- Shot-type classification: `wide`, `medium`, `close-up`, `object`, `space`
- Text presence classification: `text`, `notext`
- Scene별 prediction과 confidence 기반 guide 생성

## 프로젝트 구조

```text
deep/
├─ labeled_dataset/
├─ checkpoints/
├─ src/
├─ experiments/
├─ outputs/
└─ docs/
```

주요 문서:

- `docs/model_design.md`: 모델 설계와 선택 논리
- `docs/file_roles.md`: 파일별 역할
- `docs/evaluation_metrics.md`: 평가 지표와 결과
- `docs/progress_summary.md`: 진행 상황 요약

## 핵심 문제 정의

이 프로젝트의 가장 큰 어려움은 shot-type 라벨의 모호성입니다. `medium`, `wide`, `close-up`, `object`는 단순 객체 분류가 아니라 피사체 크기, 배경 비중, 공간감, 중심 피사체의 종류를 함께 고려해야 합니다.

주요 공통 오분류 패턴은 다음과 같습니다.

| 실제 라벨 | 자주 틀린 라벨 | 원인 |
|---|---|---|
| `medium` | `wide` | 배경과 공간 비중 경계가 애매함 |
| `medium` | `close-up` | 피사체 크기 기준이 애매함 |
| `close-up` | `object` | 사람/물체 중심 클로즈업 혼동 |
| `close-up` | `medium` | 얼굴/상반신 비율 경계가 애매함 |

따라서 단순히 큰 모델 하나를 사용하는 방식보다, 각 subtask에 필요한 visual representation을 실험적으로 비교했습니다.

## 모델 실험 흐름

1. **CLIP ViT-B/32 baseline**: image-text pretraining 기반 semantic representation 확인
2. **CLIP head 개선**: multi-task head, class weight, dropout, validation threshold tuning 적용
3. **CLIP + DINOv2 ensemble**: CLIP의 semantic cue와 DINOv2의 structural visual cue 결합
4. **Hyperparameter tuning**: alpha search, focal loss, dropout, layer norm, label smoothing 등 비교
5. **CLIP-L/14, CLIP-H/14, SigLIP, SigLIP2**: 더 큰 또는 최신 vision-language backbone 비교
6. **DINOv3**: DINOv2의 최신 대체 후보 검증
7. **Task-specific branch 조합**: shot은 CLIP+DINOv2, text는 SigLIP2가 담당
8. **C-RADIOv4**: 3-model 구조를 단일 distilled backbone으로 대체할 수 있는지 확인

## 주요 결과

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

## 현재 결론

최고 성능은 **CLIP+DINOv2 shot branch + SigLIP2 text branch**에서 나왔습니다.

```text
shot: CLIP ViT-B/32 + DINOv2 ViT-S/14 focal ensemble
text: SigLIP2 ViT-B/16-256 text head
joint accuracy: 0.8218
```

이 구조는 단순히 모델을 많이 붙인 것이 아니라, task를 다음처럼 분해한 결과입니다.

- CLIP: 장면의 semantic context
- DINOv2: 구도, 피사체 배치, visual structure
- SigLIP2: text/overlay presence에 강한 vision-language feature

다만 모델 3개를 사용하는 구조는 추론 비용과 유지보수 비용이 큽니다. 그래서 단일 backbone 대안으로 C-RADIOv4-SO400M도 실험했습니다.

```text
C-RADIOv4-SO400M single backbone
joint accuracy: 0.8000
text accuracy: 0.9745
```

발표에서는 최고 성능 모델과 단일 backbone 대안을 함께 제시하여 성능-복잡도 trade-off를 설명하는 것이 적절합니다.

## 실행 방법

최종 파이프라인 실행:

```bash
python3 src/shotguide_final_instagram_pipeline.py
```

주요 실험 실행 예시:

```bash
python3 experiments/evaluate_clip_dinov2_shot_siglip2_text_combo.py
python3 experiments/run_cradiov4_backbone_experiment.py
python3 experiments/evaluate_dinov3_siglip2_ensemble_experiment.py
```

## 주요 결과물

```text
outputs/model_experiments/clip_dinov2_shot_siglip2_text_combo/
outputs/model_experiments/cradiov4_backbones/
outputs/model_experiments/dinov3_siglip2_ensemble/
outputs/model_experiments/common_error_analysis/
outputs/final_instagram_pipeline/
```
