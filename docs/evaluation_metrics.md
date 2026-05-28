# 평가 지표 정리

## 평가 관점

이 프로젝트는 `shot_type`과 `text presence`를 동시에 맞춰야 하는 task입니다. 따라서 단일 accuracy보다 joint accuracy와 guide reliability가 중요합니다.

평가 축:

1. Shot classifier 성능
2. Text classifier 성능
3. Joint reliability
4. Error propagation
5. Calibration / confidence reliability
6. Per-class / confusion analysis
7. Runtime / efficiency
8. 성능-복잡도 trade-off

## 핵심 지표

### Shot Classifier

- accuracy
- macro F1
- balanced accuracy
- weighted F1
- top-2 accuracy
- macro precision
- macro recall
- per-class F1

### Text Classifier

- accuracy
- macro F1
- balanced accuracy
- weighted F1
- macro precision
- macro recall
- text / notext per-class F1

### Joint / Reliability

- joint accuracy
- guide reliability score
- conditional text given shot
- conditional shot given text
- both correct ratio
- shot-only error ratio
- text-only error ratio
- both wrong ratio

### Calibration

- ECE
- MCE
- high-confidence reliability
- confidence threshold trade-off

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
Model: CLIP+DINOv2 shot ensemble + SigLIP2 text head
shot_source: CLIP ViT-B/32 + DINOv2 ViT-S/14 focal ensemble
text_source: SigLIP2 ViT-B/16-256 text head
alpha_clip: 0.46
text_threshold: 0.46
```

### 전체 성능

| 지표 | 값 |
|---|---:|
| joint accuracy | 0.8218 |
| guide reliability score | 0.8218 |
| shot accuracy | 0.8618 |
| shot macro F1 | 0.8603 |
| shot balanced accuracy | 0.8889 |
| shot weighted F1 | 0.8619 |
| shot top-2 accuracy | 0.9745 |
| text accuracy | 0.9527 |
| text macro F1 | 0.9078 |
| text balanced accuracy | 0.9394 |
| text weighted F1 | 0.9544 |

### Error Propagation

| 지표 | 값 |
|---|---:|
| both correct ratio | 0.8218 |
| shot-only error ratio | 0.1309 |
| text-only error ratio | 0.0400 |
| both wrong ratio | 0.0073 |
| conditional text given shot | 0.9536 |
| conditional shot given text | 0.8626 |

현재 joint accuracy를 제한하는 주된 원인은 text가 아니라 shot-type 오류입니다.

### Confidence Threshold Trade-off

| threshold | coverage | reliability |
|---:|---:|---:|
| 0.5 | 0.8473 | 0.8584 |
| 0.6 | 0.6291 | 0.8902 |
| 0.7 | 0.4691 | 0.9147 |
| 0.8 | 0.2764 | 0.9342 |
| 0.9 | 0.0473 | 1.0000 |

## C-RADIOv4 단일 Backbone 결과

| 지표 | 값 |
|---|---:|
| joint accuracy | 0.8000 |
| shot accuracy | 0.8255 |
| shot macro F1 | 0.8216 |
| text accuracy | 0.9745 |
| text macro F1 | 0.9493 |
| joint ECE | 0.0604 |

C-RADIOv4는 3-model best보다 joint acc는 낮지만, 단일 backbone으로 `0.8000`까지 도달했습니다.

## DINOv3 관련 결과

| 모델 | shot acc | shot macro F1 | text acc | text macro F1 | joint acc |
|---|---:|---:|---:|---:|---:|
| DINOv3 single | 0.8255 | 0.8038 | 0.9018 | 0.7706 | 0.7455 |
| CLIP+DINOv3 | 0.8400 | 0.8425 | 0.8945 | 0.7943 | 0.7527 |
| DINOv3+SigLIP2 | 0.8436 | 0.8518 | 0.9455 | 0.8995 | 0.8000 |

DINOv3는 DINOv2보다 최신이지만 현재 데이터셋에서는 DINOv2를 대체하지 못했습니다.

## Shot-type 공통 오류 분석

주요 7개 모델 중 3개 이상이 틀린 shot-type 샘플을 분석했습니다.

| 실제 라벨 | 잘못 예측 | 샘플 수 | 모델 오답 vote |
|---|---|---:|---:|
| medium | wide | 16 | 58 |
| medium | close-up | 11 | 50 |
| close-up | object | 8 | 49 |
| close-up | medium | 9 | 43 |
| medium | space | 5 | 20 |
| object | space | 2 | 11 |

결론:

```text
현재 핵심 병목은 text가 아니라 shot-type ambiguity다.
특히 medium, wide, close-up, object 사이의 경계가 모델 공통 오류로 반복된다.
```

## Runtime

CPU 기준 runtime입니다. CUDA 드라이버 문제로 GPU는 사용되지 않았습니다.

| 모델 | embedding extraction | fit time | classifier inference | total runtime | throughput |
|---|---:|---:|---:|---:|---:|
| DINOv3 single | 91.41s | 4.13s | 0.0026s | 95.55s | 22.01 samples/s |
| SigLIP2 single | 234.24s | 2.16s | 0.0029s | 236.40s | 8.59 samples/s |
| C-RADIOv4-SO400M | 976.71s | 3.83s | 0.0038s | 980.55s | 2.06 samples/s |

cached embedding 기준 조합 실험:

| 모델 | cached runtime |
|---|---:|
| CLIP+DINOv2 shot + SigLIP2 text | 4.14s |
| CLIP+DINOv3 | 4.25s |
| DINOv3+SigLIP2 | 3.80s |

## 주요 산출물

```text
outputs/model_experiments/clip_dinov2_shot_siglip2_text_combo/summary_metrics.json
outputs/model_experiments/cradiov4_backbones/best_cradiov4_backbone_experiment.json
outputs/model_experiments/dinov3_siglip2_ensemble/summary_metrics.json
outputs/model_experiments/common_error_analysis/shot_type_wrong_3plus_models.csv
outputs/model_experiments/common_error_analysis/shot_type_wrong_3plus_confusion_pairs.csv
```
