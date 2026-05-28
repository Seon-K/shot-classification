# 종합 평가 지표 요약

이 문서는 현재까지의 주요 모델 실험 결과를 발표용 관점으로 요약합니다.

## 평가 목표

ShotGuide의 최종 목표는 각 컷에서 다음 두 가지를 동시에 맞히는 것입니다.

```text
1. shot_type: close-up, medium, wide, object, space
2. has_text: text / notext
```

따라서 최종 성능은 단일 task accuracy보다 `joint accuracy`와 `guide reliability`가 더 중요합니다.

## 주요 모델 비교

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

## 현재 Best Model

```text
shot: CLIP ViT-B/32 + DINOv2 ViT-S/14 focal ensemble
text: SigLIP2 ViT-B/16-256 text head
```

핵심 지표:

| 지표 | 값 |
|---|---:|
| joint accuracy | 0.8218 |
| guide reliability score | 0.8218 |
| shot accuracy | 0.8618 |
| shot macro F1 | 0.8603 |
| shot top-2 accuracy | 0.9745 |
| text accuracy | 0.9527 |
| text macro F1 | 0.9078 |

Error propagation:

| 지표 | 값 |
|---|---:|
| both correct ratio | 0.8218 |
| shot-only error ratio | 0.1309 |
| text-only error ratio | 0.0400 |
| both wrong ratio | 0.0073 |

## 단일 Backbone 대안

C-RADIOv4-SO400M:

| 지표 | 값 |
|---|---:|
| joint accuracy | 0.8000 |
| shot accuracy | 0.8255 |
| text accuracy | 0.9745 |
| joint ECE | 0.0604 |

C-RADIOv4는 최고 성능 모델보다 joint accuracy는 낮지만, 단일 backbone으로 높은 성능을 보여 성능-복잡도 trade-off 측면에서 의미 있는 대안입니다.

## 핵심 오류 분석

여러 모델이 공통적으로 틀린 shot-type 샘플을 분석한 결과, 주요 혼동은 다음과 같습니다.

| 실제 라벨 | 잘못 예측 | 모델 오답 vote |
|---|---|---:|
| medium | wide | 58 |
| medium | close-up | 50 |
| close-up | object | 49 |
| close-up | medium | 43 |
| medium | space | 20 |

해석:

```text
라벨 ambiguity가 가장 큰 병목이다.
특히 medium, close-up, wide, object의 경계가 애매하다.
```

## Runtime 요약

CPU 기준입니다.

| 모델 | embedding extraction | total runtime |
|---|---:|---:|
| DINOv3 single | 91.41s | 95.55s |
| SigLIP2 single | 234.24s | 236.40s |
| C-RADIOv4-SO400M | 976.71s | 980.55s |

cached embedding 기준 조합 실험:

| 모델 | cached runtime |
|---|---:|
| CLIP+DINOv2 shot + SigLIP2 text | 4.14s |
| CLIP+DINOv3 | 4.25s |
| DINOv3+SigLIP2 | 3.80s |

## 발표용 결론

```text
Shot-type과 text presence는 서로 다른 visual cue를 요구한다.
CLIP+DINOv2는 shot-type의 semantic/structural cue를 잘 포착했고,
SigLIP2는 text presence에서 강한 성능을 보였다.
두 branch를 결합했을 때 최고 joint reliability를 얻었다.

다만 3-model 구조는 무겁기 때문에,
C-RADIOv4 같은 단일 distilled backbone을 성능-복잡도 trade-off 대안으로 함께 제시한다.
```
