# 평가 지표 정리

## 모델 구분

- 성능 최고 모델: CLIP + DINOv2 ensemble
- 최종 파이프라인 적용 모델: CLIP ViT-B/32 frozen embedding + stronger multi-task head

성능 비교에서는 CLIP + DINOv2 ensemble을 최고 성능 모델로 기록합니다. 실제 overlay pipeline에는 속도와 안정성을 고려하여 CLIP stronger head checkpoint를 적용합니다.

## 최종 파이프라인 적용 모델 성능

- 모델: CLIP ViT-B/32 frozen embedding + stronger multi-task head
- checkpoint: `checkpoints/clip_vit_b32_multitask_head.pt`

```text
Shot Accuracy:          0.807
Shot Macro F1:          0.808
Shot Balanced Accuracy: 0.829
Shot Weighted F1:       0.806
Shot Top-2 Accuracy:    0.971

Text Accuracy:          0.895
Text Macro F1:          0.798
Text Weighted F1:       0.899

Joint Accuracy:         0.727
Joint Macro F1:         0.651
Joint Weighted F1:      0.729
```

## 최고 성능 실험 모델 성능

- 모델: CLIP + DINOv2 ensemble
- 용도: 성능 개선 실험 및 보고서 비교 결과

```text
Shot Accuracy: 0.836
Shot Macro F1: 0.839
Joint Accuracy: 0.731
```

## Shot Type별 F1-score

최종 파이프라인 적용 모델 기준입니다.

- `close-up`: `0.785`
- `medium`: `0.784`
- `object`: `0.874`
- `space`: `0.792`
- `wide`: `0.806`

## 종합 지표 산출물

- 위치: `outputs/model_experiments/comprehensive_metrics/`
- 주요 파일:
  - `01_basic_classifier_metrics.csv`
  - `01_shot_per_class_metrics.csv`
  - `02_zero_shot_vs_finetuned.csv`
  - `03_joint_reliability.csv`
  - `04_error_propagation.csv`
  - `05_calibration_summary.csv`
  - `09_multi_seed_stability_summary.csv`
  - `final_metric_summary.json`
