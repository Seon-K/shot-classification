# 종합 평가 지표 요약

## 모델 구분

- 성능 최고 모델: CLIP + DINOv2 ensemble
- 최종 파이프라인 적용 모델: CLIP ViT-B/32 frozen embedding + stronger multi-task head

이 파일의 핵심 지표는 최종 파이프라인 적용 모델 기준입니다. CLIP + DINOv2 ensemble은 별도 실험 결과에서 shot type 기준 최고 성능 모델로 기록합니다.

## 최종 파이프라인 적용 모델 핵심 지표

- shot_accuracy: 0.8073
- shot_macro_f1: 0.8081
- shot_balanced_accuracy: 0.8286
- shot_weighted_f1: 0.8058
- shot_top2_accuracy: 0.9709
- text_accuracy: 0.8945
- text_macro_f1: 0.7982
- text_weighted_f1: 0.8991
- joint_accuracy: 0.7273
- joint_macro_f1: 0.6510
- joint_weighted_f1: 0.7290
- joint_micro_f1: 0.7273

## 최고 성능 실험 모델

- 모델: CLIP + DINOv2 ensemble
- shot_accuracy: 0.836
- shot_macro_f1: 0.839
- joint_accuracy: 0.731

## 참고

- zero-shot 지표는 같은 test split에서 CLIP text prompt similarity로 계산
- scene mismatch, annotation agreement, human evaluation, full runtime metrics는 추가 ground truth 또는 log 필요
- 생성된 CSV 파일 위치: `outputs/model_experiments/comprehensive_metrics/`
