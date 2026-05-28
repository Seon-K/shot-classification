# 주요 파일 역할

## src

- `src/shotguide_batch_video_inference.py`
  - CLIP/head 모델 로드
  - scene frame 추출
  - shot type/text 여부 예측
  - guide text 생성
  - overlay video 생성
  - validation 기반 `text_threshold` 적용

- `src/shotguide_instagram_overlay_pipeline.py`
  - 인스타그램 영상 다운로드
  - CLIP distance 기반 scene detection
  - scene frame encoding
  - overlay pipeline 실행
  - cut detection 평가 함수 포함

- `src/shotguide_final_instagram_pipeline.py`
  - 인스타그램 링크 입력부터 overlay mp4 생성까지 이어지는 통합 pipeline

## experiments: 기존 주요 실험

- `experiments/retrain_clip_head_current_dataset.py`
  - 현재 dataset 기준 CLIP head 재학습
  - dataset path는 `deep/labeled_dataset` 기준

- `experiments/run_clip_head_experiments.py`
  - CLIP head 구조 실험

- `experiments/run_dinov2_shot_experiment.py`
  - DINOv2 shot-only 실험

- `experiments/run_clip_dinov2_ensemble_experiment.py`
  - CLIP + DINOv2 ensemble 실험

- `experiments/save_best_clip_stronger_checkpoint.py`
  - CLIP stronger head checkpoint 저장

- `experiments/compute_comprehensive_metrics.py`
  - 종합 평가 지표 계산

## experiments: 새로 추가한 확장 실험

- `experiments/run_clip_large_backbone_experiments.py`
  - CLIP ViT-L/14, ViT-H/14 실험

- `experiments/run_siglip_backbone_experiments.py`
  - SigLIP backbone 실험

- `experiments/run_siglip2_backbone_experiments.py`
  - SigLIP2 ViT-B/16-256 단일 backbone 실험

- `experiments/run_dinov3_backbone_experiments.py`
  - DINOv3 ViT-S/16 단일 backbone 실험

- `experiments/run_cradiov4_backbone_experiment.py`
  - C-RADIOv4-SO400M 단일 backbone 실험

- `experiments/tune_clip_dinov2_ensemble_hyperparams.py`
  - CLIP+DINOv2 ensemble hyperparameter tuning

- `experiments/evaluate_focal_loss_per_class.py`
  - focal loss 기반 CLIP+DINOv2 ensemble의 shot-type별 성능 분석

- `experiments/evaluate_clip_dinov2_ensemble_comprehensive.py`
  - CLIP+DINOv2 ensemble comprehensive metric 산출

- `experiments/evaluate_clip_dinov2_shot_siglip2_text_combo.py`
  - 현재 최고 성능 실험
  - shot은 CLIP+DINOv2 ensemble, text는 SigLIP2 text head 사용

- `experiments/evaluate_clip_dinov3_ensemble_experiment.py`
  - CLIP+DINOv3 ensemble 실험

- `experiments/evaluate_dinov3_siglip2_ensemble_experiment.py`
  - DINOv3+SigLIP2 ensemble 실험

## outputs

- `outputs/clip_embeddings/`
  - CLIP ViT-B/32 embedding과 metadata

- `outputs/model_experiments/clip_head/`
  - CLIP head 구조 실험 결과

- `outputs/model_experiments/dinov2_shot/`
  - DINOv2 shot-only embedding과 결과

- `outputs/model_experiments/ensemble_hyperparam_tuning/`
  - CLIP+DINOv2 hyperparameter tuning 결과

- `outputs/model_experiments/clip_large_backbones/`
  - CLIP ViT-L/14, ViT-H/14 실험 결과

- `outputs/model_experiments/siglip_backbones/`
  - SigLIP backbone 실험 결과

- `outputs/model_experiments/siglip2_backbones/`
  - SigLIP2 단일 backbone 실험 결과

- `outputs/model_experiments/dinov3_backbones/`
  - DINOv3 단일 backbone 실험 결과

- `outputs/model_experiments/cradiov4_backbones/`
  - C-RADIOv4-SO400M 단일 backbone 실험 결과

- `outputs/model_experiments/clip_dinov2_shot_siglip2_text_combo/`
  - 현재 최고 성능 조합 모델 결과

- `outputs/model_experiments/clip_dinov3_ensemble/`
  - CLIP+DINOv3 ensemble 결과

- `outputs/model_experiments/dinov3_siglip2_ensemble/`
  - DINOv3+SigLIP2 ensemble 결과

- `outputs/model_experiments/common_error_analysis/`
  - 모델 공통 오분류 분석
  - shot-type 3개 이상 모델 오분류 샘플
  - confusion pair summary

- `outputs/instagram_overlay/`
  - overlay pipeline 결과

- `outputs/final_instagram_pipeline/`
  - 최종 pipeline 결과
