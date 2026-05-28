# 주요 파일 역할

## src

- `src/shotguide_batch_video_inference.py`
  - CLIP/head 모델 로드
  - scene frame 추출
  - shot type/text 여부 예측
  - guide text 생성
  - overlay video 생성

- `src/shotguide_instagram_overlay_pipeline.py`
  - 인스타그램 영상 다운로드
  - CLIP distance 기반 scene detection
  - 기존 overlay pipeline 실행

- `src/shotguide_final_instagram_pipeline.py`
  - 최종 제출용 통합 pipeline
  - 인스타그램 링크 입력부터 overlay mp4 생성까지 실행
  - 최종 파이프라인 적용 모델인 CLIP stronger head 사용

## notebooks

- `notebooks/shotguide_clip_embedding_baseline.ipynb`
  - CLIP embedding 기반 head 학습 노트북
  - `clip_vit_b32_multitask_head.pt` 생성 흐름 확인

- `notebooks/shotguide_clip_scene_frame_extraction.ipynb`
  - 추가 라벨링용 frame 추출 노트북
  - 영상 번호와 이미지 번호 기준으로 저장

- `notebooks/shotguide_model_improvement_evaluation.ipynb`
  - 모델 평가 및 개선 실험 결과 정리
  - CLIP stronger head, DINOv2, ensemble 성능 비교

- `notebooks/shotguide_final_instagram_pipeline.ipynb`
  - 최종 pipeline 실행용 노트북
  - 코드 실행 흐름을 notebook 형태로 확인

## experiments

- `experiments/retrain_clip_head_current_dataset.py`
  - 현재 labeled dataset 기준 CLIP head 재학습

- `experiments/run_clip_head_experiments.py`
  - CLIP head 구조 실험
  - dropout, hidden layer, loss 등 비교

- `experiments/run_dinov2_shot_experiment.py`
  - DINOv2 shot-only 실험
  - shot type 분류 성능 비교

- `experiments/run_clip_dinov2_ensemble_experiment.py`
  - CLIP + DINOv2 ensemble 실험
  - 성능 최고 모델 산출

- `experiments/save_best_clip_stronger_checkpoint.py`
  - 최종 파이프라인 적용용 CLIP stronger head checkpoint 저장

- `experiments/compute_comprehensive_metrics.py`
  - 종합 평가 지표 계산
  - accuracy, F1, joint reliability, calibration, multi-seed stability 등 산출

## checkpoints

- `checkpoints/clip_vit_b32_multitask_head.pt`
  - 최종 overlay pipeline에서 사용하는 CLIP stronger head checkpoint

- `checkpoints/clip_vit_b32_multitask_head_baseline256_backup.pt`
  - 이전 baseline head backup checkpoint

## outputs

- `outputs/baseline/`
  - dataset index, split 정보, ResNet baseline 결과

- `outputs/clip_embeddings/`
  - CLIP embedding, prediction, metric 결과

- `outputs/model_experiments/`
  - DINOv2, ensemble, comprehensive metrics 결과

- `outputs/instagram_overlay/`
  - 기존 overlay pipeline 결과

- `outputs/final_instagram_pipeline/`
  - 최종 pipeline 결과
