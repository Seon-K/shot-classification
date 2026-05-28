# ShotGuide

숏폼 영상을 입력하면 scene별 촬영 구도와 화면 내 text 여부를 예측하고, 결과를 overlay 촬영 가이드 영상으로 생성하는 프로젝트입니다.

## 프로젝트 구조

```text
shot-classification/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ labeled_dataset/
│  ├─ videos_new_links/
│  └─ outputs_clip_frame_extraction/
├─ checkpoints/
│  ├─ clip_vit_b32_multitask_head.pt
│  └─ clip_vit_b32_multitask_head_baseline256_backup.pt
├─ src/
│  ├─ shotguide_batch_video_inference.py
│  ├─ shotguide_instagram_overlay_pipeline.py
│  └─ shotguide_final_instagram_pipeline.py
├─ notebooks/
│  ├─ shotguide_clip_embedding_baseline.ipynb
│  ├─ shotguide_clip_scene_frame_extraction.ipynb
│  ├─ shotguide_model_improvement_evaluation.ipynb
│  └─ shotguide_final_instagram_pipeline.ipynb
├─ experiments/
│  ├─ retrain_clip_head_current_dataset.py
│  ├─ run_clip_head_experiments.py
│  ├─ run_dinov2_shot_experiment.py
│  ├─ run_clip_dinov2_ensemble_experiment.py
│  ├─ save_best_clip_stronger_checkpoint.py
│  └─ compute_comprehensive_metrics.py
├─ outputs/
│  ├─ baseline/
│  ├─ clip_embeddings/
│  ├─ model_experiments/
│  ├─ instagram_overlay/
│  └─ final_instagram_pipeline/
└─ docs/
   ├─ model_design.md
   ├─ file_roles.md
   ├─ evaluation_metrics.md
   └─ progress_summary.md
```

## 모델 구분

본 프로젝트에서는 모델을 두 기준으로 구분합니다.

- 성능 최고 모델: CLIP + DINOv2 ensemble
- 최종 파이프라인 적용 모델: CLIP ViT-B/32 frozen embedding + stronger multi-task head

CLIP + DINOv2 ensemble은 shot type 분류 성능이 가장 높게 나왔습니다. 다만 DINOv2 feature 추출 시간이 길어 인스타그램 링크 입력부터 overlay 영상 생성까지 이어지는 전체 파이프라인에는 부담이 있습니다.

따라서 실제 데모 및 최종 overlay pipeline에는 CLIP 기반 stronger multi-task head를 적용합니다. 성능 비교 결과에서는 CLIP + DINOv2 ensemble을 최고 성능 모델로 기록합니다.

## 최종 파이프라인 적용 모델

- CLIP ViT-B/32 frozen image encoder
- 512차원 CLIP image embedding 사용
- stronger multi-task head 적용
- shared layer: `512 -> 512 -> 256`
- `shot_head`: `close-up`, `medium`, `wide`, `object`, `space`
- `text_head`: `notext`, `text`

적용 checkpoint:

```text
checkpoints/clip_vit_b32_multitask_head.pt
```

## 주요 성능

### 최종 파이프라인 적용 모델

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

### 최고 성능 실험 모델

```text
CLIP + DINOv2 ensemble
Shot Accuracy: 0.836
Shot Macro F1: 0.839
Joint Accuracy: 0.731
```

## 실행 방법

최종 파이프라인 실행:

```bash
python src/shotguide_final_instagram_pipeline.py
```

노트북 실행:

```text
notebooks/shotguide_final_instagram_pipeline.ipynb
```

분석할 인스타그램 링크는 코드의 `INSTAGRAM_LINKS`에 추가합니다.

## 결과물

최종 pipeline 결과:

```text
outputs/final_instagram_pipeline/
```

각 영상별 생성 파일:

```text
scene_frames/
clip_distance_profile.csv
scene_metadata.csv
scene_predictions.csv
scene_clip_embeddings.npz
*_overlay.mp4
```

## 문서

- `docs/model_design.md`: 모델 설계 기준
- `docs/file_roles.md`: 파일별 역할
- `docs/evaluation_metrics.md`: 평가 지표 정리
- `docs/progress_summary.md`: 진행 상황 요약
