# ShotGuide 모델 설계 기준

## 목적

- 숏폼 영상의 scene별 촬영 구도 분석
- `shot_type`과 `has_text` 동시 예측
- 예측 결과를 overlay 촬영 가이드로 변환

## 라벨 분리 기준

- `shot_type`: `close-up`, `medium`, `wide`, `object`, `space`
- `has_text`: `0`, `1`
- 결합 라벨 대신 multi-task 구조 사용
- 클래스 수 증가 방지
- text 존재 여부가 shot type 판단을 과도하게 방해하는 문제 완화

## Multi-task 구조

- 하나의 이미지 embedding 공유
- `shot_head`와 `text_head`를 분리
- shot type과 text 여부를 동시에 예측
- 공통 시각 특징은 공유하고, 최종 판단은 task별 head에서 수행

```text
Input image
↓
CLIP ViT-B/32 image encoder
↓
512-dimensional image embedding
↓
Shared hidden layers
├─ shot_head → shot_type 예측
└─ text_head → has_text 예측
```

## 모델 선정 기준

- 성능 최고 모델: CLIP + DINOv2 ensemble
- 최종 파이프라인 적용 모델: CLIP ViT-B/32 frozen embedding + stronger multi-task head

성능만 기준으로 보면 CLIP + DINOv2 ensemble이 가장 우수합니다. 다만 DINOv2는 feature 추출 시간이 길어 새 인스타그램 링크 입력부터 overlay 영상 생성까지 이어지는 실제 실행 파이프라인에는 부담이 있습니다.

따라서 보고서의 성능 비교에서는 CLIP + DINOv2 ensemble을 최고 성능 모델로 기록하고, 실제 overlay pipeline에는 CLIP 기반 stronger multi-task head를 적용합니다.

## 최종 파이프라인 적용 모델

- CLIP ViT-B/32 frozen image encoder
- 512차원 CLIP image embedding 사용
- stronger multi-task head 적용
- shared layer: `512 -> 512 -> 256`
- `shot_head`: 5개 shot type 예측
- `text_head`: text 여부 2개 class 예측

## 적용 이유

- 기존 1-layer head 대비 성능 개선
- DINOv2 ensemble 대비 빠른 실행
- 새 인스타그램 링크 처리와 overlay 생성에 적합
- 데모 및 반복 실행 안정성 확보
