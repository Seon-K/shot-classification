# 진행 상황 요약

## 완료

- 숏폼 영상 shot classification 문제 정의
- 라벨 체계 정리
- labeled dataset 구축 및 검수
- CLIP embedding 기반 multi-task head 학습
- close-up 데이터 보강 후 재학습
- stronger head 구조 실험
- DINOv2 shot-only 비교 실험
- CLIP + DINOv2 ensemble 비교 실험
- 종합 평가 지표 계산
- 인스타그램 링크 입력 기반 overlay pipeline 구현
- 프로젝트 폴더 구조 정리

## 모델 실험 결과 정리

- 기존 ResNet18 frozen backbone baseline보다 CLIP 기반 모델 성능 우수
- 기존 1-layer CLIP head보다 stronger multi-task head 성능 개선
- DINOv2 shot-only 실험 진행
- CLIP + DINOv2 ensemble이 shot type 기준 최고 성능 기록

## 최종 선택 기준

- 성능 최고 모델: CLIP + DINOv2 ensemble
- 최종 pipeline 적용 모델: CLIP stronger head

선택 이유:

- CLIP + DINOv2 ensemble은 성능이 가장 높음
- DINOv2 feature 추출 시간이 길어 전체 overlay pipeline에는 부담
- CLIP stronger head는 성능, 속도, 구현 안정성의 균형이 좋음
- 최종 데모와 새 링크 처리에는 CLIP stronger head 적용

## 향후 개선점

- close-up / medium 경계 샘플 추가 검수
- notext 계열 데이터 보강
- scene detection threshold 정교화
- DINOv2 ensemble pipeline 최적화 가능성 검토
- human evaluation 설문 추가
- annotation agreement 측정
