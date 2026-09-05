# 모델 구현·학습 현황

갱신: 2026-09-06T02:27:46+09:00

사용자 지정 관리 Phase: **1 MambaVision/TransNeXt/TCMax**, **2 IVON/ABNN/NCG/HENN/LATA**, **3 MedRegA/DyMo/DARC/RadZero**.
기존 `configs/phases.yaml`과 `results/phase*`는 과거 실행 provenance이며 변경하지 않았다. 새 관리 Phase는 이 문서/CSV가 기준이다.
빈칸은 실행 없음 또는 미확정이며 0점이 아니다. 타깃은 대화에서 확인된 조합만 기록했다.

## Phase 1

| 모델 | 대상 데이터셋 | 구현·검증 | 학습·평가 | 현재 실행 위치(로컬 직접 확인) | 서버 마지막 확인(현재 상태 아님) | 데이터 확보 |
| --- | --- | --- | --- | --- | --- | --- |
| MambaVision | milk10k | 기존 러너 구현 / 신규 패키지 CUDA smoke 미검증 | 완료 결과 없음 / 과거 mamba_ssm 누락 차단 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음 |
| MambaVision | chexchonet | 기존 러너 구현 / 신규 패키지 CUDA smoke 미검증 | 5/5 완료 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |
| TransNeXt | milk10k | 기존 러너 구현 / 신규 CPU forward 검증·GPU smoke 미검증 | 5/5 완료 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음 |
| TransNeXt | chexchonet | 기존 러너 구현 / 신규 CPU forward 검증·GPU smoke 미검증 | 5/5 완료 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |
| TCMax | milk10k | 기존 두 데이터셋 러너 구현 / 논문 예산 적용판 CPU 검증·GPU smoke 미검증 | 완료 결과 없음 |  | 연세대 GPU 5·6, 2026-09-06 01:17:54 KST 사용자 캡처: 기존 benchmark 실행 / 현 dataset·fold 미확인 | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음 |
| TCMax | chexchonet | 기존 두 데이터셋 러너 구현 / 논문 예산 적용판 CPU 검증·GPU smoke 미검증 | 1/5 완료 / 로컬 실행 중 | 로컬 NixOS / RTX 2080 | 연세대 GPU 5·6, 2026-09-06 01:17:54 KST 사용자 캡처: 기존 benchmark 실행 / 현 dataset·fold 미확인 | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |

## Phase 2

| 모델 | 대상 데이터셋 | 구현·검증 | 학습·평가 | 현재 실행 위치(로컬 직접 확인) | 서버 마지막 확인(현재 상태 아님) | 데이터 확보 |
| --- | --- | --- | --- | --- | --- | --- |
| IVON | milk10k | 공식 v0.1.2 CPU 실행 검증 / 실제 데이터 학습 러너 미완성 | 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음 |
| IVON | chexchonet | 공식 v0.1.2 CPU 실행 검증 / 실제 데이터 학습 러너 미완성 | 완료 결과 없음 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |
| ABNN | milk10k | 공식 구현 조사 / 학습 어댑터 미구현 | 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음 |
| ABNN | chexchonet | 공식 구현 조사 / 학습 어댑터 미구현 | 완료 결과 없음 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |
| NCG | milk10k | 공식 구현 조사 / DAG·train-only 어댑터 미준비 | 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 방법별 추가 입력/아티팩트는 미준비 |
| HENN | milk10k | 공식 구현 조사 / 복합 라벨·어댑터 미준비 | 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 방법별 추가 입력/아티팩트는 미준비 |
| LATA | chexchonet | 방법 조사 / 공식 아티팩트·calibration 미준비 | 완료 결과 없음 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보; 방법별 추가 입력/아티팩트는 미준비 |

## Phase 3

| 모델 | 대상 데이터셋 | 구현·검증 | 학습·평가 | 현재 실행 위치(로컬 직접 확인) | 서버 마지막 확인(현재 상태 아님) | 데이터 확보 |
| --- | --- | --- | --- | --- | --- | --- |
| MedRegA | milk10k | 소스 정보 확보 / 환경·어댑터 미준비 | 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 방법별 추가 입력/아티팩트는 미준비 |
| DyMo | milk10k | 소스 확보 / 데이터셋 전용 학습 경로 미준비 | 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 방법별 추가 입력/아티팩트는 미준비 |
| DARC | chexchonet | 공식 구현·체크포인트 미확보 | 완료 결과 없음 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |
| RadZero | chexchonet | zero-shot 평가 러너 구현 | 5/5 완료 (zero-shot 평가) |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |

## Phase-ex — 추가 데이터셋 적용 검토

사용자 요청으로 추가 조합을 Phase-ex에 별도 분류했다. 구현·실험 설정은 아직 미확정이며 성능/실행 위치는 빈칸이다. 기본 Phase 2 대상과 혼합하지 않는다.

| 모델 | 대상 데이터셋 | 구현·검증 | 학습·평가 | 현재 실행 위치(로컬 직접 확인) | 서버 마지막 확인(현재 상태 아님) | 데이터 확보 |
| --- | --- | --- | --- | --- | --- | --- |
| NCG | chexchonet | 추가 적용 검토 / 해당 데이터셋 어댑터 미구현 | 미실행 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보; 추가 조합 프로토콜 미확정 |
| HENN | chexchonet | 추가 적용 검토 / 해당 데이터셋 어댑터 미구현 | 미실행 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보; 추가 조합 프로토콜 미확정 |
| LATA | milk10k | 추가 적용 검토 / 해당 데이터셋 어댑터 미구현 | 미실행 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 추가 조합 프로토콜 미확정 |

## 판독 주의

- 현재 서버 접속 없음: 연세대 작업의 완료 fold/현재 dataset/PID 매핑을 직접 검증하지 못했다. 로컬 TCMax와 서버 TCMax를 합산하지 않는다.
- `--runtime server`라는 이름만으로 실행 서버를 판단하지 않는다. 로컬 TCMax는 실제 RTX 2080에서 실행 중이다.
- 수정 패키지 `../model-suite-staging/`: CPU 18개 테스트 통과(기록); 신규 6개 조합 GPU smoke는 미확인. 기존 학습 성공이 새 패키지 검증을 대신하지 않는다.
- 기존 이미지 전용 MambaVision/TransNeXt와 멀티모달 TCMax의 입력·예산이 다르다. 논문 조건 동일 재현으로 표시하지 않는다.
- MILK 원본 메타데이터 조사상 결측: 나이 20병변(40행), 부위 31병변(62행); train 중앙값+indicator/MISSING 승인. paired 측정값 사용도 이후 사용자 승인되었으나 최신 staging의 비대칭 차단 구현과는 구분한다(현 데이터 비대칭 0).
- 데이터 수: 기존 MILK manifest 5,240병변, Chest manifest 71,589영상. 이는 파일 목록 행 수이며 모든 이미지의 바이트 무결성 검사를 뜻하지 않는다.
- NCG/HENN의 추가 Chest 적용, LATA의 추가 MILK 적용은 Phase-ex 검토 대상으로 분리했다. 성능과 구체 프로토콜은 미확정(빈칸). 기본 Phase 2 우선 dataset은 각각 MILK/MILK/Chest다.

## 현재 로컬 실행 (프로세스 스냅샷)

```text
 802510  802496       14:55 /home/midori/Projects/ICLR-2027/medical-benchmark/.venv/bin/python3 -m medical_benchmark.runners.train_tcmax --dataset chexchonet --fold 1 --runtime server --max-epochs 10 --batch-size 8 --num-workers 4 --precision fp32
```
