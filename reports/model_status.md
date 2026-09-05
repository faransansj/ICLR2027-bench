# 모델 구현·학습 현황

갱신: 2026-09-06T04:04:57+09:00

사용자 지정 관리 Phase: **1 MambaVision/TransNeXt/TCMax**, **2 IVON/ABNN/NCG/HENN/LATA**, **3 MedRegA/DyMo/DARC/RadZero**.
기존 `configs/phases.yaml`과 `results/phase*`는 과거 실행 provenance이며 변경하지 않았다. 새 관리 Phase는 이 문서/CSV가 기준이다.
빈칸은 실행 없음 또는 미확정이며 0점이 아니다. 타깃은 대화에서 확인된 조합만 기록했다.

## Phase 1

| 모델 | 대상 데이터셋 | 구현·검증 | 학습·평가 | 현재 실행 위치(로컬 직접 확인) | 서버 최신 제공 로그 기준(실시간 아님) | 데이터 확보 |
| --- | --- | --- | --- | --- | --- | --- |
| MambaVision | milk10k | 기존 러너 구현 / 신규 두 데이터셋 GPU smoke 완료(사용자 종료코드 로그) | 로컬 보관 완료 결과 없음 / 과거 mamba_ssm 누락 차단 |  | 연세대 신규: GPU6 fold0 smoke 완료·종료코드0; reference4조합 전체 SMOKE_EXIT=0 사용자 확인 / 새 본학습 미시작 | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 서버 신규 manifests/milk10k.csv 생성 성공 사용자 확인 |
| MambaVision | chexchonet | 기존 러너 구현 / 신규 두 데이터셋 GPU smoke 완료(사용자 종료코드 로그) | 로컬 보관 결과 5/5 완료 |  | 연세대 신규: GPU5 fold0 smoke 완료·종료코드0; reference4조합 전체 SMOKE_EXIT=0 사용자 확인 / 새 본학습 미시작 | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보; 서버 신규 manifests/chexchonet.csv 생성 성공 사용자 확인 |
| TransNeXt | milk10k | 기존 러너 구현 / 신규 두 데이터셋 GPU smoke 완료(사용자 종료코드 로그) | 로컬 보관 결과 5/5 완료 |  | 연세대 신규: GPU6 fold0 smoke 완료·종료코드0; reference4조합 전체 SMOKE_EXIT=0 사용자 확인 / 새 본학습 미시작 | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 서버 신규 manifests/milk10k.csv 생성 성공 사용자 확인 |
| TransNeXt | chexchonet | 기존 러너 구현 / 신규 두 데이터셋 GPU smoke 완료(사용자 종료코드 로그) | 로컬 보관 결과 5/5 완료 |  | 연세대 신규: GPU5 fold0 smoke 완료·종료코드0; reference4조합 전체 SMOKE_EXIT=0 사용자 확인 / 새 본학습 미시작 | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보; 서버 신규 manifests/chexchonet.csv 생성 성공 사용자 확인 |
| TCMax | milk10k | 두 데이터셋 러너 구현·GPU smoke 완료 / CheXchoNET 예비 5-fold·10-epoch 검증 완료 | 로컬 보관 완료 결과 없음 |  | 연세대: 기존 4/5 완료(0–3), fold4 미완료; 기존 스케줄러 종료 사용자 확인; 신규 GPU5 fold0 smoke 3epoch 완료·종료코드0 / 새 본학습 미시작 | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 서버 신규 manifests/milk10k.csv 생성 성공 사용자 확인 |
| TCMax | chexchonet | 두 데이터셋 러너 구현·GPU smoke 완료 / CheXchoNET 예비 5-fold·10-epoch 검증 완료 | 로컬 보관 결과 5/5 완료 |  | 연세대: 기존 2/5 완료(0–1), fold2·3 checkpoint 보존, fold4 미완료; 기존 스케줄러 종료 사용자 확인; 신규 GPU6 fold0 smoke 3epoch 완료·종료코드0 / 새 본학습 미시작 | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보; 서버 신규 manifests/chexchonet.csv 생성 성공 사용자 확인 |

## Phase 2

| 모델 | 대상 데이터셋 | 구현·검증 | 학습·평가 | 현재 실행 위치(로컬 직접 확인) | 서버 최신 제공 로그 기준(실시간 아님) | 데이터 확보 |
| --- | --- | --- | --- | --- | --- | --- |
| IVON | milk10k | 공식 v0.1.2 CPU 실행 검증 / 실제 데이터 학습 러너 미완성 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음 |
| IVON | chexchonet | 공식 v0.1.2 CPU 실행 검증 / 실제 데이터 학습 러너 미완성 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |
| ABNN | milk10k | 공식 구현 조사 / 학습 어댑터 미구현 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음 |
| ABNN | chexchonet | 공식 구현 조사 / 학습 어댑터 미구현 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |
| NCG | milk10k | 공식 구현 조사 / DAG·train-only 어댑터 미준비 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 방법별 추가 입력/아티팩트는 미준비 |
| HENN | milk10k | 공식 구현 조사 / 복합 라벨·어댑터 미준비 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 방법별 추가 입력/아티팩트는 미준비 |
| LATA | chexchonet | 방법 조사 / 공식 아티팩트·calibration 미준비 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보; 방법별 추가 입력/아티팩트는 미준비 |

## Phase 3

| 모델 | 대상 데이터셋 | 구현·검증 | 학습·평가 | 현재 실행 위치(로컬 직접 확인) | 서버 최신 제공 로그 기준(실시간 아님) | 데이터 확보 |
| --- | --- | --- | --- | --- | --- | --- |
| MedRegA | milk10k | 소스 정보 확보 / 환경·어댑터 미준비 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 방법별 추가 입력/아티팩트는 미준비 |
| DyMo | milk10k | 소스 확보 / 데이터셋 전용 학습 경로 미준비 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 방법별 추가 입력/아티팩트는 미준비 |
| DARC | chexchonet | 공식 구현·체크포인트 미확보 | 로컬 보관 완료 결과 없음 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |
| RadZero | chexchonet | zero-shot 평가 러너 구현 | 로컬 보관 결과 5/5 완료 (zero-shot 평가) |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보 |

## Phase-ex — 추가 데이터셋 적용 검토

사용자 요청으로 추가 조합을 Phase-ex에 별도 분류했다. 구현·실험 설정은 아직 미확정이며 성능/실행 위치는 빈칸이다. 기본 Phase 2 대상과 혼합하지 않는다.

| 모델 | 대상 데이터셋 | 구현·검증 | 학습·평가 | 현재 실행 위치(로컬 직접 확인) | 서버 최신 제공 로그 기준(실시간 아님) | 데이터 확보 |
| --- | --- | --- | --- | --- | --- | --- |
| NCG | chexchonet | 추가 적용 검토 / 해당 데이터셋 어댑터 미구현 | 미실행 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보; 서버 신규 manifests/chexchonet.csv 생성 성공 사용자 확인; 추가 조합 프로토콜 미확정 |
| HENN | chexchonet | 추가 적용 검토 / 해당 데이터셋 어댑터 미구현 | 미실행 |  |  | 서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보; 서버 신규 manifests/chexchonet.csv 생성 성공 사용자 확인; 추가 조합 프로토콜 미확정 |
| LATA | milk10k | 추가 적용 검토 / 해당 데이터셋 어댑터 미구현 | 미실행 |  |  | 서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음; 서버 신규 manifests/milk10k.csv 생성 성공 사용자 확인; 추가 조합 프로토콜 미확정 |

## 로컬 TCMax CheXchoNET 예비 검증 근거

RTX 2080, FP32, batch 8, 공식 5개 fold에서 각각 10 epoch를 완료했다. 다섯 run manifest의 `COMPLETED`, 총 50개 연속 epoch 기록, fold별 필수 artifact 6종과 run identity를 검증했으며 학습 오류 로그는 없었다. 공개 결과는 sample 단위 artifact가 아닌 aggregate만 사용하며 **preliminary 5-fold/10-epoch**로 표시한다.

- Macro-F1: 0.038820
- Macro-AUROC: 0.747219
- EMR: 0.862926
- Label F1: SLVH 0.011273, DLV 0.066367
- 4상태 집계는 저장된 `target_0/prediction_0=SLVH`, `target_1/prediction_1=DLV` 이진 결정을 `SLVH + 2*DLV`로 매핑해 계산한다.

## 서버 신규 패키지 검증 근거 (2026-09-06 사용자 제공)

위치: 연세대 L40S 서버 `model-suite-smoke-20260906/`.
배포 archive SHA256: `01029612ef6a522d0ce676ddce6c9881e4db0148922de9b498d11d2c19e782ab`.
독립 환경 설치/두 신규 manifest 생성 성공. TCMax 결과는 본학습 수치가 아닌 bounded smoke다.

| 모델 | Dataset | 근거 | 현재 판정 |
| --- | --- | --- | --- |
| TCMax 적용판 | MILK10k | GPU5, fold0, epoch0–2, COMPLETED, rc=0 | GPU smoke 완료 |
| TCMax 적용판 | CheXchoNET | GPU6, fold0, epoch0–2, COMPLETED, rc=0 | GPU smoke 완료 |
| MambaVision | MILK10k | GPU6, fold0, rc=0 | GPU smoke 완료 |
| MambaVision | CheXchoNET | GPU5, fold0, rc=0 | GPU smoke 완료 |
| TransNeXt | MILK10k | GPU6, fold0, rc=0 | GPU smoke 완료 |
| TransNeXt | CheXchoNET | GPU5, fold0, rc=0 | GPU smoke 완료 |

- 이전 ps에서는 마지막 reference 사전검사가 관측되었고, 이후 사용자가 네 작업 모두 rc=0 및 전체 SMOKE_EXIT=0을 제공했다. 최종 판정은 이 종료코드 증거를 따른다.
- TCMax 로그: `logs/smoke/{milk10k,chexchonet}-fold0.log`. 결과: `outputs/tcmax_resnet18_paper_budget_adaptation/smoke/<dataset>/fold_0`.
- reference 학습 로그: `logs/reference-smoke/`. 최종 네 작업 종료코드0 및 SMOKE_EXIT=0 확인. 서버 원시 결과 파일은 아직 로컬로 동기화하지 않았으며, 상세 수치/별도 결과 검증기 실행은 추가 확인 대상이다.
- 기존 TCMax는 별도 `medical-benchmark/`에서 PGID2111310 종료 완료를 사용자가 확인했다. 종료 전 Chest fold2 last epoch73/best0, fold3 last44/best1 읽기 확인; 파일 삭제 없음.
- 기존 서버 MILK 완료 fold0–3(4/5), Chest 완료 fold0–1(2/5). 중단된 fold2·3 및 미실행 fold4를 완료로 계산하지 않는다.
- 신규 여섯 GPU smoke 모두 통과. 서버 신규 본학습 시작 증거는 없으며 별도 full 실행이 필요하다. smoke 성능을 논문용 5-fold 표에 넣지 않는다.

## 판독 주의

- 서버 상태는 2026-09-06 사용자 제공 종료 확인·smoke 로그·ps 출력 기준이다. 직접 원격 재조회가 아니며 보고서 생성 시간이 서버 확인 시간을 뜻하지 않는다. 로컬 TCMax와 서버 TCMax는 합산하지 않는다.
- `--runtime server`라는 이름만으로 실행 서버를 판단하지 않는다. 현재 로컬 실행은 하단 프로세스 스냅샷을 따른다.
- 수정 패키지 `../model-suite-staging/`: CPU 18개 테스트 통과(기록); 신규 TCMax MILK/Chest GPU smoke 2/2 완료(각3epoch·2배치 제한), MambaVision/TransNeXt 4조합 완료(각1epoch·2배치 제한). 총6조합 사용자 제공 성공 종료코드 확인. 기존 학습 성공이 새 패키지 검증을 대신하지 않는다.
- 기존 이미지 전용 MambaVision/TransNeXt와 멀티모달 TCMax의 입력·예산이 다르다. 논문 조건 동일 재현으로 표시하지 않는다.
- MILK 원본 메타데이터 조사상 결측: 나이 20병변(40행), 부위 31병변(62행); train 중앙값+indicator/MISSING 승인. paired 측정값 사용도 이후 사용자 승인되었으나 최신 staging의 비대칭 차단 구현과는 구분한다(현 데이터 비대칭 0).
- 데이터 수: 기존 MILK manifest 5,240병변, Chest manifest 71,589영상. 이는 파일 목록 행 수이며 모든 이미지의 바이트 무결성 검사를 뜻하지 않는다.
- NCG/HENN의 추가 Chest 적용, LATA의 추가 MILK 적용은 Phase-ex 검토 대상으로 분리했다. 성능과 구체 프로토콜은 미확정(빈칸). 기본 Phase 2 우선 dataset은 각각 MILK/MILK/Chest다.

## 현재 로컬 실행 (프로세스 스냅샷)

```text
학습 프로세스 없음
```
