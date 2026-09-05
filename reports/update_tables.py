"""Refresh aggregate-only reporting. Does not modify configs, jobs, or raw results.
Run: python reports/update_tables.py
"""
from pathlib import Path
import csv
import io
import json
import math
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
LABELS = ('AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC')
ORDER = ('BCC', 'SCCKA', 'MEL', 'AKIEC', 'MAL_OTH', 'NV', 'BKL', 'DF', 'INF', 'VASC', 'BEN_OTH')
HEADERS = ('BCC (48.1%)', 'SCCKA (9.0%)', 'MEL (8.6%)', 'AKIEC (5.8%)', 'MAL OTH (0.2%)', 'NV (14.2%)', 'BKL (10.4%)', 'DF (1.0%)', 'INF (1.0%)', 'VASC (0.9%)', 'BEN OTH (0.8%)')
PHASES = {1: ('MambaVision', 'TransNeXt', 'TCMax'), 2: ('IVON', 'ABNN', 'NCG', 'HENN', 'LATA'), 3: ('MedRegA', 'DyMo', 'DARC', 'RadZero')}
TARGETS = {m: ('milk10k', 'chexchonet') for m in ('MambaVision', 'TransNeXt', 'TCMax', 'IVON', 'ABNN')}
TARGETS.update({m: ('milk10k',) for m in ('NCG', 'HENN', 'MedRegA', 'DyMo')})
TARGETS.update({m: ('chexchonet',) for m in ('LATA', 'DARC', 'RadZero')})
EXTRAS = (('NCG', 'chexchonet'), ('HENN', 'chexchonet'), ('LATA', 'milk10k'))
# Manually verified readiness, not inferred from a config named READY.
IMPLEMENTATION = {
    'MambaVision': '기존 러너 구현 / 신규 두 데이터셋 GPU smoke 완료(사용자 종료코드 로그)',
    'TransNeXt': '기존 러너 구현 / 신규 두 데이터셋 GPU smoke 완료(사용자 종료코드 로그)',
    'TCMax': '두 데이터셋 러너 구현·GPU smoke 완료 / CheXchoNET 예비 5-fold·10-epoch 검증 완료',
    'IVON': '공식 v0.1.2 CPU 실행 검증 / 실제 데이터 학습 러너 미완성',
    'ABNN': '공식 구현 조사 / 학습 어댑터 미구현',
    'NCG': '공식 구현 조사 / DAG·train-only 어댑터 미준비',
    'HENN': '공식 구현 조사 / 복합 라벨·어댑터 미준비',
    'LATA': '방법 조사 / 공식 아티팩트·calibration 미준비',
    'MedRegA': '소스 정보 확보 / 환경·어댑터 미준비',
    'DyMo': '소스 확보 / 데이터셋 전용 학습 경로 미준비',
    'DARC': '공식 구현·체크포인트 미확보',
    'RadZero': 'zero-shot 평가 러너 구현',
}
# Transcribed from the user's MILK table, not locally reproduced experiment outputs.
PAPER = '''MLR-GCN,0.74±0.02,0.16±0.03,0.21±0.02,0.00±0.02,0.00±0.00,0.59±0.05,0.00±0.00,0.00±0.00,0.00±0.00,0.00±0.00,0.00±0.00,0.16
MVFA-AD,0.88±0.13,0.66±0.13,0.58±0.15,0.46±0.08,0.00±0.00,0.78±0.17,0.45±0.05,0.33±0.12,0.27±0.06,0.78±0.04,0.36±0.11,0.50
PanDerm,0.87±0.02,0.64±0.06,0.67±0.07,0.53±0.08,0.00±0.00,0.75±0.02,0.51±0.05,0.63±0.11,0.00±0.00,0.89±0.08,0.43±0.12,0.53
JI-ADF,0.89±0.01,0.69±0.03,0.61±0.03,0.53±0.02,0.00±0.00,0.78±0.02,0.52±0.05,0.65±0.06,0.27±0.05,0.72±0.14,0.26±0.19,0.54
TMCEK,0.81±0.01,0.58±0.02,0.48±0.06,0.18±0.04,0.00±0.00,0.75±0.01,0.23±0.08,0.04±0.08,0.04±0.08,0.37±0.25,0.22±0.15,0.34
I²MoE,0.83±0.01,0.61±0.06,0.52±0.06,0.28±0.08,0.00±0.00,0.75±0.01,0.36±0.07,0.26±0.26,0.08±0.12,0.46±0.26,0.14±0.13,0.39
SMV,0.82±0.01,0.55±0.04,0.51±0.09,0.22±0.08,0.00±0.00,0.74±0.02,0.29±0.07,0.23±0.16,0.23±0.11,0.53±0.14,0.21±0.16,0.39
CMoB,0.86±0.02,0.65±0.03,0.61±0.04,0.47±0.03,0.22±0.05,0.76±0.03,0.47±0.04,0.59±0.03,0.31±0.04,0.64±0.03,0.29±0.05,0.53
Ours,0.89±0.01,0.69±0.03,0.62±0.03,0.52±0.06,0.00±0.00,0.78±0.01,0.54±0.00,0.53±0.03,0.29±0.06,0.81±0.01,0.37±0.03,0.55'''


def mean(values):
    assert values and all(math.isfinite(x) and 0 <= x <= 1 for x in values)
    return f'{statistics.mean(values):.2f}'


def spread(values):
    return mean(values) + '±' + f'{statistics.stdev(values):.2f}' if len(values) > 1 else ''


def records(dataset, model):
    phase = 'phase2' if model in ('TCMax', 'RadZero') else 'phase1'
    parent = ROOT / 'results' / phase / dataset / model.lower()
    result = []
    for fold in range(5):
        directory = parent / f'fold_{fold}'
        file = directory / 'fold_metrics.json'
        manifest = directory / 'run_manifest.json'
        if not file.exists() or not manifest.exists():
            continue
        metrics = json.loads(file.read_text())
        run = json.loads(manifest.read_text())
        if metrics.get('status', '').lower() != 'completed' or run.get('status', '').lower() != 'completed':
            continue
        assert metrics['run_id'] == run['run_id'], directory
        result.append((directory, metrics['test'], run))
    return result


def state_scores(path):
    # Derive four-state F1 from the same stored binary decisions used for EMR.
    tp, actual, predicted = [0]*4, [0]*4, [0]*4
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            bits = [int(row[k]) for k in ('target_0', 'target_1', 'prediction_0', 'prediction_1')]
            assert all(x in (0, 1) for x in bits)
            a, b = bits[0] + 2*bits[1], bits[2] + 2*bits[3]
            actual[a] += 1
            predicted[b] += 1
            tp[a] += int(a == b)
    assert sum(actual)
    return [2*tp[i]/(actual[i]+predicted[i]) if actual[i]+predicted[i] else 0.0 for i in range(4)]


def csv_write(name, header, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---']*len(header)) + ' |'] +
                     ['| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |' for row in rows])


def self_check():
    assert ORDER[0] == LABELS[1] == 'BCC'
    assert spread([0.0, 1.0]) == '0.50±0.71'
    assert spread([0.0]) == ''  # A single fold cannot supply sample std.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'predictions.csv'
        path.write_text('target_0,target_1,prediction_0,prediction_1\n'
                        '0,0,0,0\n1,0,0,0\n0,1,0,1\n1,1,0,1\n')
        assert state_scores(path) == [2/3, 0.0, 2/3, 0.0]


def main():
    self_check()
    now = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    ps = subprocess.check_output(['ps', '-ww', '-eo', 'pid,ppid,etime,args'], text=True)
    live = [line for line in ps.splitlines() if '/.venv/bin/python' in line and '-m medical_benchmark.runners.' in line]
    pids = {line.split()[0] for line in live}
    live = [line for line in live if line.split()[1] not in pids]  # Exclude DataLoader children.
    complete = {(ds, m): records(ds, m) for m, dslist in TARGETS.items() for ds in dslist}
    status_rows = []
    for phase, models in PHASES.items():
        for model in models:
            for ds in TARGETS[model]:
                n = len(complete[ds, model])
                active = [line for line in live if f'--dataset {ds}' in line and
                          (f'--model {model.lower()}' in line or (model == 'TCMax' and 'runners.train_tcmax ' in line))]
                training = f'로컬 보관 결과 {n}/5 완료' if n else '로컬 보관 완료 결과 없음'
                if model == 'RadZero' and n: training += ' (zero-shot 평가)'
                if active: training += ' / 로컬 실행 중'
                if model == 'MambaVision' and ds == 'milk10k': training += ' / 과거 mamba_ssm 누락 차단'
                server = '로컬 NixOS / RTX 2080' if active else ''
                remote = ''
                if model == 'TCMax':
                    legacy = '기존 4/5 완료(0–3), fold4 미완료' if ds == 'milk10k' else '기존 2/5 완료(0–1), fold2·3 checkpoint 보존, fold4 미완료'
                    gpu = '5' if ds == 'milk10k' else '6'
                    remote = f'연세대: {legacy}; 기존 스케줄러 종료 사용자 확인; 신규 GPU{gpu} fold0 smoke 3epoch 완료·종료코드0 / 새 본학습 미시작'
                elif model in ('MambaVision', 'TransNeXt'):
                    gpu = '6' if ds == 'milk10k' else '5'
                    remote = f'연세대 신규: GPU{gpu} fold0 smoke 완료·종료코드0; reference4조합 전체 SMOKE_EXIT=0 사용자 확인 / 새 본학습 미시작'
                data = '서버 원본·기존 manifest 확보(사용자 확인); 로컬 manifest 확보, 이미지 폴더 비어 있음' if ds == 'milk10k' else '서버 원본·manifest 확보(사용자 확인); 로컬 원본 링크·manifest 확보'
                if model in PHASES[1]:
                    data += '; 서버 신규 manifests/' + ds + '.csv 생성 성공 사용자 확인'
                if model in ('NCG', 'HENN', 'LATA', 'DyMo', 'MedRegA'):
                    data += '; 방법별 추가 입력/아티팩트는 미준비'
                status_rows.append([str(phase), model, ds, IMPLEMENTATION[model], training, server, remote, data])
    for model, ds in EXTRAS:
        data = next(row[7] for row in status_rows if row[2] == ds)
        status_rows.append(['ex', model, ds, '추가 적용 검토 / 해당 데이터셋 어댑터 미구현',
                            '미실행', '', '', data + '; 추가 조합 프로토콜 미확정'])
    header = ['Phase', '모델', '대상 데이터셋', '구현·검증', '학습·평가', '현재 실행 위치(로컬 직접 확인)', '서버 최신 제공 로그 기준(실시간 아님)', '데이터 확보']
    csv_write('model_status.csv', header, status_rows)
    status_text = f'''# 모델 구현·학습 현황\n\n갱신: {now}\n\n사용자 지정 관리 Phase: **1 MambaVision/TransNeXt/TCMax**, **2 IVON/ABNN/NCG/HENN/LATA**, **3 MedRegA/DyMo/DARC/RadZero**.\n기존 `configs/phases.yaml`과 `results/phase*`는 과거 실행 provenance이며 변경하지 않았다. 새 관리 Phase는 이 문서/CSV가 기준이다.\n빈칸은 실행 없음 또는 미확정이며 0점이 아니다. 타깃은 대화에서 확인된 조합만 기록했다.\n\n'''
    for phase in PHASES:
        status_text += f'## Phase {phase}\n\n' + table(header[1:], [r[1:] for r in status_rows if r[0] == str(phase)]) + '\n\n'
    status_text += '## Phase-ex — 추가 데이터셋 적용 검토\n\n사용자 요청으로 추가 조합을 Phase-ex에 별도 분류했다. 구현·실험 설정은 아직 미확정이며 성능/실행 위치는 빈칸이다. 기본 Phase 2 대상과 혼합하지 않는다.\n\n'
    status_text += table(header[1:], [r[1:] for r in status_rows if r[0] == 'ex']) + '\n\n'
    status_text += '''## 로컬 TCMax CheXchoNET 예비 검증 근거\n\nRTX 2080, FP32, batch 8, 공식 5개 fold에서 각각 10 epoch를 완료했다. 다섯 run manifest의 `COMPLETED`, 총 50개 연속 epoch 기록, fold별 필수 artifact 6종과 run identity를 검증했으며 학습 오류 로그는 없었다. 공개 결과는 sample 단위 artifact가 아닌 aggregate만 사용하며 **preliminary 5-fold/10-epoch**로 표시한다.\n\n- Macro-F1: 0.038820\n- Macro-AUROC: 0.747219\n- EMR: 0.862926\n- Label F1: SLVH 0.011273, DLV 0.066367\n- 4상태 집계는 저장된 `target_0/prediction_0=SLVH`, `target_1/prediction_1=DLV` 이진 결정을 `SLVH + 2*DLV`로 매핑해 계산한다.\n\n## 서버 신규 패키지 검증 근거 (2026-09-06 사용자 제공)\n\n위치: 연세대 L40S 서버 `model-suite-smoke-20260906/`.\n배포 archive SHA256: `01029612ef6a522d0ce676ddce6c9881e4db0148922de9b498d11d2c19e782ab`.\n독립 환경 설치/두 신규 manifest 생성 성공. TCMax 결과는 본학습 수치가 아닌 bounded smoke다.\n\n| 모델 | Dataset | 근거 | 현재 판정 |\n| --- | --- | --- | --- |\n| TCMax 적용판 | MILK10k | GPU5, fold0, epoch0–2, COMPLETED, rc=0 | GPU smoke 완료 |\n| TCMax 적용판 | CheXchoNET | GPU6, fold0, epoch0–2, COMPLETED, rc=0 | GPU smoke 완료 |\n| MambaVision | MILK10k | GPU6, fold0, rc=0 | GPU smoke 완료 |\n| MambaVision | CheXchoNET | GPU5, fold0, rc=0 | GPU smoke 완료 |\n| TransNeXt | MILK10k | GPU6, fold0, rc=0 | GPU smoke 완료 |\n| TransNeXt | CheXchoNET | GPU5, fold0, rc=0 | GPU smoke 완료 |\n\n- 이전 ps에서는 마지막 reference 사전검사가 관측되었고, 이후 사용자가 네 작업 모두 rc=0 및 전체 SMOKE_EXIT=0을 제공했다. 최종 판정은 이 종료코드 증거를 따른다.\n- TCMax 로그: `logs/smoke/{milk10k,chexchonet}-fold0.log`. 결과: `outputs/tcmax_resnet18_paper_budget_adaptation/smoke/<dataset>/fold_0`.\n- reference 학습 로그: `logs/reference-smoke/`. 최종 네 작업 종료코드0 및 SMOKE_EXIT=0 확인. 서버 원시 결과 파일은 아직 로컬로 동기화하지 않았으며, 상세 수치/별도 결과 검증기 실행은 추가 확인 대상이다.\n- 기존 TCMax는 별도 `medical-benchmark/`에서 PGID2111310 종료 완료를 사용자가 확인했다. 종료 전 Chest fold2 last epoch73/best0, fold3 last44/best1 읽기 확인; 파일 삭제 없음.\n- 기존 서버 MILK 완료 fold0–3(4/5), Chest 완료 fold0–1(2/5). 중단된 fold2·3 및 미실행 fold4를 완료로 계산하지 않는다.\n- 신규 여섯 GPU smoke 모두 통과. 서버 신규 본학습 시작 증거는 없으며 별도 full 실행이 필요하다. smoke 성능을 논문용 5-fold 표에 넣지 않는다.\n\n'''
    status_text += '''## 판독 주의\n\n- 서버 상태는 2026-09-06 사용자 제공 종료 확인·smoke 로그·ps 출력 기준이다. 직접 원격 재조회가 아니며 보고서 생성 시간이 서버 확인 시간을 뜻하지 않는다. 로컬 TCMax와 서버 TCMax는 합산하지 않는다.\n- `--runtime server`라는 이름만으로 실행 서버를 판단하지 않는다. 현재 로컬 실행은 하단 프로세스 스냅샷을 따른다.\n- 수정 패키지 `../model-suite-staging/`: CPU 18개 테스트 통과(기록); 신규 TCMax MILK/Chest GPU smoke 2/2 완료(각3epoch·2배치 제한), MambaVision/TransNeXt 4조합 완료(각1epoch·2배치 제한). 총6조합 사용자 제공 성공 종료코드 확인. 기존 학습 성공이 새 패키지 검증을 대신하지 않는다.\n- 기존 이미지 전용 MambaVision/TransNeXt와 멀티모달 TCMax의 입력·예산이 다르다. 논문 조건 동일 재현으로 표시하지 않는다.\n- MILK 원본 메타데이터 조사상 결측: 나이 20병변(40행), 부위 31병변(62행); train 중앙값+indicator/MISSING 승인. paired 측정값 사용도 이후 사용자 승인되었으나 최신 staging의 비대칭 차단 구현과는 구분한다(현 데이터 비대칭 0).\n- 데이터 수: 기존 MILK manifest 5,240병변, Chest manifest 71,589영상. 이는 파일 목록 행 수이며 모든 이미지의 바이트 무결성 검사를 뜻하지 않는다.\n- NCG/HENN의 추가 Chest 적용, LATA의 추가 MILK 적용은 Phase-ex 검토 대상으로 분리했다. 성능과 구체 프로토콜은 미확정(빈칸). 기본 Phase 2 우선 dataset은 각각 MILK/MILK/Chest다.\n\n## 현재 로컬 실행 (프로세스 스냅샷)\n\n'''
    # Only operational arguments, no private samples or credentials.
    status_text += '```text\n' + ('\n'.join(live) or '학습 프로세스 없음') + '\n```\n'
    (OUT / 'model_status.md').write_text(status_text)

    paper = {r[0]: r[1:] for r in csv.reader(io.StringIO(PAPER))}
    assert len(paper) == 9 and all(len(v) == 12 for v in paper.values())
    milk_order = ['MLR-GCN', 'MVFA-AD', 'MambaVision', 'TransNeXt', 'PanDerm', 'JI-ADF', 'TMCEK', 'I²MoE', 'SMV', 'CMoB', 'IVON', 'ABNN', 'NCG', 'HENN', 'MedRegA', 'TCMax', 'DyMo', 'LATA [Phase-ex]', 'Ours']
    milk = []
    for model in milk_order:
        runs = complete.get(('milk10k', model), [])
        if model in paper:
            values, origin, count = paper[model], '사용자 제공 논문 표 (재실행 아님)', ''
        elif len(runs) == 5:
            values = [spread([r[1]['class_f1'][str(LABELS.index(label))] for r in runs]) for label in ORDER]
            values += [mean([r[1]['macro_f1'] for r in runs])]
            origin, count = '기존 조건 완료 결과 / 논문 조건 미정렬', '5/5'
        else:
            values, origin, count = ['']*12, '미완료 / 숫자 미기입', f'{len(runs)}/5'
        if model == 'LATA [Phase-ex]': origin = 'Phase-ex 추가 적용 검토 / 미실행'
        milk.append([model] + values + [origin, count])
    mh = ['Method', *HEADERS, 'Macro-F1', '출처·조건', '완료 folds']
    csv_write('milk10k_benchmark.csv', mh, milk)

    chest = []
    for model in ['MambaVision', 'TransNeXt', 'TCMax', 'IVON', 'ABNN', 'LATA', 'DARC', 'RadZero', 'NCG [Phase-ex]', 'HENN [Phase-ex]', 'Ours']:
        runs = complete.get(('chexchonet', model), [])
        if model == 'Ours':
            values = ['0.58±0.01', '0.60±0.01', '0.59', '0.92±0.01', '0.22±0.02', '0.24±0.03', '0.16±0.04', '0.87']
            origin, count = '제공 PDF Table 2 (재실행 아님)', ''
        elif len(runs) == 5:
            states = [state_scores(r[0] / 'predictions.csv') for r in runs]
            values = [spread([r[1]['label_f1'][label] for r in runs]) for label in ('SLVH', 'DLV')]
            values += [mean([r[1]['macro_f1'] for r in runs])]
            values += [spread([s[i] for s in states]) for i in range(4)]
            values += [mean([r[1]['emr'] for r in runs])]
            condition = 'preliminary 5-fold/10-epoch' if model == 'TCMax' else '기존 결과'
            origin, count = f'{condition} / 4상태 F1은 저장 binary prediction에서 재계산', '5/5'
        else:
            values, origin, count = ['']*8, '미완료 / 숫자 미기입', f'{len(runs)}/5'
        if '[Phase-ex]' in model: origin = 'Phase-ex 추가 적용 검토 / 미실행'
        chest.append([model] + values + [origin, count])
    ch = ['Method', 'SLVH F1', 'DLV F1', 'Avg F1', 'Neither F1', 'SLVH only F1', 'DLV only F1', 'Both F1', 'EMR', '출처·조건', '완료 folds']
    csv_write('chexchonet_benchmark.csv', ch, chest)
    text = f'''# 논문 형식 벤치마크 표\n\n갱신: {now}\n\n- 완료된 **5개 fold**만 평균±표본 표준편차(ddof=1), 소수점 2자리로 집계한다. Macro/Avg-F1·EMR은 fold 평균이다. 빈칸은 미확정/미완료; `0.00`은 실제 반올림 값이다.\n- 사용자 제공 논문 수치는 그대로 전사하며 원시 예측으로 검증한 값이 아니다. 논문 표의 std 정의는 별도 확인 필요.\n- 기존 조건 결과와 논문 행을 함께 표시한 **진행 관리용 표**이며 동일 입력·예산의 최종 비교표가 아니다.\n- 부분 fold 중간 결과, smoke, 서버 미동기화 결과는 최종 칸에 채우지 않는다. 서버 기존 TCMax는 MILK4/5·Chest2/5 확인 후 중단; 수정 TCMax의 2개 GPU smoke 완료는 본학습 결과가 아니다. 최신 검증 상태는 model_status.md를 따른다.\n- 로컬 TCMax CheXchoNET 완료 행은 **preliminary 5-fold/10-epoch**이며 최종 논문 예산 결과가 아니다.\n- MILK class index는 기존 LABELS 순서(AKIEC,BCC,BEN_OTH,BKL,DF,INF,MAL_OTH,MEL,NV,SCCKA,VASC)에서 논문 열 순서로 재배열했다.\n\n## MILK10k\n\n'''
    text += table(mh, milk) + '\n\n## CheXchoNET\n\n' + table(ch, chest)
    text += '\n\nChest 4상태 F1은 `target_0/prediction_0=SLVH`, `target_1/prediction_1=DLV`로 저장된 예측을 사용한다. 별도 categorical argmax로 바꾸지 않는다. 원래 fold_metrics의 state_f1가 빈 경우에도 예측 CSV에서 계산하며 원본은 수정하지 않는다.\n\n## 중간 결과 (최종 표에 미반영)\n\n'
    partial = []
    for (ds, model), runs in complete.items():
        if 0 < len(runs) < 5:
            for path, metrics, _ in runs:
                partial.append([ds, model, path.name, f'{metrics["macro_f1"]:.6f}', str(path.relative_to(ROOT))])
    text += table(['Dataset', 'Model', 'Fold', 'Macro-F1', '근거 경로'], partial) if partial else '부분 완료 결과 없음'
    text += '\n\n## 근거 및 갱신\n\n`results/phase1/{dataset}/{model}/fold_*/{fold_metrics,run_manifest}.json`, `results/phase2/...`, Chest `predictions.csv`. 새 관리 Phase와 과거 저장 Phase는 다르다. 서버 결과는 전달·검증 후 별도 실행 identity로 추가해야 한다.\n\n```bash\npython reports/update_tables.py\n```\n\n이 스크립트는 reports 출력만 갱신하고 configs/학습/데이터/원시 결과를 변경하지 않는다. 구현·서버 마지막 상태 문구는 수동 확인 기록이므로 GPU 검증/서버 갱신 시 함께 수정해야 한다.\n'
    (OUT / 'benchmark_tables.md').write_text(text)
    assert len(status_rows) == 20
    assert sum(row[0] == 'ex' for row in status_rows) == 3
    assert set(PHASES[1]) == {'MambaVision', 'TransNeXt', 'TCMax'}
    print(f'Updated 5 report files; {len(status_rows)} dataset/model rows, {len(milk)} MILK rows, {len(chest)} Chest rows. {now}')


if __name__ == '__main__':
    main()
