# Analysis Chain Automation — Pipeline Plan

## Overview

tamsa2와 private-snu에 걸쳐 있는 ChargedHiggs 분석 체인을 자동화.
Claude Agent SDK 기반 오케스트레이터가 각 단계를 실행하고, 이상 시 Telegram으로 보고 후 대기.

---

## Infrastructure

```
tamsa2 (/data9/.../Sync/)          private-snu (/home/choij/Sync/)
├── SKNanoAnalyzer/                 └── (same files via Syncthing)
├── SKNanoOutput_V0/  ─────────────────────────────────────────▶ 공유
└── ChargedHiggsAnalysisV3/         └── ChargedHiggsAnalysisV3/
     (SF/validation 실행)                (SF/validation 실행)

Orchestrator: tamsa2 cron
  → condor submit (tamsa2 로컬)
  → file sync check (Syncthing 대기)
  → validation (private-snu SSH)
  → Telegram 알림
```

**Telegram:** `config.tamsa2`의 기존 bot token / chatId 재사용

---

## Pipeline Stages & Checkpoints

```
[STAGE 1] SUBMIT
  ↓ tamsa2에서 SKNano.py로 condor job 제출
  ↓ checkpoint: job IDs 저장

[STAGE 2] CONDOR_WAIT                            ← condor_monitor 재사용
  ↓ condor_q polling (매 N분)
  ↓ checkpoint: 모든 잡 Done
  ✗ Held/Failed 잡 → Telegram 보고 + PAUSE

[STAGE 3] OUTPUT_CHECK                           ← tamsa2
  ↓ 예상 출력 파일 목록 vs 실제 파일 비교
  ↓ 파일 크기 > 0 확인
  ↓ ROOT 파일 open 테스트 (python -c "import ROOT; f=ROOT.TFile(...)")
  ↓ checkpoint: 모든 파일 유효
  ✗ 누락/빈 파일 → Telegram 보고 + PAUSE

[STAGE 4] SYNC_WAIT                              ← tamsa2 → private-snu
  ↓ private-snu에 파일 도착 polling (ssh ls)
  ↓ checkpoint: 파일 수/크기 일치
  (Syncthing lag, 보통 < 1분)

[STAGE 5] VALIDATION                             ← private-snu
  ↓ source setup.sh && python validate.py
  ↓ Agent가 출력/플롯 sanity check
     - histogram 비어있지 않음
     - Z mass peak 위치 확인 (if applicable)
     - Data/MC ratio 1±0.5 범위
  ↓ checkpoint: validation passed
  ✗ 이상 감지 → Telegram 보고 + PAUSE (사람 확인 대기)

[STAGE 6] ANALYSIS                               ← private-snu
  ↓ SF measurement / signal region study 실행
  ↓ 출력 파일 생성
  ↓ checkpoint: 완료
  → Telegram 완료 알림
```

---

## Checkpoint 파일 형식

`pipeline_state.json` (tamsa2 `condor-monitor/` 에 저장):

```json
{
  "run_id": "2026-02-21_MeasFakeRateV4_2022",
  "stage": "VALIDATION",
  "status": "PAUSED",
  "paused_reason": "Z mass peak at 87.3 GeV (expected 91.2)",
  "condor_cluster": "12345",
  "expected_files": ["2022/DYJets.root", "2022/TTTo2L2Nu.root"],
  "started_at": "2026-02-21T10:00:00Z",
  "updated_at": "2026-02-21T12:34:00Z"
}
```

Resume: `node pipeline.mjs --resume` (checkpoint에서 재시작)

---

## File Structure

```
condor-monitor/
  ├── condor_monitor.mjs      ← 이미 완성 (Phase 1)
  ├── pipeline.mjs            ← 오케스트레이터 (신규)
  ├── pipeline_state.json     ← 런타임 상태
  ├── config.json             ← Telegram token, 경로, 임계값
  ├── PLAN.md                 ← condor monitor 계획
  ├── PIPELINE_PLAN.md        ← 이 파일
  └── logs/
```

---

## Allowed Commands (canUseTool whitelist)

```
tamsa2:
  condor_q, condor_history, condor_status
  SKNano.py (submit)
  python -c "import ROOT..." (file check)
  ls, wc -l, du -sh (file system)

private-snu (via SSH):
  ssh private-snu "source setup.sh && python ..."
  ls, cat (output 확인)
  python validate.py (validation)
```

---

## Phases

### Phase 1 — 현재 (condor monitor)
- [x] condor_monitor.mjs 초안

### Phase 2 — Pipeline 오케스트레이터
- [ ] `pipeline.mjs` 작성 (tamsa2)
- [ ] `validate.py` 작성 (private-snu, 모듈별 sanity check)
- [ ] checkpoint 저장/복원 로직
- [ ] Telegram PAUSE/RESUME 인터페이스

### Phase 3 — 자동화 강화
- [ ] 모듈별 파이프라인 설정 (MeasFakeRateV4, TriLepton, etc.)
- [ ] 실패 패턴 분류 (메모리 부족, ROOT 파일 corrupt 등)
- [ ] 일일 진행 상황 리포트

---

## Open Questions

1. 어떤 모듈부터 자동화? (MeasFakeRateV4 우선?)
2. validation 기준 — 모듈마다 다름, 초기엔 파일 존재 + 비어있지 않음만?
3. PAUSE 후 resume 방법 — Telegram 명령어? (`/resume pipeline`)
4. private-snu에서 output 저장 경로?
