# Evidence Index

This index maps SİNAPTİC5G engineering claims to repository artifacts.

## FTR acceptance and reproducibility

- `../../5G PROJE/scripts/ftr_pre_submission_check.py` — model-lock SHA-256, schema, label contract, static path, Dockerfile and dependency checks.
- `../../5G PROJE/scripts/ftr_docker_acceptance.py` — Docker-oriented acceptance tooling.
- `../../Dockerfile` — official offline FTR container surface.
- `../../5G PROJE/model_lock.json` — model-artifact integrity contract.

## Dataset and evaluation evidence

- `../../5G PROJE/reports/dataset_audit.md` — dataset audit.
- `../../5G PROJE/reports/ftr_performance_profile.md` — FTR performance profile.
- `../../5G PROJE/reports/ai_solution_methods.md` — AI method/evidence summary.
- `../../5G PROJE/reports/mathematical_foundations.md` — physical/mathematical reasoning notes.

## Known blockers and review evidence

- `../../5G PROJE/reports/teknocan_blocker_report.md` — controlled Teknocan-data blocker.
- `../../5G PROJE/reports/open_questions_for_team.md` — unresolved team questions.
- `../../deliverables/SINAPTIC5G_JURI_TEKNIK_RAPORU.md` — jury-facing technical report.

## Interpretation rules

1. Dataset counts and label mappings are tied to the documented FTR contract.
2. Performance/latency values are hardware and scenario dependent.
3. Locked model hashes are part of reproducibility evidence.
4. A blocker that prevents unsafe data generation is a reliability feature, not missing evidence to be silently bypassed.
5. Offline FTR and live 5G operation are separate evaluation surfaces.

See `../../KNOWN_LIMITATIONS.md` for scope boundaries.
