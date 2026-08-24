# Known Limitations

SİNAPTİC5G is a competition-oriented Edge AI / 5G road-safety system. The following limitations are explicit engineering boundaries.

## 1. Teknocan data blocker

The repository documents a controlled blocker for Teknocan synthetic-data generation: approved transparent foreground assets are not available in the required source directory. The pipeline intentionally refuses to synthesize semantically unsafe copy-paste data from unsuitable images. See `5G PROJE/reports/teknocan_blocker_report.md`.

## 2. Physical calibration dependency

BEV projection and speed estimation depend on valid camera/homography calibration and reliable timestamps. A correct algorithm cannot compensate for an invalid physical calibration contract.

## 3. Detection confidence varies with scene conditions

Low light, glare, motion blur, compression, occlusion and small-object scale can reduce perception quality. Augmentation, image conditioning and adaptive filtering reduce risk but do not make these conditions universally solved.

## 4. Competition label contract

The canonical label mapping is tied to the FTR/competition contract. Changing the target schema or deployment task requires explicit remapping and revalidation; the existing 9-label dataset contract should not be treated as a universal road-scene ontology.

## 5. Real-time performance is hardware-dependent

Latency and throughput depend on GPU, video resolution, model artifacts, preprocessing and runtime configuration. A repository-level real-time claim is not a guarantee for arbitrary hardware.

## 6. 5G/live mode depends on external infrastructure

Network-assisted/live extensions depend on device, network and service availability. The offline FTR Docker path remains a distinct evaluation surface and should not be conflated with live 5G behavior.

## 7. Model artifacts are part of the release contract

The FTR Docker image expects locked ONNX artifacts and validates their SHA-256 values. Replacing weights without updating and re-running the model-lock/evidence process invalidates the release claim.

## Evidence policy

Use the FTR acceptance scripts, dataset audit, performance profile, blocker reports and jury/final reports as evidence. See `docs/evidence/README.md`.
