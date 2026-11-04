# LC-RankUp status

LC-RankUp was a NO-GO training-side experiment. It is not required for the retained successful RankUp reproduction, RankUp-3D, SSR-RCUS-3D, supervised baselines, or active M1. Its branch history, run configurations, metrics, and training histories are archived here.

The `lcrankup_c0_s0/model_best.pth` checkpoint was mistakenly deleted on 2026-11-03 when a deletion script followed a symlink. The original SHA-256 was `3275bd33f1ba95dada24b047e4cb84d91a49005a0d442381e2d987495cd8e88f`. No byte-identical copy was found on the host; the symlink in `lcrankup_m1_s0` is broken. `archive/INCIDENT_REPORT_20261103.md` records the details and a GPU rerun recipe. We have not claimed exact byte recovery or retrained this excluded baseline.
