# Independent validity review (2026-11-04)

This note supersedes claims that all configurations in E0_STAR_FROZEN.md and CONFIG_AUDIT_20261104.md have been validated.

1. The 1600-step MLP OOF predictions exist for all 3 seeds × 5 folds; direct recomputation gives Macro-MAE 0.1764117. This is a measured score, not proof that the E0* selection is final.
2. The Ridge arm in `audit_e0_assemble.py` and `audit_components_oof.py` is invalid: `train_oof_fold.py` saves `features_labeled.npy` before JACF training but `features_holdout.npy` after JACF training. The 0.18456 Ridge OOF score uses mismatched encoder states. Recompute Ridge with a holdout feature array from the same pre-JACF encoder before comparing candidates.
3. For the earlier JACF/ACLDP OOF folds, `train_oof_fold.py` calls `set_seed(a.fold)` and ignores `data_seed` for initialization. The 3 label splits are correct and disjoint within each fold, but the 3 seeds do not have independent initializations as specified. The strong supervised recipe does use `1000*data_seed+fold`; rerun paired arms with independent base checkpoints.
4. Old M0 and the 1600-step E0* have the same model structure and L1 loss, but also differ in learning rate (2e-5/5e-6 versus 1e-4), batch size (16 versus 64), and training duration. “Only training budget changed” is inaccurate. The encoder is fine-tuned by `AdamW(model.parameters())`; it is not frozen.
5. The 1e-4 learning rate was previously explored on dev. The old 0.0975 OOF score from an encoder fine-tuned on all 500 labels is invalid due to label leakage. The old dev results remain exploratory because dev guided configuration and checkpoint selection.
6. No 1600-step base checkpoint was saved for the 45 supervised OOF runs (`save_ckpt=false` in configs). Predictions and configurations are retained, but a deployable checkpoint requires retraining. The strong-base M1 grid has produced zero results because CUDA is unavailable.

Required before final M1 admission: restore GPU, generate consistent pre-JACF Ridge features, rerun seed-distinct paired controls, and report raw versus final metrics under the pre-registered gate. Preserve all original artifacts as historical evidence.
