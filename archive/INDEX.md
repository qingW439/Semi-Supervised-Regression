# Archived research branches

This branch keeps the original Git commits reachable after the active branch list is reduced. Each source ref is merged using Git's `ours` strategy; inspect an archived tree with `git show <SHA>:<path>` or restore it with `git branch <name> <SHA>`. Binary checkpoints remain on the experiment host and are listed separately in the asset manifests.

## Former remote branches

| Branch | Original tip |
|---|---|
| `dev` | `bacd87ab44bd3bb242d57edcd2d0a4fccdcf1e6c` |
| `emobank-ssr-rcus-3d-adapt` | `0803a7e87a7cc4828256588561766f4ec833cd48` |
| `emobank-vad` | `1125b7841bda62aae485f0e9ce462a8b5739814c` |
| `m1/acvr-vad` | `376baf22183676d08a8b63d47423cf0949641624` |
| `m1/cpr-rankup-vad` | `7406d4e4cbebb5db3b51701312a76b4b67a41410` |
| `m1/jacf-ablation` | `218bdb8f5b4e44ab5d90c4b4d769f9b52a9c0975` |
| `m1/jacf-grad` | `f907bb8a30d7a11c6b50dbb3735c5642fa237db5` |
| `m1/jacf-relcal` | `ba7709a968ea7e9d647c84dac55ad1d525f60888` |
| `m1/jacf-source` | `535955e6abb521823b31fe8190da27985c29ca75` |
| `m1/lc-rankup-vad` | `869c41ff59f759591ceef23b45835135305ca960` |
| `m1/reliability-study` | `f13e7895eb3238eddf866f35d723fb0e1d1c13db` |
| `rankup` | `8f11c4d7b633cd310cad0321516d5103723ccf5d` |
| `vad-m0-cpu-prep` | `998c900329723660411213e86ea846a4be12a203` |
| `vad-m0-simregmatch-3d` | `c01a992d91892dfdff80c18d3060aef04494fdde` |
| `vad-module-swap` | `b364ed97352c9547ed15fec3007f2770d3fe8801` |
| `vad-reliable-ssr` | `d2b5959401f5419cf5f77d26017f06d3d4601973` |

## Host-only branches

| Branch | Original tip |
|---|---|
| `m1/comp2` | `8dda24bf828ba9848da33583851bffaa68fa500b` |
| `vad-lact-mechanism-audit` | `4be1cacf225b3531482245c15855d50202fe7085` |
| `vad-m0-spectral-readiness` | `b9cbfc6f2d331fbed8cb9392e84a19d595d09a90` |
| `vad-m1-hpl-ggr-utility` | `1b70ee2b71badb9d6c6016d94bfb133f33396035` |
| `vad-m1-utility-audit-v2` | `fef5516b2e1ec518ba381cb3ff1b9bfae160f858` |
| `vad-m1-v3-pseudolabel-calibration` | `e45575d642b301e0410642054cef6ee0999bdc7d` |

The active branches are `main`, `rankup-reproduction`, `emobank-rankup-3d`, `emobank-ssr-rcus-3d`, `emobank-strong-supervised`, and `m1/avrrc-oof`. `emobank-ssr-rcus-3d-adapt` is retained in the `emobank-ssr-rcus-3d` history.

## Former default branch

`main` ended at `7ed6d267783d9dc8f99ae8188b126d125bff3ab79`; that commit is an ancestor of this archive branch and of `rankup-reproduction`. The repository default branch was moved to `rankup-reproduction` before removing `main`.
