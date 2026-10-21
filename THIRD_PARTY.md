# Third-Party Code and Attribution

CSformer builds on the training framework and model utilities from
[PatchTST](https://github.com/yuqinie98/PatchTST). The reference source is commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9`, licensed under Apache-2.0. The license text is retained
in [licenses/PatchTST-Apache-2.0.txt](licenses/PatchTST-Apache-2.0.txt).

[RevIN](https://github.com/ts-kim/RevIN) is MIT-licensed. Its ETRI 2022 copyright notice and license
are retained in [licenses/RevIN-MIT.txt](licenses/RevIN-MIT.txt).

## Components

- `models/FTST.py` and `layers/FTST_backbone.py` implement CSformer's multi-scale patch embedding,
  residual dense block, cross-scale attention and forecast head. Their attention utilities build
  on PatchTST; the inherited utilities are not claimed as new contributions.
- `models/CSformer.py` exposes the paper model name while retaining checkpoint parameter names.
- PatchTST, DLinear, Linear, NLinear, Autoformer, Informer and Transformer modules are included as
  reference baselines. They were not independently reproduced as part of the CSformer runs.

## Modifications

Modified files carry notices describing their changes. Changes to the inherited framework include
CSformer model dispatch and CLI configuration, complete test-batch aggregation, portable checkpoint
loading, structured metric output, optional complete validation, final-only test evaluation,
and dataset launchers. The model-specific implementation, normalization and inherited baseline
modules retain their source attribution.

The project license is [Apache-2.0](LICENSE). Third-party components retain their original licenses.
