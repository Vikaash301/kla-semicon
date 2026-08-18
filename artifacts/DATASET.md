# Released dataset provenance

The archives are excluded from Git. These hashes identify the exact local inputs used for all reported experiments.

| Archive | SHA-256 |
|---|---|
| `train.zip` | `b93dc4486a1181338630a55a88596e722cfdf75a0c1bbe2ed8404f01980c0abb` |
| `Test_NoisyLR.zip` | `f2904f75d6938c23f7ad5f7d41194744a5cdceb3c1d1ea066b59a9dbf9b45f83` |

Audit result: 3,200 aligned training pairs (`128x128` LR to `256x256` GT) and 400 official test inputs. GT arrays are float32 in `[0,1]`; degraded arrays are float32 and may lie outside `[0,1]`.
