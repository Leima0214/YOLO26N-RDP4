# M2 experiment status

- Verdict: **FAILED / USER-STOPPED**
- Protocol: Japan4-cleanV3, intended 30 epochs
- Run: `formal_M2_mshc_p4x2_japan4_cleanv3_e30_img640_b32_seed42_20260808`
- Stop point: epoch 22 training was interrupted; epoch 21 is the final complete validation row.
- Final complete epoch AP50-95: `0.16962`
- Final complete epoch AP50: `0.36363`
- B0-S30 AP50-95 reference: `0.23570`
- Delta at final complete epoch: `-0.06608`
- Runtime exit code: `1` (intentional interruption)
- `results.csv` SHA256: `6e078f94359737940c0ada2c5d2c5fb5b919a752b82635e3b740d30311bed824`
- `best.pt` SHA256: `33512b63d8246a3e1ce8d13ea1363208b0ef60256e63b34748220bb214e7928c`
- `last.pt` SHA256: `33512b63d8246a3e1ce8d13ea1363208b0ef60256e63b34748220bb214e7928c`

This run is not a completed 30E result and must not be reported as one. Its partial trajectory is retained only as negative evidence; M2 is frozen and must not be resumed.
