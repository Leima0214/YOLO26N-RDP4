# Frozen promotion gates

Reference evidence:

```text
B0-100E unified AP       0.24142
RoadSnake-R1 full AP     0.24727
RoadSnake gamma0/T1 AP   0.24694
RoadSnake-R1-30E AP      about 0.2404
RoadSnake-R1-30E AP75    about 0.1928
```

## SA-RS 30E to 100E

All primary conditions are evaluated with the unified Val-only evaluator:

- AP50-95 >= 0.2424.
- AP75 >= 0.1928.
- AP-small is not materially below matched R1-30E.
- D10 does not materially regress.
- scales do not mass-saturate below 0.45 or above 2.40.
- mechanism evidence preferably shows increasing scale with size, such as `median(s_small) < median(s_medium) < median(s_large)` or positive area-scale Spearman.

Failure means **FAIL / FREEZE**. No range, K, max-offset, or R2 rescue sweep follows.

## SA-RS 100E deployment result

The physically-pruned checkpoint evaluated on Val is decisive:

```text
AP < 0.2470          FAIL
0.2470 to <0.2499   weak positive
AP >= 0.2499         GO
AP >= 0.2519         Strong GO
```

## MG-SA-RS unlock

MG-SA-RS remains locked unless SA-RS 100E pruned unified Val AP is at least 0.2499. The training entry checks this from the SA pruned `metrics.json`; a prose claim or full-unpruned metric cannot unlock MG.
