# SC-3 Prospective Control

SC-3 was introduced after the primary outcome; its own protocol was recorded
before SC-3 outcome analysis. It reuses the panel, membership, parser and CoT
template. See [chronology limits](FREEZE_CHRONOLOGY.md).

Seeds=20260831,20260832,20260833; three samples/item, reseeded per item/sample.
[Settings](../configs/sc3_generation_settings.json) retain primary stochastic
decoding for Gemma4/Qwen3.6; the other three change from greedy primary decoding
to the specified stochastic profiles. No primary CoT row is reused.

Two recovered agreeing slots form a majority; otherwise use first recovered in
sample_1 > sample_2 > sample_3 order. No recovered answer is incorrect. All six
orders are sensitivity analyses. Mean-single is fractional item correctness:
the mean of three binary sample correctness values. Its comparison isolates
aggregation, distinct from original one-call CoT as deployment comparator.

SC-3 B=10,000; base seed=20260816 plus the code's SHA256-derived
model/comparison/salt offset. Paired within-dataset resampling, equal weights,
percentile 95% intervals. Ordinary McNemar does not apply to the fractional
comparator; no multiplicity-adjusted SC-3 p-value family is claimed.
Three calls/item matches Fusion's calls, not output-token cost or FLOPs.
