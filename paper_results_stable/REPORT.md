# FedWork BNN communication analysis

## Comparison summary

| Method | Last accuracy | Best accuracy | Uplink GB | Downlink GB | Total GB |
|:--|--:|--:|--:|--:|--:|
| FedBop | 72.94% | 74.67% | 1.4694 | 1.0640 | 2.5334 |
| FedVote | 80.57% | 82.25% | 12.9834 | 415.3604 | 428.3438 |
| BiFL-BiML | 39.15% | 52.79% | 26.1275 | 65.5670 | 91.6946 |
| BiPruneFL | 37.89% | 38.92% | 1.6520 | 1.0810 | 2.7330 |

## Cost to accuracy target

| Target | Method | Reached | Round | Total GB |
|:--|:--|:--|--:|--:|
| 60% | FedBop | Yes | 223 | 2.0218 |
| 60% | FedVote | Yes | 23 | 12.8185 |
| 60% | BiFL-BiML | No | — | — |
| 60% | BiPruneFL | No | — | — |
| 65% | FedBop | Yes | 251 | 2.0511 |
| 65% | FedVote | Yes | 37 | 20.3055 |
| 65% | BiFL-BiML | No | — | — |
| 65% | BiPruneFL | No | — | — |
| 70% | FedBop | Yes | 610 | 2.4904 |
| 70% | FedVote | Yes | 73 | 39.5576 |
| 70% | BiFL-BiML | No | — | — |
| 70% | BiPruneFL | No | — | — |
| 72% | FedBop | Yes | 755 | 2.5232 |
| 72% | FedVote | Yes | 91 | 49.1837 |
| 72% | BiFL-BiML | No | — | — |
| 72% | BiPruneFL | No | — | — |
| 75% | FedBop | No | — | — |
| 75% | FedVote | Yes | 156 | 83.9445 |
| 75% | BiFL-BiML | No | — | — |
| 75% | BiPruneFL | No | — | — |

## Interpretation and cautions

- Source: supplied FedWork profiler .data files; not pixel digitization from plots.
- GB = 10^9 bytes, matching existing config x_axis_scale=1e-9.
- Counters are cumulative serialized application traffic from server perspective: total = ServerTotalRecvBytes + ServerTotalSentBytes. Downlink counts per recipient.
- Data are aligned by profiler round key, NOT by sequence index; repeated cumulative samples use the maximum sample in that round.
- Best-so-far plots / budget tables represent retrospective best observed checkpoints, not accuracy guaranteed stable at that round.
- Cost-to-target requires 10 consecutive evaluation(s) >= threshold; first qualifying point is the end of that sequence.
- Unreached thresholds are reported as not reached (never estimated/extrapolated).
- Nonstandard BN recalibration messages, possible initial setup overhead, network framing, retransmissions and edge inference compute are NOT necessarily included by these counters.
- For true paper-level total communication, audit any communications outside standard model packets. In particular, FedBop BN calibration may be unaccounted.
- A single run/seed does not provide uncertainty estimates; repeat with multiple seeds.
- FedVote four-bit lossless vote-count broadcast is NOT assumed here: only measured bytes are shown.
