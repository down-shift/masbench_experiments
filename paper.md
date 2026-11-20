# Candidate Listing Order Changes Scheduling Decisions in a Controlled LLM Leader Task

## Abstract

A scheduling assistant must select a meeting consistent with participants' availability. We study whether a language model makes the same decision when two candidate lines are presented in a different order. We construct paired, two-candidate decisions from MASBench scheduling calendars. Each instance contains explicit availability statements for five participants and exactly one feasible candidate. On 32 held-out scenarios, each evaluated in two availability worlds, Qwen3-8B with 4-bit BitsAndBytes quantization selected the feasible candidate in 56 of 64 worlds when candidate A was listed first and 64 of 64 when candidate B was listed first. All eight changed answers went from an incorrect A to a correct B. The paired accuracy difference was 12.5 percentage points, with a scenario-clustered bootstrap interval of 4.7 to 20.3 points. A Qwen3-4B run did not obey the required output format, preventing the same strict comparison. These results show sensitivity to a specific presentation change in one model configuration. They do not identify an internal mechanism or establish an effect of participant speaking order.

## 1. Introduction

A scheduler that aggregates participant constraints should choose a meeting everyone can attend. Even when availability is stated explicitly, the form in which candidates are displayed may influence the model's decision. This is relevant to multi-agent systems because a leader may receive equivalent information in different textual layouts.

We investigate a controlled decision derived from MASBench scheduling cases. Each prompt presents two candidate times and five participant messages. Exactly one candidate is available to everyone. We compare two prompts for the same decision that differ only in the order of the two candidate listing lines. The candidate labels remain attached to the same times; the participant messages continue to describe A before B.

Exploratory free-form scheduling runs produced no valid final meetings, making it difficult to assess how the leader used evidence. We therefore isolate the leader's decision using constructed availability messages. On a separate set of 32 scenarios, reversing the candidate lines corrected eight decisions and worsened none under the tested configuration. This result concerns a controlled leader task rather than successful end-to-end agent collaboration.

## 2. Related work

MASBench studies multi-agent collaboration under partial observability, including scheduling with private participant calendars [1]. Our task uses its scheduling calendars to construct explicit leader inputs; the participant messages are generated from calendar availability, not obtained from a model conversation. Prior research has studied position sensitivity in multiple-choice and other LLM evaluations [2]. Our experiment applies a paired presentation intervention to a small scheduling decision. It does not claim that position sensitivity is a new phenomenon.

## 3. Task and experimental design

### Calendar pairs and candidate selection

Each scenario contains released and alternative calendars. The two worlds have disjoint feasible meeting starts. One designated participant's calendar changes between worlds, while the other four participants' availability remains fixed. From each pair, we deterministically select one 20-minute meeting feasible in the released world and one feasible in the alternative world. We require the designated participant's availability for those candidates to reverse between worlds and all four other participants to be available for both. The saved calendar-derived inputs were checked against the paired plan.

Each world is rendered as two labeled candidate lines followed by five factual messages of the form `Member1: A available; B unavailable.` A deterministic intersection of the availability statements yields exactly one correct candidate. The leader is asked to output exactly `FINAL A` or `FINAL B`. Each world is evaluated independently; the model is not told the answer in the other world.

### Listing-order intervention

The A-first prompt lists the line for A followed by the line for B. The B-first prompt exchanges those two lines. Their labels, candidate times, availability statements, message order, and output instruction remain unchanged. Thus this intervention changes the location of both candidate lines while leaving other A-before-B references intact. It does not distinguish all possible effects of labels, position, and distance from subsequent text.

### Data and evaluation

Four smoke scenarios and 32 exploratory scenarios were excluded from the confirmation pool. From the remaining audited cases, 32 eligible scenarios were selected using seed 9401. The confirmation set has 64 world decisions, each evaluated under both listing orders, for 128 model calls. The comparison was selected after inspecting exploratory results; the scenarios themselves were held out from that development. We do not claim formal preregistration.

The primary outcome is exact-format decision accuracy per world. Any output other than the required single line is scored as incorrect. We report paired outcomes and the B-first minus A-first accuracy difference. A percentile bootstrap resamples scenarios, retaining both worlds and both prompt orders together, for 10,000 draws with seed 6975. This interval describes uncertainty under scenario resampling and should be interpreted cautiously given the small, selected sample.

The main model was Qwen3-8B served by vLLM with BitsAndBytes 4-bit quantization, temperature zero, thinking disabled, and a 256-token output limit. Calls for the two orders were interleaved by scenario and world. The saved responses report one model ID and server fingerprint. The 8B checkpoint revision is not established by the saved manifest and requires the server startup record for complete provenance.

## 4. Results

### Held-out listing-order comparison

| A-first outcome | B-first correct | B-first incorrect |
|---|---:|---:|
| Correct | 56 | 0 |
| Incorrect | 8 | 0 |

A-first accuracy was 56/64 (87.5%); B-first accuracy was 64/64 (100%). The paired difference was 12.5 percentage points, with a scenario-bootstrap 95% interval of 4.7 to 20.3 percentage points. All eight changed answers moved from `FINAL A` to the correct `FINAL B`. The remaining 56 outputs were unchanged and correct. At the scenario level, both worlds were correct in 24/32 scenarios with A first and 32/32 with B first. All outputs met the strict format and all requests finished normally.

### Exploratory observations

In the preceding 32-scenario exploratory set, the same listing-line reversal changed ten incorrect A responses to correct B responses, with no regressions. A separate candidate-label swap changed the chosen meeting in 21 of 64 paired world comparisons while retaining label A in each of those comparisons. These analyses motivated the held-out comparison and are not independent confirmation.

### Qwen3-4B output format

A second run used Qwen3-4B in BF16 on the held-out inputs. All 128 responses violated the requested single-line format; four A-first and six B-first outputs reached the token limit. Under the prespecified strict scorer, both conditions therefore have zero accuracy, which does not resolve whether the model selected the intended meeting in its prose. An exploratory search for a unique `FINAL A` or `FINAL B` found 60 correct among 60 identifiable A-first responses and 56 correct among 57 identifiable B-first responses. Unidentified responses remain unclassified. This extraction was chosen after inspecting the outputs and is not directly comparable to the primary analysis.

## 5. Discussion

Changing the order of the candidate listing lines changed Qwen3-8B's decisions on eight held-out worlds. The unaffected participant messages still contained enough explicit information to determine the correct choice. The result supports a presentation-sensitivity claim for this prompt and model configuration, without establishing whether the effect is caused by a preference for a label, line position, proximity to the participant messages, or another textual cue.

The task intentionally supplies the two candidates and factual availability statements. It does not evaluate whether agents can find candidates, communicate their calendars faithfully, or negotiate a meeting. The original free-form pilots provide motivation for isolating the leader, but the difference between those pilots and the controlled task cannot locate the source of their failures.

The sample consists of 32 selected scenarios from one benchmark and one prompt family. The main result uses one model and quantization configuration at temperature zero. The 4B run exposes a format limitation rather than a clean cross-model replication. Further work should verify model checkpoint provenance, test additional held-out scenarios and models, and prespecify a response protocol that can distinguish formatting failures from decision errors.

## 6. Conclusion

In a controlled scheduling leader task derived from MASBench, reversing only two candidate listing lines changed eight of 64 held-out decisions for the tested Qwen3-8B configuration. The change improved strict accuracy from 87.5% to 100% on this sample. Equivalent availability evidence did not guarantee an equivalent decision across these two presentations.

## References

[1] Qizhi Chu, Zekai Yu, Sijie Wen, Yang Liu, Chen Qian, Cheng Yang, Chuan Shi, and Zhiyuan Liu. *MASBench: Benchmarking LLM-based Multi-Agent Collaboration under Partial Observability*. 2026. https://arxiv.org/abs/2610.04672

[2] Yining Li, Huayang Li, Lemao Liu, and Taro Watanabe. *On the Position Bias of Large Language Models: A Comprehensive Study*. 2026. https://arxiv.org/abs/2610.22807
