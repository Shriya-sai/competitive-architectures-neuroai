# First DNN experiment: result and diagnosis

## Question

Does a structured cooperative–competitive channel topology improve
class-incremental learning relative to a degree- and capacity-matched random
signed topology?

## Design

The model was a small CNN trained on five two-class Split CIFAR-10 experiences
with a shared ten-class head and fixed replay memory. The three conditions were
standard, random signed, and structured signed. The confirmation configuration
was frozen before testing and used ten fresh paired class-order seeds; the
development seed was excluded. Average incremental accuracy was the primary
outcome, and structured minus random signed was the primary contrast.

The complete frozen configuration is in
`configs/split_cifar10_replay_confirmation.json`.

## Confirmation result

Mean average incremental accuracy was:

| Condition | Mean |
| --- | ---: |
| Standard | 0.6126 |
| Random signed | 0.6109 |
| Structured signed | 0.6116 |

For structured minus random signed, the paired mean difference was 0.00066,
the 95% paired-bootstrap interval was [-0.00257, 0.00446], and the exact
two-sided sign-flip p-value was 0.764. Structured won five of ten paired seeds.
No secondary outcome showed a reliable structured-topology advantage.

## Causal pathway diagnosis

The confirmation result alone could not reveal whether signed topology was
irrelevant or whether the network barely used the signed mechanism. We
therefore retrained frozen confirmation seed 31 and evaluated each signed model
normally and with its lateral interaction bypassed after training.

| Diagnostic | Random signed | Structured signed |
| --- | ---: | ---: |
| Normal accuracy | 0.4763 | 0.4734 |
| Bypass accuracy | 0.4769 | 0.4731 |
| Normal minus bypass accuracy | -0.0006 | 0.0003 |
| Prediction disagreement | 0.0059 | 0.0066 |
| Residual/backbone norm ratio | 0.0128 | 0.0121 |
| Post/backbone cosine similarity | 0.99992 | 0.99993 |

Random and structured post-interaction representations were also nearly
identical under linear CKA (0.9874). In the structured model, correlations on
cooperative edges were not greater than correlations on competitive edges;
the cooperative-minus-competitive separation was -0.0172.

This was an exploratory single-seed diagnosis, not a new confirmation test.
Nevertheless, all converging measurements indicate that the classifier could
rely on the residual backbone and largely bypass the signed correction.

## Interpretation and decision

The result does not support a claim that the present structured topology
improves continual learning. It also does not yet constitute a strong test of
whether an engaged structured competitive mechanism changes computation.

Before another multi-seed performance experiment, a revised architecture must
pass a pathway-engagement gate:

1. bypassing the signed pathway materially changes predictions;
2. the pathway produces a measurable representational change;
3. the intended topology is reflected in feature organization;
4. training remains stable and capacity matching remains defensible.

Candidate interventions are a signed-gain dose response, a learned gate, and a
non-bypassable signed bottleneck. These should first be compared on development
seeds. A new confirmation set should be run only after one intervention passes
the engagement gate.

## Development-seed pathway screen

The weak residual, a learned gated residual, and a normalized non-bypassable
signed bottleneck were subsequently compared on development seed 23. Random
and structured topology remained degree- and capacity-matched within every
pathway design. This screen was diagnostic and was not a confirmatory
performance comparison.

| Pathway | Topology | Accuracy | Bypass effect | Prediction disagreement | Post/backbone cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| Weak residual | Random | 0.4384 | 0.0002 | 0.0052 | 0.99993 |
| Weak residual | Structured | 0.4637 | 0.0013 | 0.0071 | 0.99990 |
| Gated residual | Random | 0.4520 | 0.0003 | 0.0025 | 0.99997 |
| Gated residual | Structured | 0.4626 | 0.0007 | 0.0035 | 0.99997 |
| Signed bottleneck | Random | 0.4694 | 0.3804 | 0.9100 | 0.06995 |
| Signed bottleneck | Structured | 0.4288 | 0.3132 | 0.9570 | 0.00425 |

The learned gates remained active (0.599 for random and 0.577 for structured),
but they scaled an interaction that was already too weak to affect decisions.
The gated residual therefore did not solve the bypass problem.

The bottleneck was the only design to establish causal pathway necessity while
remaining trainable. Random-versus-structured post-interaction CKA fell from
approximately 0.98--0.99 in the residual designs to 0.839 in the bottleneck.
Its single-seed accuracy difference must not be interpreted as evidence for or
against structured topology.

The bottleneck did not yet satisfy the provisional topology-organization test:
cooperative-edge backbone correlations were not greater than competitive-edge
correlations in the structured condition. The next step is therefore to define
and validate a topology-engagement measurement appropriate to a
non-bypassable signed transformation before running additional performance
seeds.

## Class-tuning topology diagnosis

The initial correlation test compared raw activations and did not directly ask
whether connected channels represented similar or opposing class information.
A new instrument therefore computed each channel's centered class-tuning
profile and measured two preregisterable quantities:

- signed-edge tuning gap: cooperative-edge similarity minus
  competitive-edge similarity;
- group tuning gap: within-structural-group similarity minus between-group
  similarity.

Synthetic unit tests verified that both scores recover known group-organized
class tuning and remain near zero for unorganized features. Applied to the
development-seed bottleneck models, the results were:

| Representation | Topology | Signed-edge gap | Group gap |
| --- | --- | ---: | ---: |
| Backbone | Random | 0.091 | -0.029 |
| Backbone | Structured | 0.025 | 0.004 |
| Post-bottleneck | Random | 0.013 | -0.005 |
| Post-bottleneck | Structured | 0.659 | 0.240 |

The structured bottleneck therefore imposes a functionally expressed
cooperative--competitive organization on its output: cooperative edges link
channels with similar class tuning, competitive edges link channels with
opposing tuning, and the predefined groups become internally more coherent.
The matched random bottleneck does not show the same organization.

The absence of comparable alignment in the structured backbone is equally
important. This is evidence for **architecturally imposed topology engagement**,
not evidence that the upstream CNN learned or spontaneously discovered the
predefined modules. Replication across development seeds is required before
freezing a new performance confirmation.

## Reproduction

Run the frozen confirmation:

```bash
PYTHONPATH=src python scripts/run_frozen_multiseed.py
```

Run the seed-31 pathway diagnosis:

```bash
PYTHONPATH=src python scripts/run_pathway_diagnosis.py
```

Run the development-seed pathway engagement screen:

```bash
PYTHONPATH=src python scripts/run_pathway_engagement_suite.py
```

Run the bottleneck class-tuning topology diagnosis:

```bash
PYTHONPATH=src python scripts/run_bottleneck_topology_diagnosis.py
```

Generated result JSON files are intentionally excluded from version control.
