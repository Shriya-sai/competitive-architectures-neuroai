# Structured Competition: Construct Specification

## Scientific question

Can structured cooperative–competitive interactions inspired by biological
brain networks improve artificial computation, and do they make learned
representations more brain-like?

## Biological starting point

The motivating whole-brain work suggests that cooperative and competitive
interactions jointly shape network dynamics, and that their organization matters.
This motivates an artificial analogue; it does not imply that a negative DNN
weight is biologically equivalent to an inhibitory or competitive connection.

## Provisional construct

Give a DNN an explicit signed lateral interaction architecture in which activity
in one hidden representation can selectively facilitate some representations
and suppress others, then test whether the existence, composition, and
organization of those interactions changes computation.

For hidden representation vector `h`, the provisional single-step operator is

`h_out = phi(h + g_pos C_pos h - g_neg C_neg h)`,

where `C_pos` and `C_neg` are nonnegative cooperative and competitive magnitude
matrices with disjoint frozen masks. Interaction magnitudes may be learned under
their sign constraints; mask placement and the allowed gain range are frozen.
The operator is an artificial signed lateral inductive bias, not a claim that a
DNN unit or negative parameter is biologically equivalent to a neuron or
inhibitory synapse.

This definition remains provisional until its stability rule, precise graph
construction, controls, and falsification thresholds pass the Stage 0 gate.

## Initial organization hypothesis

The first candidate is a modular signed topology. Channels are assigned before
training to structurally neutral populations. Cooperative interactions are
concentrated within populations, while selective competitive interactions occur
between populations. Membership carries no semantic meaning: task learning must
determine whether functional specialization emerges.

The structured graph will be evaluated across multiple partition seeds. Its
random control will use signed degree-preserving rewiring so that positive and
negative edge counts, per-node signed degree, sparsity, magnitude distributions,
layer placement, parameter count, and interaction gains are matched.

## What the construct is not

- the mere presence of negative weights;
- global normalization relabeled as biological inhibition;
- an unmatched increase in parameters or compute;
- a mechanism whose structure is derived from the test set;
- a performance regularizer described as brain-like without representational evidence.

## Required controls

1. **Standard DNN:** no explicit lateral interaction module.
2. **Compute-matched lateral control:** comparable parameters and computation
   without the frozen signed organization.
3. **Random signed architecture:** matched positive and negative budgets with
   degree-preserving rewired topology.
4. **Structured signed architecture:** modular cooperative–competitive topology
   under the same budget.

Cooperative-only and competitive-only conditions are secondary mechanistic
ablations. An activation-matched control is required if the signed operator
changes representation scale systematically.

## Candidate mechanism families

- groupwise lateral channel competition;
- similarity-organized competition;
- local topographic competition;
- competitive gating between pathways;
- hierarchical competition at selected depths;
- learned sparse competition under explicit constraints.

Stage 0 will compare these candidates conceptually before implementation. The
initial preference is groupwise lateral channel competition because it permits
clean matched controls and direct inspection.

## Construct-validity requirements

Before training a full model, the selected module must demonstrate that:

- increasing competitive gain predictably changes only the intended interaction;
- zero gain exactly recovers the standard computation;
- shuffled and structured variants are matched in non-topological properties;
- parameter count and effective training budget are controlled;
- gradients remain finite over the frozen operating range;
- any advantage cannot be explained solely by reduced activation magnitude.

## Initial computational task

The primary candidate task is sequential class-incremental visual learning. A
single shared classifier encounters disjoint category sets over time and is
evaluated on every category seen so far without task identity at test time.
Ordinary sequential fine-tuning, without replay or a specialized continual-
learning method, isolates architectural effects on representational
interference.

Development will use a small Split-CIFAR stream to verify learning, forgetting,
stability, and absence of floor or ceiling effects. Untouched confirmation will
use a frozen ten-experience Split CIFAR-100 stream, paired initialization and
class-order seeds, and multiple partition seeds.

The provisional primary endpoint is average incremental accuracy because it
rewards both acquisition and retention. Final average accuracy and forgetting
are secondary, while new-experience acquisition is a mandatory guardrail: a
model cannot count as successful merely because it learns little and therefore
forgets little.

Representation specialization, interference, dynamics, robustness, and RSA are
explanatory or secondary outcomes unless explicitly promoted before
confirmatory training.

## Later exploration–exploitation extension

Only after the signed mechanism and its computational effect have been
validated will all architectures be placed behind the same action-selection and
reward-learning rule in a nonstationary contextual-bandit or reversal-learning
task. This phase will test adaptive exploration, perseveration, switch latency,
recommitment, reward, and regret without building the desired behaviour into a
condition-specific policy. It is not part of the first confirmatory claim.

## Stage 0 decision gate

Stage 0 ends only when one mechanism family has:

1. a precise mathematical definition;
2. a magnitude- and topology-matched random control;
3. a zero-competition equivalence test;
4. frozen primary endpoints and failure criteria;
5. a minimal task capable of falsifying the hypothesis.

Only then may architecture and dataset selection begin.
