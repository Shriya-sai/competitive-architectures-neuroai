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

Structured competition is a constrained interaction among artificial units or
feature populations in which activity in one group selectively suppresses
activity in specified other groups according to an organized topology, while
ordinary feedforward or residual pathways retain cooperative integration.

This definition is provisional until the Stage 0 decision gate is passed.

## What the construct is not

- the mere presence of negative weights;
- global normalization relabeled as biological inhibition;
- an unmatched increase in parameters or compute;
- a mechanism whose structure is derived from the test set;
- a performance regularizer described as brain-like without representational evidence.

## Required controls

1. **Standard:** no explicit competitive module.
2. **Random competition:** matched strength, sparsity, degree distribution,
   parameter count, and placement depth, but shuffled interaction topology.
3. **Structured competition:** the same competitive budget with the frozen
   organized topology.

Any additional global-competition condition is secondary.

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

## Outcome families

Primary endpoints must be selected before confirmatory runs from:

- in-distribution performance;
- held-out generalization;
- corruption or adversarial robustness;
- distribution shift;
- adaptation or flexibility;
- sample efficiency.

Representation specialization, interference, dynamics, and RSA are explanatory
or secondary outcomes unless explicitly promoted before confirmatory training.

## Stage 0 decision gate

Stage 0 ends only when one mechanism family has:

1. a precise mathematical definition;
2. a magnitude- and topology-matched random control;
3. a zero-competition equivalence test;
4. frozen primary endpoints and failure criteria;
5. a minimal task capable of falsifying the hypothesis.

Only then may architecture and dataset selection begin.
