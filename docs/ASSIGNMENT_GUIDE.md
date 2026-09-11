# Assignment guide

The [original three-page assignment](../assignment/original-assignment.pdf) is the source of the requirements below. The [completed report](../report/AI6102_Assignment.pdf) contains the author's answers and experimental evidence.

## Scope: 25 marks available

| Question | Requirement | Lecture reference | Deliverable |
|---|---|---|---|
| Q1 · 10 marks | Derive regularized logistic regression for C classes using the supplied conditional probability model. Define the loss and derive gradient descent updates. | L4, slide 39; binary regularization on slide 38 | Probability model, objective, gradients, learning procedure |
| Q2 · 5 marks | Use scikit-learn SVC on the official a9a train/test split with both linear and RBF kernels. | L5, slides 63–64 | Three result tables and the chosen model settings |
| Q3 · 5 marks | Reformulate linear soft-margin SVM as structural risk minimization using hinge loss. | L5, slides 36–37 | Equivalent unconstrained objective and coefficient correspondence |
| Q4 · 5 marks | Apply the kernel trick to regularized linear regression and derive its closed-form solution. | L3, slide 50; L5 kernel material | Kernel coefficient solution and prediction expression |

## Q2 experiment requirements

1. Use the preprocessed **a9a** training and test files from LIBSVM.
2. For the linear kernel, evaluate **C ∈ {0.01, 0.05, 0.1, 0.5, 1}** with three-fold cross-validation on the training set. Report classification accuracy in Table 1.
3. For the RBF kernel, evaluate all **25 combinations** of those C values and **gamma ∈ {0.01, 0.05, 0.1, 0.5, 1}**. Report three-fold validation accuracy in Table 2.
4. Select the best kernel and settings using Tables 1–2, fit that configuration on the complete training set, and report test accuracy and the selected configuration in Table 3.
5. Other SVC parameters may remain at defaults; explicitly report any settings that are specified differently.

The brief does not specify a random seed, shuffling policy, or a required choice of stratified folds. The choices for this run are recorded in [the experiment protocol](EXPERIMENTS.md).

## Submission requirements in the source brief

- Submit **one PDF** through **Assignments** on the course's NTULearn site.
- Source code is **not required** for submission. This repository retains code and results for reproducibility.
- Multiple submission attempts are allowed; the last one is graded.
- Textbooks and online reference material may be consulted. Plagiarism from classmates is prohibited.
- No page limit, filename convention, font requirement, or mandated report template is stated in the brief.

## Version notes

The original PDF title reads **AI6012**, while the supplied lecture decks and final report use **AI6102**. The source file is retained unchanged.

The brief's text layer contains an old deadline of **30 September 2022, 11:59 p.m.**, while that deadline is not visibly printed on its rendered first page. It is historical source metadata, not a current deadline. The current course submission page is authoritative for scheduling.

Uploading this archive to GitHub does not itself submit the assignment to NTULearn.
