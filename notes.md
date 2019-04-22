#Notes

# April 22
Current status: training of (complex-valued) MPS on single-basis data using local updates precedes very smoothly indeed. On the L=4 dataset, without hyperparam fine-tuning I get better results than I'd managed to acheive with global, non-adaptive updates.

However, training so far has not succeeded on data sets including multiple bases. I think there are two primary suspects:
    - My implementation of the unitaries is either wrong, or not consistent with conventions used in generating the datasets.
    - All the implementations are OK, it's just the optimization procedure itself that needs to be improved. 

The former seems more likely.

Update: just found at least one major flaw in the computation of amplitudes: the difference between basis state index and eigenvalue of the number operator. Under standard basis ordering conventions, where the most highly excited states comes first, these are opposites: (n eigenvalue) = 1 - (basis state index). My code for computing amplitudes and gradients was not taking this into account. 

That can be fixed simply by preprocessing the measurement outcomes:
indices = (1 - outcomes)/2
and feeding indices the MPS training. This doesn't affect the quality of training itself.

During training on rotated-basis sets, I find that the cost function starts increasing after an initial decrease. What I ought to do is train on a really simple set, like a two-qubit product state. 

Implementation of unitaries in pauli_exp seems to be OK.