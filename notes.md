#Notes

# April 22
Current status: training of (complex-valued) MPS on single-basis data using local updates precedes very smoothly indeed. On the L=4 dataset, without hyperparam fine-tuning I get better results than I'd managed to acheive with global, non-adaptive updates.

However, training so far has not succeeded on data sets including multiple bases. I think there are two primary suspects:
    - My implementation of the unitaries is either wrong, or not consistent with conventions used in generating the datasets.
    - All the implementations are OK, it's just the optimization procedure itself that needs to be improved. 

The former seems more likely.