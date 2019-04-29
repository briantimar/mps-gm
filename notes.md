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

### Meeting with Manuel and Evert
One thing to check: can random-unitary measurements be used to obtain fidelities between two density matrices? Ie Tr(rho1 rho2), where rho1 is an experimental state and rho2 is a reconstruction. 

Another mixed state ansatz: MPS with some sites traced out. 

To try: 
    - 
    - pure state MPS "just try it"
        - first, for globally pure states
        - then, add a bit of Lindblad
            - "how far away are we in local density matrices"
        - then, how to make nonpure ansatzes?
            - 
        - can you get (bounds on) reconstruction fidelity using rotated bases? 

### update: I'm a dumbass
The contract_interval() method wasn't including local rotations. fixed.
Also in grad_twosite_logprob()

Having fixed typos in models.py, it seems that the gradient-computation under local rotations works OK. I think my current difficulties in training stem from entropy of the measurement-outcome distribution for small datasets. Consider:
    - Training on z-product state, 100 samples in z basis only -> no problem
    - Training on x-product state, 100 samples in x basis only -> no problem
    - Training on z-product state, 100 samples in x basis only -> poor training
    - Training on x-product state, 100 samples in z basis only -> poor training.

Aha! And indeed, if I now switch back to the larger datasets, training on informationally complete basis sets, training seems to work very well indeed.

### A few practical notes.
    - So far, still haven't trained well on GHZ states (have only tried discrete bases here)
        - Using a smaller SV cutoff leads to better results, but not great
        - Random-basis training on GHZ states does lead to some learning, but it's very poor. Need to go to an easier case first (eg product states in random bases)
            - haven't yet implemented entropy penalty during training.


## April 29

### Status:
    * Have demonstrated successful discrete, multi-basis training on product states in the z basis.
    * Updated models.py to include two-site SGD update. In the process I may have fixed some bug in the original SGD update code, because I can now train on multi-discrete-basis GHZ data to get MPSs with the correct probabilities in the z-basis. 
    * **However**, the relative phase between the two basis states is incorrect: out of phase by 75 degrees (should be zero).
        * This success seems to be precarious / initialization-dependent: on a second training run the ghz probabilities come out at .4 each, with the angle still far from zero. I guess the training is getting stuck in local minima. Right now I'm using vanilla SGD, and no entropy regularization. Might be cool to try adding the regularization next.
        * Seems that the final angle varies strongly between training attempts -- indicating perhaps that it's not well-constrained by the data?