## fidelity-variation

I've found a large variation in the final MPS fidelity when training on the XY-quench states.

1. Is this generally true?
2. If so, can it be 'cured' by reducing batch size? Or lr decay?

## ID
(Used to specify data folder, etc)
### fidelity-variation-001
    Here, I'm just training with a fixed batch size on various ground states, to see whether the high fidelity variance observed in the XY training
    exists when training on ground states as well.