# Bayesian statistics

During this session the following topics will be discussed:

1. Bayesian statistics in practice (ch.1 Bayesian methods for hackers)

  - *showcase the process*: brief visualization of Bayesian posteriors
  - *automation*: introduce a Bayesian toolbox, PyMC
  - *examples*: inference with PyMC

2. Markov Chain Monte Carlo (custom notebook)

  - *from scratch*: let's build our simple Metropolis MCMC

3. MCMC usage (ch.3 Bayesian methods for hackers)

  - *realistic example*: 
  - *examination*: investigate the critical issues of MCMC, using PyMC tools


## Further tools

PyMC is an example of a family, whose structure follows more or less always the
following pattern: a probability and inference library in Python built on top of
a Python tensor library.

- **PyMC**, originally built on top of **Theano**, then **Aesara**, now
    **PyTensor** (just a fork chain)
- **TensorFlow Probability**, obviously built on top of **TensorFlow**
- **Pyro**, on top of **Pytorch**
- **NumPyro**, on top of **JAX**

A further inference library, publishing Python bindings, is **Stan**. 
Stan is a different and very popular library, following a more
statistically-oriented approach (as opposed to a more software oriented one),
essentially with its own declarative language.
