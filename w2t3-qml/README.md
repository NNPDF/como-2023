## Quantum Machine Learning tutorial 

The tutorial is divided into three parts (+ one optional):

1. we learn some basic concepts about QML and we execute a simple example of optimization using
   a quantum circuit as model and a genetic algorithm as optimizer.
2. we learn how to evaluate quantum gradients in a quantum-hardware compatible 
   way defining and using the [Parameter-Shift Rule](https://arxiv.org/abs/1811.11184) (PSR);
3. we build a quantum regressor to fit 1d Parton Density Functions (PDFs). We 
   implement a gradient-based optimization strategy based on the PSR;
4. **Optional**: we use an adiabatic evolution model to fit a probability density function. In particular
   , we encode the Cumulative Density Function of a distribution into an Adiabatic Evolution,
   which is then translated into a circuit composed of rotational gates. We apply the Parameter
   Shift Rule to derivate the circuit and calculate the probability density function from the
   cumulative one.

