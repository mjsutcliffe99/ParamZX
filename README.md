ParamZX

This repository hosts the necessary code to demonstrate the parameterised ZX-calculus reduction (and GPU parallel evaluations) method outlined in the paper 'Fast classical simulation of quantum circuits via parametric rewriting in the ZX-calculus'.

Contained within is:
  - an updated version of the PyZX library (https://github.com/Quantomatic/pyzx) to support parameterised diagrams and reduction
  - a Jupyter notebook which demonstrates how Clifford+T circuits may be reduced into parameterised scalars (and structured to be GPU-ready)
  - CUDA code that reads these GPU-ready parameterised scalars and performs speedy evaluations upon then (and benchmarks the speed measurements)

