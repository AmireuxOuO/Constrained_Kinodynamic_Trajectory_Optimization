# Constrained Kinodynamic Trajectory Optimization

Optimal control problems (OCP) are discretized and formulated as nonlinear programming (NLP) problems.

## Dependencies

To get started, ensure the following dependencies are already installed:

+ [CasADi](https://github.com/casadi/casadi): Symbolic automatic differentiation and optimization.
+ [Pinocchio3](https://github.com/nmansard/jnrh2023): Library for both numerical and symbolic rigid body kinematics and dynamics.
+ [CuRobo](https://github.com/NVlabs/curobo/tree/main): Collision geometry generation.

## Demo

### Algorithms

+ Direct discretization of optimal control problems using Gauss-Legendre collocation method.
+ Constrained dynamics on configuration manifolds using local parametrization on targent charts.

The repository includes demo implementations for the following robotic systems:

- Simple unconstrained systems (with collision avoidance)   ->   UR demo
- Systems with holonomic constraints   ->   five-bar parallel robot demo

### TODO

- [ ]  RRT path planning for generating initial guesses in NLP with collision avoidance.
- [ ]  General collision geometries, i.e., cuboids, cylinders, meshes, etc.
- [ ]  Systems with non-holonomic constriants.

## Reference

- Haug, Edward J. "Multibody dynamics on differentiable manifolds." Journal of Computational and Nonlinear Dynamics. 2021. https://doi.org/10.1115/1.4049995
