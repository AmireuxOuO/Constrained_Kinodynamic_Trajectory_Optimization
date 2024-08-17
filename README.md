# Constrained Kinodynamic Trajectory Optimization

The optimization problems are discretized using collocation methods and formulated as nonlinear programming (NLP) problems.

## Dependencies

To get started, ensure the following dependencies are already installed:

+ [CasADi](https://github.com/casadi/casadi): Symbolic automatic differentiation and optimization.
+ [Pinocchio3](https://github.com/nmansard/jnrh2023): Library for both numerical and symbolic rigid body kinematics and dynamics.
+ [CuRobo](https://github.com/NVlabs/curobo/tree/main): Collision geometry generation.

## Demo

The repository includes demo implementations for the following robotic systems:

- Simple unconstrained systems   <-   UR10 demo
- Unconstrained systems with collision avoidance   <-   UR5 demo
- Systems with holonomic constraints   <-   four-bar parallel robot demo

### TODO

- [ ]  Differentiable collision detection for general geometries, i.e., cuboids, cylinders, meshes, etc.
- [ ]  Systems with non-holonomic constriants.

## Reference

- Haug, Edward J. "Multibody dynamics on differentiable manifolds." Journal of Computational and Nonlinear Dynamics. 2021. https://doi.org/10.1115/1.4049995
