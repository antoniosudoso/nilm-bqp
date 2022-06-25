## Mixed-Integer Nonlinear Programming for State-based Non-Intrusive Load Monitoring 
This repository provides the implementation of the NILM algorithm described in the IEEE Transactions on Smart Grid journal paper [Mixed-Integer Nonlinear Programming for State-based Non-Intrusive Load Monitoring](https://ieeexplore.ieee.org/document/9714495).

The optimization model is written in [AMPL](https://ampl.com/) and solved with [Gurobi Optimizer](https://www.gurobi.com/).

If you use this paper in your research please cite:
> M. Balletti, V. Piccialli and A. M. Sudoso, "Mixed-Integer Nonlinear Programming for State-Based Non-Intrusive Load Monitoring", IEEE Transactions on Smart Grid 2022, vol. 13, no. 4, pp. 3301-3314. https://doi.org/10.1109/TSG.2022.3152147.

### Main Scripts
For each dataset (i.e. AMPDS, UDKALE, REFIT):
- AMPL model file `nilm_binary.mod` contains the implementation of the Binary Quadratic Programming (NILM-BQP) model.
- AMPL run file `nilm_binary.run` loads the BQP model, reads the data and optimizes the model.


### Related Work

> V. Piccialli and A. M. Sudoso, "Improving Non-Intrusive Load Disaggregation through an Attention-Based Deep Neural Network", Energies 2021, 14, 847. https://doi.org/10.3390/en14040847.

See the source code [here](https://github.com/antoniosudoso/attention-nilm).
