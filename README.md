# Correlated Equilibria Q-Learning

This project reporduces the results of the ["Correlated Q-Learning" by _Amy Greenwald](https://www.aaai.org/Papers/ICML/2003/ICML03-034.pdf).

In this project I've implemented and compared 4 Q-Learning algorithms in application to Markov game "Soccer" similarly to the paper approach.

Summary of the project results can be found in Report.pdf

## Environment
To run this code one has to create _python 3_ environment with following libraries:

_numpy, pandas, matplotlib, cvxopt, jupyter_

- src/envs package contains Soccer environment implementation . 

- src/learners package contains all Q-Learners implementations . 

- src/Q-Learner.ipynb - jupyter notebook with standard Q-Learner experiment . 

- src/Foe-Q.ipynb - jupyter notebook with Foe Q-Learner experiment . 

- src/Friend-Q.ipynb - jupyter notebook with Friend Q-Learner experiment . 

- src/CE-Q.ipynb - jupyter notebook with CE Q-Learner experiment . 


## Execution
To reproduce the results one can run in the terminal with created environment: _jupyter notebook_

After that open corresponding to the experiment .ipynb notebook file in jupyter web interface and run all cells.

All graphs, results mentioned in the report can be found in this notebook.
