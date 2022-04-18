# The Cross Entropy Method

The [Cross Entropy Method](http://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf) (CE or CEM) is an approach for optimization or rare-event sampling in a given class of distributions *{D_p}* and a score function *R(x)*.
* In its sampling version, it is given a reference *p0* and aims to sample from the tail of the distribution *x ~ (D_p0 | R(x)<q)*, where *q* is defined as either a numeric value *q* or a quantile *alpha* (where *q=q_alpha(R)*).
* In its optimization version, it aims to find *argmin_x{R(x)}*.

The exact implementation of the CEM depends on the problem setup.
This repo provides a general implementation as an abstract class, where a concrete use requires writing a simple, small inherited class.
The attached [`tutorial.ipynb`](https://github.com/ido90/CEM/blob/master/tutorial.ipynb) provides a more detailed background on the CEM and on this package, along with usage examples.

In our [separate work](https://github.com/ido90/CeSoR), we demonstrate the use of the CEM for the more realistic problem of sampling "difficult" environment-conditions in risk-averse reinforcement learning. There, *D_p* determines the distribution of the environment-conditions, *p0* corresponds to the original distribution (or test distribution), and *R(x; agent)* is the return function of the agent given the conditions *x*.

| <img src="https://idogreenberg.neocities.org/linked_images/CEM_toy_sampling.png" width="260"> <img src="https://idogreenberg.neocities.org/linked_images/CEM_toy_optimization.png" width="260"> |
| :--: |
| **CEM for sampling** (left): the mean of the sample distribution (blue) aims to coincide with the mean of the tail of the original distribution (black). **CEM for optimization** (right): the mean of the sample distribution aims to be minimized.   (images from `tutorial.ipynb`) |
