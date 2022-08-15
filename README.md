# The Cross Entropy Method

The [Cross Entropy Method](http://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf) (CE or CEM) is an approach for optimization or rare-event sampling in a given class of distributions *{D_p}* and a score function *R(x)*.
* In its sampling version, it is given a reference *p0* and aims to sample from the tail of the distribution *x ~ (D_p0 | R(x)<q)*, where *q* is defined as either a numeric value *q* or a quantile *alpha* (where *q=q_alpha(R)*).
* In its optimization version, it aims to find *argmin_x{R(x)}*.

The exact implementation of the CEM depends on the distributions family *{D_p}* as defined in the problem.
This repo provides a general implementation as an abstract class, where a concrete use requires writing a simple, small inherited class.
The attached [`tutorial.ipynb`](https://github.com/ido90/CEM/blob/master/tutorial.ipynb) provides a more detailed background on the CEM and on this package, along with usage examples.

**Installation**: `pip install cross-entropy-method`.

| <img src="https://idogreenberg.neocities.org/linked_images/CEM_toy_quantile_sampling.png" width="260"> <img src="https://idogreenberg.neocities.org/linked_images/CEM_toy_optimization.png" width="260"> |
| :--: |
| **CEM for sampling** (left): the mean of the sample distribution (green) shifts from the mean of the original distribution (blue) towards its 10%-tail (orange). **CEM for optimization** (right): the mean of the sample distribution aims to be minimized.   (images from [`tutorial.ipynb`](https://github.com/ido90/CEM/blob/master/tutorial.ipynb)) |

### Supporting non-stationary score functions

On top of the standard CEM, we also support a non-stationary score function *R*.
This affects the reference distribution of scores and thus the quantile threshold *q* (if specified as a quantile).
Thus, we have to repeatedly re-estimate *q*, using importance-sampling correction to compensate for the CEM distributional shift.

In our [separate work](https://github.com/ido90/CeSoR), we demonstrate the use of the CEM for the more realistic problem of sampling high-risk environment-conditions in risk-averse reinforcement learning. There, *D_p* determines the distribution of the environment-conditions, *p0* corresponds to the original distribution (or test distribution), and *R(x; agent)* is the return function of the agent given the conditions *x*.
Note that since the agent evolves with the training, the score function is indeed non-stationary.

### Cite this repo
```
@misc{cross_entropy_method,
  title={Cross Entropy Method with Non-stationary Score Function},
  author={Ido Greenberg},
  howpublished={\url{https://pypi.org/project/cross-entropy-method/}},
  year={2022}
}
```
