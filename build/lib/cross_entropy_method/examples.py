
import numpy as np
from scipy import stats
from cem import CEM


class CEM_Ber(CEM):
    '''Implementation example of the CEM for a 1D Bernoulli distribution.'''

    # Note: in this 1D case, phi is a scalar in [0,1]. In general, phi may be
    #  any object that represents a distribution (e.g., any kind of array).

    def __init__(self, *args, **kwargs):
        super(CEM_Ber, self).__init__(*args, **kwargs)
        # Optional variables specifying the names of the distribution parameters and the samples
        #  in the summarizing tables. In our case, both are one-dimensional. If they were
        #  multi-dimensional, we should have provided lists of names instead of strings.
        self.default_dist_titles = 'p'
        self.default_samp_titles = 'sample'

    def do_sample(self, phi):
        return int(np.random.random()<phi)

    def pdf(self, x, phi):
        # note: x should be either 0 or 1
        return 1-phi if x<0.5 else phi

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
        return np.mean(w*s)/np.mean(w)


class CEM_Beta(CEM):
    '''CEM for 1D Beta distribution.'''

    def __init__(self, *args, **kwargs):
        super(CEM_Beta, self).__init__(*args, **kwargs)
        self.default_dist_titles = 'beta_mean'
        self.default_samp_titles = 'sample'

    def do_sample(self, phi):
        return np.random.beta(2*phi, 2-2*phi)

    def pdf(self, x, phi):
        return stats.beta.pdf(np.clip(x,0.001,0.999), 2*phi, 2-2*phi)

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
        # We avoid boundary-values for numerical stability
        return np.clip(np.mean(w*s)/np.mean(w), 0.001, 0.999)


if __name__ == '__main__':
    print('We draw numbers from U(0,1) (which is equivalent to Beta(1,1)). '
          'Every batch we update the distribution using the bottom half of '
          'the samples, or all the samples below the 10%-quantile of the '
          'original distribution (which is 0.1). Thus, we expect to converge '
          'to a distribution with E[X]=E[U(0,0.1)]=0.05. '
          'Below is the actual sample mean per Cross Entropy iteration:')

    n_steps = 10
    N = 1000

    ce = CEM_Beta(phi0=0.5, batch_size=N, min_batch_update=0.5, ref_alpha=0.1)
    for batch in range(n_steps):
        for iter in range(N):
            x, _ = ce.sample()
            # For this demonstration, we consider x as
            #  both sampled configuration and resulted score.
            score = x
            ce.update(score)
        # Take last batch scores (note: scores[-1] is the new empty batch).
        #  Exclude original-distribution scores.
        scores = ce.scores[-2][ce.n_orig_per_batch:]
        print(f'[{batch:d}] {np.mean(scores):.3f}')
