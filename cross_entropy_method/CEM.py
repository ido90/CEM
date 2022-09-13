'''
This module implements the Cross Entropy Method (CEM) for sampling of low quantiles.

The user should have a stochastic process P(score; theta), whose distribution
depends on the parameter theta (the module was originally developed to sample
"difficult" conditions in reinforcement learning, so theta was the parameters
of the environment, and the score was the return of the episode).

The basic CEM class is abstract: any use of it requires inheritance with
implementation of the following methods:
- do_sample(dist): returning a sample from the distribution represented by dist.
- pdf(x, dist): the probability of x under the distribution dist.
- update_sample_distribution(samples, weights): updating dist given new samples.
- likelihood_ratio(x) (optional): implemented by default as
                                  pdf(x, orig_dist) / pdf(x, curr_dist).
                                  the user may provide a more efficient or stable
                                  implementation according to the underlying
                                  family of distributions.
Note that dist is an object that represents the distribution, and its type is
up to the user. A standard type may be a list of distribution parameters.

Examples for inheritance from CEM (CEM_Ber, CEM_Beta) are provided in examples.py,
as well as a simple usage example (__main__).

Module structure:
CEM:
    sample(): return a sample from the current distribution, along with a weight
              corresponding to the likelihood-ratio wrt the original distribution.
        do_sample(curr_dist): do the sampling.              [IMPLEMENTED BY USER]
        get_weight(x): calculate the LR weight.
            likelihood_ratio(x).
                pdf(x, dist).                               [IMPLEMENTED BY USER]
    update(score):
        update list of scores.
        if there're enough samples and it's time to update the distribution:
            select_samples().
            update_sample_distribution(samples, weights).   [IMPLEMENTED BY USER]

    sample_batch(): sample and shuffle a whole batch together (if reference samples
                    are used, the shuffling can prevent them from concentrating in
                    the beginning of the batch).
        sample(): see above.
    update_batch(scores): update a whole batch of samples (can only be called after
                          sample_batch(), since the shuffled indices must be synced).
        update(score): see above.

Written by Ido Greenberg, 2022.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import copy, warnings


class CEM:
    def __init__(
            self,
            # initial distribution parameters
            phi0,

            #########   General   #########
            # number of samples between distribution updates
            # (0 = no updates)
            batch_size=0,
            # optimization mode: force ref_mode='none' and w_clip=1
            optim_mode=False,
            # clip IS weights to the range (1/w_clip, w_clip)
            # (0 = no clipping; 1 = no weights; 0<w_clip<1 will be inverted)
            w_clip=0,
            # experiment title
            title='CEM',

            #########   Sampling parameters   #########
            # number of reference samples per batch, taken from the original distribution
            n_orig_per_batch=None,

            #########   Update-step parameters   #########
            # where to estimate the reference-distribution from (see details below)
            ref_mode=None,
            # *absolute* objective: aim to sample X<=ref_thresh
            ref_thresh=None,
            # *quantile* objective (if ref_thresh==None): aim to sample ref_alpha-tail
            ref_alpha=0.05,
            # minimum samples to use for update step
            # (fraction for percent; int for absolute)
            min_batch_update=0.2,
            # when filtering R<q, add samples with R==q if update-samples are too few
            force_min_samples=True,
            # phi <- (1-soft_update)*new_phi + soft_update*current_phi
            soft_update=0,
            # quantile_estimate <- (1-soft_q_update)*estimate + soft_q_update*prev_estimate
            soft_q_update=0,
    ):
        self.title = title
        self.default_dist_titles = None
        self.default_samp_titles = None

        # An object defining the original distribution to sample from.
        # This can be any object (e.g., list of distribution parameters),
        # depending on the implementation of the inherited class.
        self.original_dist = phi0

        # Use the CEM for optimization rather than sampling.
        self.optim_mode = optim_mode
        if self.optim_mode:
            ref_mode = 'none'
            ref_thresh = None
            w_clip = 1

        # Number of samples to draw before updating distribution.
        # 0 is interpreted as infinity.
        self.batch_size = batch_size
        
        # When updating the distribution parameter phi, average the new phi
        # with the previous one, to make the update smoother.
        # Should be within [0,1) (0=only new phi; 1=only previous phi).
        self.soft_update = soft_update
        if self.soft_update > 0:
            try:
                null = self.soft_update*phi0 + (1-self.soft_update)*phi0
            except:
                print(phi0)
                warnings.warn(f'If soft_update>0, phi must be of a type that '
                               'supports addition and scalar-multiplication '
                               '(e.g., a numpy array).')
                raise

        # When updating the reference quantile, average the new estimate
        # with the previous one, to make the update smoother.
        # Should be within [0,1) (0=only new quantile; 1=only previous one).
        self.soft_q_update = soft_q_update

        # Clip the likelihood-ratio weights to the range [1/w_clip, w_clip].
        # If None or 0 - no clipping is done.
        if 0<w_clip<1:
            w_clip = 1/w_clip
        self.w_clip = w_clip

        # How to use reference scores to determine the threshold for the
        # samples selected for distribution update?
        # If ref_thresh is not None, then ref_thresh is the constant value of the threshold.
        # Otherwise, it is determined by ref_mode:
        # - 'none': ignore reference scores.
        # - 'train_ref': every batch, draw the first n=n_orig_per_batch samples
        #                from the original distribution instead of the updated one.
        #                then use quantile(batch_scores[:n]; ref_alpha).
        #                This is the default when w_clip>0, to reduce estimation bias.
        # - 'train': weighted quantile over the whole batch:
        #            quantile(batch_scores; ref_alpha, IS_weights).
        # - 'valid': use quantile(external_validation_scores; ref_alpha).
        #            in this case, update_ref_scores() must be called to
        #            feed reference scores before updating the distribution.
        # In CVaR optimization, ref_alpha would typically correspond to
        # the CVaR risk level.
        if ref_mode is None:
            # By default, we deduce the quantile internally from the train samples;
            # unless weights-clipping is used, in which case the quantile estimator
            # is skewed, thus we prefer estimation only from reference train samples.
            ref_mode = 'train' if w_clip==0 else 'train_ref'
        self.ref_thresh = ref_thresh
        self.ref_mode = ref_mode
        self.ref_alpha = ref_alpha

        # Number of samples to draw every batch from the original distribution
        # instead of the updated one. Either integer or ratio in (0,1).
        if n_orig_per_batch is None:
            n_orig_per_batch = 0.2 if self.ref_thresh is None and self.ref_mode!='none' else 0
        self.n_orig_per_batch = n_orig_per_batch
        if 0<self.n_orig_per_batch<1:
            self.n_orig_per_batch = int(self.n_orig_per_batch*self.batch_size)
        if self.batch_size < self.n_orig_per_batch:
            warnings.warn(f'samples per batch = {self.batch_size} < '
                          f'{self.n_orig_per_batch} = original-dist samples per batch')
            self.n_orig_per_batch = self.batch_size

        active_train_mode = \
            self.ref_mode == 'train_ref' and self.batch_size and self.ref_thresh is None
        if active_train_mode and self.n_orig_per_batch < 1:
            raise ValueError('"train_ref" reference mode must come with a positive '
                             'number of original-distribution samples per batch.')

        # In a distribution update, use from the current batch at least
        # min_batch_update * (batch_size - n_orig_per_batch)
        # samples. min_batch_update should be in the range (0,1).
        self.min_batch_update = min_batch_update
        if active_train_mode:
            self.min_batch_update *= 1 - self.n_orig_per_batch / self.batch_size
        # If multiple scores R equal the alpha quantile q, then the selected
        #  R<q samples may be strictly fewer than min_batch_update*batch_size.
        #  If force_min_samples==True, we fill in the missing entries from
        #  samples with R==q.
        self.force_min_samples = force_min_samples

        # State
        self.batch_count = 0
        self.sample_count = 0
        self.update_count = 0
        self.ref_scores = None
        self.ref_indices = set()
        self.batch_shuffled_indices = None

        # Data
        self.sample_dist = []  # n_batches
        self.sampled_data = [[]]  # n_batches x batch_size
        self.weights = [[]]  # n_batches x batch_size
        self.is_reference = [[]]  # n_batches x batch_size
        self.scores = [[]]  # n_batches x batch_size
        self.ref_quantile = []  # n_batches
        self.internal_quantile = []  # n_batches
        self.selected_samples = [[]]  # n_batches x batch_size
        self.n_update_samples = []  # n_batches

        self.reset()

    def reset(self):
        self.batch_count = 0
        self.sample_count = 0
        self.update_count = 0
        self.ref_scores = None
        if self.n_orig_per_batch > 0:
            self.ref_indices = set(np.random.choice(
                np.arange(self.batch_size), self.n_orig_per_batch, replace=False))

        self.sample_dist = [copy.copy(self.original_dist)]
        self.sampled_data = [[]]
        self.weights = [[]]
        self.is_reference = [[]]
        self.scores = [[]]
        self.ref_quantile = []
        self.internal_quantile = []
        self.selected_samples = [[]]
        self.n_update_samples = []

    def save(self, filename=None, base_path='./models', create_dir=True):
        if not os.path.exists(base_path) and create_dir:
            warnings.warn(f'Creating directory "{base_path}".')
            os.makedirs(base_path)
        if filename is None: filename = f'{base_path}/{self.title}'
        filename += '.cem'
        obj = (
            self.title, self.original_dist, self.batch_size, self.w_clip, self.ref_thresh,
            self.ref_mode, self.ref_alpha, self.n_orig_per_batch, self.min_batch_update,
            self.batch_count, self.sample_count, self.update_count, self.ref_scores,
            self.sample_dist, self.sampled_data, self.weights, self.is_reference,
            self.scores, self.ref_quantile, self.internal_quantile, self.selected_samples,
            self.n_update_samples, self.optim_mode
        )
        with open(filename, 'wb') as h:
            pkl.dump(obj, h)

    def load(self, filename=None, base_path='./models'):
        if filename is None: filename = f'{base_path}/{self.title}'
        filename += '.cem'
        with open(filename, 'rb') as h:
            obj = pkl.load(h)
        self.title, self.original_dist, self.batch_size, self.w_clip, self.ref_thresh, \
        self.ref_mode, self.ref_alpha, self.n_orig_per_batch, self.min_batch_update, \
        self.batch_count, self.sample_count, self.update_count, self.ref_scores, \
        self.sample_dist, self.sampled_data, self.weights, self.is_reference, \
        self.scores, self.ref_quantile, self.internal_quantile, self.selected_samples, \
        self.n_update_samples, self.optim_mode = obj

    def is_original_dist(self, shuffle=True):
        if not shuffle:
            return self.sample_count < self.n_orig_per_batch
        return self.sample_count in self.ref_indices

    ########   Sampling-related methods   ########

    def sample(self):
        orig_dist = self.is_original_dist()
        dist = self.sample_dist[0] if orig_dist else self.sample_dist[-1]
        x = self.do_sample(dist)
        w = self.get_weight(x, orig_dist)
        self.sampled_data[-1].append(x)
        self.weights[-1].append(w)
        self.is_reference[-1].append(orig_dist)
        self.sample_count += 1
        if 0 < self.batch_size < self.sample_count:
            warnings.warn(f'Drawn {self.sample_count}>{self.batch_size} samples '
                          f'without updating (only {self.update_count}<'
                          f'{self.batch_size} scores for update)')
        return x, w

    def sample_batch(self, n=None, shuffle=True):
        if n is None: n = self.batch_size
        samples = []
        for i in range(n):
            samples.append(self.sample())
        if shuffle:
            ids = list(np.random.permutation(n))
            samples = [samples[i] for i in ids]
            self.batch_shuffled_indices = [ids.index(i) for i in range(n)]
        else:
            self.batch_shuffled_indices = list(range(n))
        return samples

    def get_weight(self, x, use_original_dist=False):
        if use_original_dist:
            return 1

        lr = self.likelihood_ratio(x)
        if self.w_clip:
            lr = np.clip(lr, 1/self.w_clip, self.w_clip)
        return lr

    def likelihood_ratio(self, x, use_original_dist=False):
        if use_original_dist:
            return 1
        return self.pdf(x, self.sample_dist[0]) / \
               self.pdf(x, self.sample_dist[-1])

    def do_sample(self, dist):
        '''Given dist. parameters, return a sample drawn from the distribution.'''
        raise NotImplementedError()

    def pdf(self, x, dist):
        '''Given a sample x and distribution parameters dist, return P(x|dist).'''
        raise NotImplementedError()

    ########   Update-related methods   ########

    def update(self, score, save=False):
        self.scores[-1].append(score)
        self.update_count += 1

        if 0 < self.batch_size <= self.update_count:

            self.select_samples()
            samples = [self.sampled_data[-1][i] for i in range(self.sample_count)
                       if self.selected_samples[-1][i]]
            weights = [self.weights[-1][i] for i in range(self.sample_count)
                       if self.selected_samples[-1][i]]

            if len(samples) > 0:
                dist = self.update_sample_distribution(samples, weights)
                if self.soft_update > 0:
                    dist = self.soft_update * self.sample_dist[-1] + \
                           (1-self.soft_update) * dist
            else:
                dist = self.sample_dist[-1]
            self.sample_dist.append(dist)

            self.reset_batch()
            if save:
                filename = save if isinstance(save, str) else None
                self.save(filename)

    def update_batch(self, scores, save=False):
        if self.batch_shuffled_indices is None:
            raise RuntimeError('update_batch() can only be called after sample_batch().')
        n = len(scores)
        for i in range(n):
            self.update(scores[self.batch_shuffled_indices[i]], save)
        self.batch_shuffled_indices = None

    def reset_batch(self):
        self.sampled_data.append([])
        self.scores.append([])
        self.weights.append([])
        self.is_reference.append([])
        self.batch_count += 1
        self.sample_count = 0
        self.update_count = 0
        if self.n_orig_per_batch > 0:
            self.ref_indices = set(np.random.choice(
                np.arange(self.batch_size), self.n_orig_per_batch, replace=False))

    def select_samples(self):
        # Get internal quantile
        q_int = quantile(self.scores[-1], self.min_batch_update)

        # Get reference quantile from "external" data
        q_ref = -np.inf
        if self.ref_thresh is not None:
            q_ref = self.ref_thresh
        elif self.ref_mode == 'train_ref':
            ref_scores = [s for idx,s in enumerate(self.scores[-1])
                          if idx in self.ref_indices]
            q_ref = quantile(
                ref_scores, self.ref_alpha, estimate_underlying_quantile=True)
        elif self.ref_mode == 'train':
            q_ref = quantile(
                self.scores[-1], self.ref_alpha, w=self.weights[-1],
                estimate_underlying_quantile=True)
        elif self.ref_mode == 'valid':
            if self.ref_scores is None:
                warnings.warn('ref_mode=valid, but no '
                              'validation scores were provided.')
            else:
                q_ref = quantile(self.ref_scores, 100*self.ref_alpha,
                                 estimate_underlying_quantile=True)
        elif self.ref_mode == 'none':
            q_ref = -np.inf
        else:
            warnings.warn(f'Invalid ref_mode: {self.ref_mode}')

        # Soft-update reference quantile estimator
        if self.soft_q_update>0 and len(self.ref_quantile)>0:
            q_ref = self.soft_q_update * self.ref_quantile[-1] + \
                   (1 - self.soft_q_update) * q_ref

        # Take the max over the two
        self.internal_quantile.append(q_int)
        self.ref_quantile.append(q_ref)
        q = max(q_int, q_ref)

        # Select samples
        R = np.array(self.scores[-1])
        selection = R < q
        if self.force_min_samples:
            missing_samples = int(
                self.min_batch_update*self.batch_size - np.sum(selection))
            if missing_samples > 0:
                samples_to_add = np.where(R == q)[0]
                if missing_samples < len(samples_to_add):
                    samples_to_add = np.random.choice(
                        samples_to_add, missing_samples, replace=False)
                selection[samples_to_add] = True
        self.selected_samples.append(selection)
        self.n_update_samples.append(int(np.sum(selection)))

    def update_ref_scores(self, scores):
        self.ref_scores = scores

    def update_sample_distribution(self, samples, weights):
        '''Return the parameters of a distribution given samples.'''
        raise NotImplementedError()

    ########   Analysis-related methods   ########

    def get_data(self, dist_obj_titles=None, sample_dimension_titles=None,
                 exclude_last_batch=True):
        if dist_obj_titles is None:
            dist_obj_titles = self.default_dist_titles
        if sample_dimension_titles is None:
            sample_dimension_titles = self.default_samp_titles
        n_batches = self.batch_count + 1 - bool(exclude_last_batch)
        bs = self.batch_size
        if bs == 0: bs = self.sample_count

        # Batch-level data
        # Create a map from table-titles to distribution parameters.
        sd = self.sample_dist[:-1] if exclude_last_batch else self.sample_dist
        dist_objs = {}
        if isinstance(dist_obj_titles, str):
            dist_objs = {dist_obj_titles:sd}
        elif isinstance(dist_obj_titles, (tuple, list)):
            dist_objs = {t:[ds[i] for ds in sd]
                         for i,t in enumerate(dist_obj_titles)}

        d1_dict = dict(
            title=self.title,
            batch=np.arange(n_batches),
            ref_quantile=self.ref_quantile,
            internal_quantile=self.internal_quantile,
            n_update_samples=self.n_update_samples,
            update_samples_perc=100*np.array(self.n_update_samples)/bs,
        )
        for k,v in dist_objs.items():
            d1_dict[k] = v
        d1 = pd.DataFrame(d1_dict)

        # Sample-level data
        # Create a map from table-titles to sampled dimensions.
        samples = {}
        if isinstance(sample_dimension_titles, str):
            samples = {sample_dimension_titles:np.concatenate(self.sampled_data)}
        elif isinstance(sample_dimension_titles, (tuple, list)):
            sampled_data = self.sampled_data[:-1] if exclude_last_batch \
                else self.sampled_data
            samples = {t:[sample[i] for batch in sampled_data
                          for sample in batch]
                         for i,t in enumerate(sample_dimension_titles)}

        w, s, is_ref = self.weights, self.scores, self.is_reference
        if n_batches and exclude_last_batch:
            w, s, is_ref = self.weights[:-1], self.scores[:-1], self.is_reference[:-1]
        d2_dict = dict(
            title=self.title,
            batch=np.repeat(np.arange(n_batches), bs),
            sample_id=n_batches*list(range(bs)),
            selected=np.concatenate(self.selected_samples),
            weight=np.concatenate(w),
            is_ref=np.concatenate(is_ref),
            score=np.concatenate(s),
        )
        for k,v in samples.items():
            d2_dict[k] = v[:len(d2_dict['weight'])]
        d2 = pd.DataFrame(d2_dict)

        return d1, d2

    def show_sampled_scores(self, ax=None, ylab=None):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(5,3.5))[1]
            ax.grid(color='k', linestyle=':', linewidth=0.3)
        if ylab is None:
            ylab = 'score'

        # Prepare tail calculations & labels according to the settings
        #  (e.g., const threshold or quantile?)
        if self.ref_thresh is not None:
            cvar = lambda x, alpha: np.mean(x[x <= self.ref_thresh])
            cvar_lab = f'mean$\\{{x|x <= {self.ref_thresh}\\}}$'
            def wcvar(x, w, alpha):
                q = self.ref_thresh
                ids = x <= q
                x = x[ids]
                w = w[ids]
                return np.mean(x * w) / np.mean(w)
        elif self.ref_mode=='none':
            cvar = None
        else:
            cvar = lambda x, alpha: np.mean(np.sort(x)[:int(np.ceil(alpha*len(x)))])
            cvar_lab = f'CVaR$_{{{100*self.ref_alpha:.0f}\%}}$'
            def wcvar(x, w, alpha):
                q = quantile(x, alpha, w)
                ids = x <= q
                x = x[ids]
                w = w[ids]
                return np.mean(x * w) / np.mean(w)

        # Calculate reference mean & tail-mean
        c1, c2 = self.get_data()
        if not self.optim_mode:
            if self.n_orig_per_batch > 0:
                mean_orig = c2[c2.is_ref].groupby('batch').apply(lambda d: d.score.mean())
                ax.plot(mean_orig, label='Reference mean')
                if cvar is not None:
                    cvar_orig = c2[c2.is_ref].groupby('batch').apply(
                        lambda d: cvar(d.score.values,self.ref_alpha))
                    ax.plot(cvar_orig, label=f'Reference {cvar_lab:s}')
            else:
                if self.w_clip > 0:
                    warnings.warn('Reference distribution reconstruction may be inaccurate '
                                  'with clipped weights and without reference samples.')
                mean_orig = c2.groupby('batch').apply(
                    lambda d: np.mean(d.score*d.weight)/d.weight.mean())
                ax.plot(mean_orig, label='Reference mean (IS)')
                if cvar is not None:
                    cvar_orig = c2.groupby('batch').apply(
                        lambda d: wcvar(d.score.values,d.weight.values,self.ref_alpha))
                    ax.plot(cvar_orig, label=f'Reference {cvar_lab:s} (IS)')

        # Calculate sample mean & tail-mean
        mean_samp = c2[~c2.is_ref].groupby('batch').apply(lambda d: d.score.mean())
        ax.plot(mean_samp, label='Sample mean')
        if cvar is not None:
            cvar_samp = c2[~c2.is_ref].groupby('batch').apply(
                lambda d: cvar(d.score.values,self.ref_alpha))
            ax.plot(cvar_samp, label=f'Sample {cvar_lab:s}')

        ax.set_xlabel('iteration', fontsize=15)
        ax.set_ylabel(ylab, fontsize=15)
        ax.legend(fontsize=13)
        return ax

    def show_tail_level(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(5,3.5))[1]
            ax.grid(color='k', linestyle=':', linewidth=0.3)

        c1, c2 = self.get_data()

        if self.n_orig_per_batch > 0:
            tail_level = []
            for b in range(c2.batch.values[-1]+1):
                bb = c2[c2.batch==b]
                sample_mean = bb[~bb.is_ref].score.mean()
                refs = sorted(bb[bb.is_ref].score.values)
                ref_means = np.cumsum(refs) / np.arange(1,len(refs)+1)
                alphas = np.where(ref_means<=sample_mean)[0]
                alpha = 0 if len(alphas)==0 else 100 * alphas[-1] / len(ref_means)
                tail_level.append(alpha)

        else:
            if self.w_clip > 0:
                warnings.warn('Reference distribution reconstruction may be inaccurate '
                              'with clipped weights and without reference samples.')

            def wcvar(x, w, alpha):
                q = quantile(x, alpha, w)
                ids = x <= q
                x = x[ids]
                w = w[ids]
                return np.mean(x * w) / np.mean(w)

            tail_level = []
            for b in range(c2.batch.values[-1]+1):
                bb = c2[c2.batch==b]
                sample_mean = bb[~bb.is_ref].score.mean()
                x, w = bb.score.values, bb.weight.values
                refs = np.array([wcvar(x,w,alpha) for alpha in np.linspace(0,1,101)])
                alpha = np.where(refs<=sample_mean)[0][-1]
                tail_level.append(alpha)

        if self.ref_thresh is None and not self.optim_mode:
            ax.axhline(100*self.ref_alpha, color='k', linestyle='--')
        ax.plot(tail_level)
        ax.set_ylim((-1, 101))
        ax.set_xlabel('iteration', fontsize=15)
        ax.set_ylabel('tail level [%]', fontsize=15)
        ax.set_title(r'$\alpha$ | $mean$(sample)==$CVaR_{\alpha}$(ref)', fontsize=15)
        return ax

    def show_summary(self, axs=None, ylab=None):
        if self.optim_mode:
            return self.show_sampled_scores(axs, ylab)

        if axs is None:
            axs = plt.subplots(1, 2, figsize=(10,4))[1]
            for i in range(2):
                axs[i].grid(color='k', linestyle=':', linewidth=0.3)

        self.show_sampled_scores(axs[0], ylab)
        self.show_tail_level(axs[1])
        plt.tight_layout()
        return axs


def quantile(x, q, w=None, is_sorted=False, estimate_underlying_quantile=False):
    n = len(x)
    # If we estimate_underlying_quantile, we refer to min(x),max(x) not as
    #  quantiles 0,1, but rather as quantiles 1/(n+1),n/(n+1) of the
    #  underlying distribution from which x is sampled.
    if estimate_underlying_quantile and n > 1:
        q = q * (n+1)/(n-1) - 1/(n-1)
        q = np.clip(q, 0, 1)
    # Unweighted quantiles
    if w is None:
        return np.percentile(x, 100*q)
    # Weighted quantiles
    x = np.array(x)
    w = np.array(w)
    if not is_sorted:
        ids = np.argsort(x)
        x = x[ids]
        w = w[ids]
    w = np.cumsum(w) - 0.5*w
    w -= w[0]
    w /= w[-1]
    return np.interp(q, w, x)
