import io

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import PIL


from plotting import _bootstrap_error


def get_rosenblatt_stat(real, gen):
    n, m = len(real), len(gen)
    real, gen = np.sort(real), np.sort(gen)
    common = np.concatenate([real, gen])
    mask = np.concatenate([np.zeros(n), np.ones(m)])
    mask_sorted = mask[np.argsort(common)]
    R = np.arange(n + m)[mask_sorted == 0]
    S = np.arange(n + m)[mask_sorted == 1]

    mseX = np.sum((R - np.arange(n) - 1) ** 2) / m
    mseY = np.sum((S - np.arange(m) - 1) ** 2) / n
    omega = (1 / 6 + mseX + mseY) / n / m - 2 / 3
    rstat = float(n * m) / (n + m) * omega
    return omega, rstat


def get_kolmogorov_smirnov_stat(real, gen):
    n, m = len(real), len(gen)
    common = np.concatenate([real, gen])
    mask = np.concatenate([np.zeros(n), np.ones(m)])
    sorted_common = np.argsort(common)
    sorted_mask = mask[sorted_common]
    cum_G = np.cumsum(sorted_mask)
    cum_F = np.cumsum(1 - sorted_mask)

    D_positive = np.max((np.arange(n) + 1) / n - cum_G[sorted_mask == 0] / m)
    D_negative = np.max((np.arange(m) + 1) / m - cum_F[sorted_mask == 1] / n)
    D = np.max([D_positive, D_negative])
    ks_stat = np.sqrt(float(n * m) / (n + m)) * D
    return D, ks_stat


def calc_bin_stats(feature_real, real, feature_gen, gen, bins=25, window_size=1):
    assert bins > 0
    bins = np.linspace(feature_real.min(), feature_real.max(), bins+1)
    bounds = (min(feature_real.min(), feature_gen.min()), max(feature_real.max(), feature_gen.max()))
    bins[0] = bounds[0]
    bins[-1] = bounds[1]

    cats_real = (feature_real[:, np.newaxis] < bins[np.newaxis, 1:]).argmax(axis=1)
    cats_real[feature_real == np.max(feature_real)] = len(bins) - 2
    cats_gen = (feature_gen[:, np.newaxis] < bins[np.newaxis, 1:]).argmax(axis=1)
    cats_gen[feature_gen == np.max(feature_gen)] = len(bins) - 2

    counts = [Counter(cats_real)[c] for c in range(len(bins) - 1)]
    assert np.all(np.array(counts) > 30)
    counts = [Counter(cats_gen)[c] for c in range(len(bins) - 1)]
    assert np.all(np.array(counts) > 30)

    def stats(x1, x2):
        return (
            get_kolmogorov_smirnov_stat(x1, x2)[0],
            get_rosenblatt_stat(x1, x2)[0]
        )

    ksstats, rstats, bin_centers = np.array([
        stats(
            gen[(cats_gen >= left) & (cats_gen < right)],
            real[(cats_real >= left) & (cats_real < right)]
        ) + ((bins[left] + bins[right]) / 2,)
        for left, right in zip(
            range(len(bins) - window_size),
            range(window_size, len(bins))
        )
    ]).T

    return ksstats, rstats


def calc_bin_stats_multidimensional(feature_real, real, feature_gen, gen, nbins=4):
    assert nbins > 0
    assert feature_real.ndim == 2 and feature_gen.ndim == 2
    nfeatures = feature_real.shape[1]
    assert nfeatures == 3  # TODO
    cats_real, cats_gen = [], []

    for i in range(nfeatures):
        bins = np.linspace(feature_real[:, i].min(), feature_real[:, i].max(), nbins + 1)
        bounds_ = (min(feature_real[:, i].min(), feature_gen[:, i].min()),
                   max(feature_real[:, i].max(), feature_gen[:, i].max()))
        bins[0] = bounds_[0]
        bins[-1] = bounds_[1]

        cats_real_ = (feature_real[:, i][:, np.newaxis] < bins[np.newaxis, 1:]).argmax(axis=1)
        cats_real_[feature_real[:, i] == bounds_[1]] = len(bins) - 2
        cats_gen_ = (feature_gen[:, i][:, np.newaxis] < bins[np.newaxis, 1:]).argmax(axis=1)
        cats_gen_[feature_gen[:, i] == bounds_[1]] = len(bins) - 2

        cats_real.append(cats_real_)
        cats_gen.append(cats_gen_)

    cats_real = np.stack(cats_real).T
    counter = np.zeros(tuple([nbins] * nfeatures))
    for mdbin in cats_real:
        counter[tuple(mdbin)] += 1
    assert ((counter > 30).all())

    cats_gen = np.stack(cats_gen).T
    counter = np.zeros(tuple([nbins] * nfeatures))
    for mdbin in cats_gen:
        counter[tuple(mdbin)] += 1
    assert ((counter > 30).all())

    # only 3 dimensions
    x, y, z = np.mgrid[0:nbins:1, 0:nbins:1, 0:nbins:1]
    grid = np.stack([x, y, z], axis=-1).reshape(-1, nfeatures)

    ksstats = np.array([get_kolmogorov_smirnov_stat(
        real[(cats_real == xyz).all(axis=-1)],
        gen[(cats_gen == xyz).all(axis=-1)]
    )[0] for xyz in grid]
    )
    return ksstats


_METRIC_NAMES = ['Mean0', 'Mean1', 'Sigma0^2', 'Sigma1^2', 'Cov01', 'Sum']


def get_val_metric_v(imgs):
    """Returns a vector of gaussian fit results to the image.
    The components are: [mu0, mu1, sigma0^2, sigma1^2, covariance, integral]
    """
    assert imgs.ndim == 3, 'get_val_metric_v: Wrong images dimentions'
    assert (imgs >= 0).all(), 'get_val_metric_v: Negative image content'
    assert (imgs > 0).any(axis=(1, 2)).all(), 'get_val_metric_v: some images are empty'
    imgs_n = imgs / imgs.sum(axis=(1, 2), keepdims=True)
    mu = np.fromfunction(
        lambda i, j: (imgs_n[:,np.newaxis,...] * np.stack([i, j])[np.newaxis,...]).sum(axis=(2, 3)),
        shape=imgs.shape[1:]
    )

    cov = np.fromfunction(
        lambda i, j: (
            (imgs_n[:,np.newaxis,...] * np.stack([i * i, j * j, i * j])[np.newaxis,...]).sum(axis=(2, 3))
        ) - np.stack([mu[:,0]**2, mu[:,1]**2, mu[:,0] * mu[:,1]]).T,
        shape=imgs.shape[1:]
    )

    return np.concatenate([mu, cov, imgs.sum(axis=(1, 2))[:,np.newaxis]], axis=1)


def make_histograms(data_real, data_gen, title, figsize=(8, 8), n_bins=100, logy=False):
    if title == 'Sum':
        data_real = data_real[data_real < 5e+4]
        data_gen = data_gen[data_gen < 5e+4]
    l = min(data_real.min(), data_gen.min())
    r = max(data_real.max(), data_gen.max())
    bins = np.linspace(l, r, n_bins + 1)
    
    fig = plt.figure(figsize=figsize)
    plt.hist(data_real, bins=bins, density=True, label='real')
    plt.hist(data_gen , bins=bins, density=True, label='generated', alpha=0.7)
    if logy:
        plt.yscale('log')
    plt.legend()
    plt.title(title)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    img = PIL.Image.open(buf)
    return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[0], img.size[1], -1)


def make_metric_plots(images_real, images_gen, metric_real=None, metric_gen=None, features=None,
                      return_raw_metrics=False, calc_chi2=False, window_size=10):
    plots = {}
    if calc_chi2:
        chi2 = 0

    try:
        if metric_real is None:
            metric_real = get_val_metric_v(images_real)
        if metric_gen is None:
            metric_gen = get_val_metric_v(images_gen)
    
        plots.update({name: make_histograms(real, gen, name)
                      for name, real, gen in zip(_METRIC_NAMES, metric_real.T, metric_gen.T)})

        if features is not None:
            for feature_name, (feature_real, feature_gen) in features.items():
                for metric_name, real, gen in zip(_METRIC_NAMES, metric_real.T, metric_gen.T):
                    name = f'{metric_name} vs {feature_name}'
                    if calc_chi2 and (metric_name != "Sum"):
                        plots[name], chi2_i = make_trend(feature_real, real, feature_gen, gen,
                                                         name, calc_chi2=True, window_size=window_size)
                        chi2 += chi2_i
                    else:
                        plots[name] = make_trend(feature_real, real,
                                                 feature_gen, gen, name)

    except AssertionError as e:
        print(f"WARNING! Assertion error ({e})")
        metric_real, metric_gen = np.array([]), np.array([])

    result = [plots]

    if return_raw_metrics:
        result += [(metric_real, metric_gen)]

    if calc_chi2:
        result += [chi2]

    if len(result) == 1:
        return result[0]
    return result


def calc_trend(x, y, do_plot=True, bins=100, window_size=10, issum=False, **kwargs):
    assert x.ndim == 1, 'calc_trend: wrong x dim'
    assert y.ndim == 1, 'calc_trend: wrong y dim'

    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.7

    if isinstance(bins, int):
        bins = np.linspace(np.min(x), np.max(x), bins + 1)
    sel = (x >= bins[0])
    x, y = x[sel], y[sel]
    cats = (x[:,np.newaxis] < bins[np.newaxis,1:]).argmax(axis=1)
    
    def stats(arr):
        return (
            arr.mean(),
            arr.std() / (len(arr) - 1)**0.5,
            arr.std(),
            _bootstrap_error(arr, np.std)
        )
    
    mean, mean_err, std, std_err, bin_centers = np.array([
        stats(
            y[(cats >= left) & (cats < right)]
        ) + ((bins[left] + bins[right]) / 2,) for left, right in zip(
            range(len(bins) - window_size),
            range(window_size, len(bins))
        )
    ]).T

    if do_plot:
        mean_p_std_err = (mean_err**2 + std_err**2)**0.5
        plt.fill_between(bin_centers, mean - mean_err, mean + mean_err, **kwargs)
        kwargs['alpha'] *= 0.5
        kwargs = {k : v for k, v in kwargs.items() if k != 'label'}
        plt.fill_between(bin_centers, mean - std - mean_p_std_err, mean - std + mean_p_std_err, **kwargs)
        plt.fill_between(bin_centers, mean + std - mean_p_std_err, mean + std + mean_p_std_err, **kwargs)
        kwargs['alpha'] *= 0.25
        plt.fill_between(bin_centers, mean - std + mean_p_std_err, mean + std - mean_p_std_err, **kwargs)
        if issum:
            mu_min, mu_max = (mean - mean_err).min(), (mean + mean_err).max()
            plt.ylim(max((mean - std - mean_p_std_err).min(), mu_min - 2*(mu_max - mu_min)),
                     min((mean + std + mean_p_std_err).max(), mu_max + 2*(mu_max - mu_min)))

    return (mean, std), (mean_err, std_err)


def make_trend(feature_real, real, feature_gen, gen, name, calc_chi2=False, figsize=(8, 8), window_size=10):
    feature_real = feature_real.squeeze()
    feature_gen = feature_gen.squeeze()
    real = real.squeeze()
    gen = gen.squeeze()

    bounds = (min(feature_real.min(), feature_gen.min()), max(feature_real.max(), feature_gen.max()))
    bins = np.linspace(bounds[0], bounds[1], 100)

    fig = plt.figure(figsize=figsize)
    calc_trend(feature_real, real, bins=bins, label='real', color='blue', window_size=window_size, issum=name[:3] == 'Sum')
    calc_trend(feature_gen, gen, bins=bins, label='generated', color='red', window_size=window_size, issum=name[:3] == 'Sum')
    plt.legend()
    plt.title(name)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    img = PIL.Image.open(buf)
    img_data = np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[0], img.size[1], -1)

    if calc_chi2:
        bins = np.linspace(bounds[0], bounds[1], 20)
        (real_mean, real_std), (real_mean_err, real_std_err) = calc_trend(feature_real, real,
                                                                          do_plot=False, bins=bins, window_size=1)
        (gen_mean, gen_std), (gen_mean_err, gen_std_err) = calc_trend(feature_gen, gen,
                                                                          do_plot=False, bins=bins, window_size=1)
        gen_upper = gen_mean + gen_std
        gen_lower = gen_mean - gen_std
        gen_err2 = gen_mean_err**2 + gen_std_err**2

        real_upper = real_mean + real_std
        real_lower = real_mean - real_std
        real_err2 = real_mean_err**2 + real_std_err**2

        chi2 = (
            ((gen_upper - real_upper)**2 / (gen_err2 + real_err2)).sum() +
            ((gen_lower - real_lower)**2 / (gen_err2 + real_err2)).sum()
        )

        return img_data, chi2
    
    return img_data
