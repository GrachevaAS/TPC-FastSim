import numpy as np
import tensorflow as tf

import metrics
from metrics import make_metric_plots, make_histograms, calc_bin_stats


def unscale(x):
    return 10 ** x - 1


def get_images(model, sample, features_names,
               metrics_real=None,
               return_raw_data=False, return_raw_metrics=False, calc_chi2=False,
               gen_more=None,
               batch_size=128):
    X, Y = sample
    assert X.ndim == 2
    assert X.shape[1] == len(features_names)

    if gen_more is None:
        gen_features = X
    else:
        gen_features = np.tile(
            X,
            [gen_more] + [1] * (X.ndim - 1)
        )
    gen_scaled = np.concatenate([
        model.make_fake(gen_features[i:i + batch_size]).numpy()
        for i in range(0, len(gen_features), batch_size)
    ], axis=0)
    real = unscale(Y)
    gen = unscale(gen_scaled)
    gen[gen < 0] = 0
    gen1 = np.where(gen < 1., 0, gen)

    features = {feature: (X[:, i], gen_features[:, i]) for i, feature in enumerate(features_names)}
    if len(features_names) >= 3:
        features['time_bin_fraction'] = (X[:, 2] % 1, gen_features[:, 2] % 1)

    result = make_metric_plots(real, gen, metric_real=metrics_real, features=features, return_raw_metrics=True, calc_chi2=calc_chi2)
    images = result[0]
    metrics_real, metrics_gen = result[1]
    if calc_chi2:
        chi2 = result[-1]

    images1 = make_metric_plots(real, gen1, metric_real=metrics_real, features=features)

    img_amplitude = make_histograms(Y.flatten(), gen_scaled.flatten(), 'log10(amplitude + 1)', logy=True)

    result = [images, images1, img_amplitude]

    if return_raw_data:
        result += [(gen_features, gen)]

    if return_raw_metrics:
        result += [(metrics_real, metrics_gen)]

    if calc_chi2:
        result += [chi2]

    return result


def write_hist_summary(step, save_every, writer, features_names,
                       model=None, sample=None, metrics_real=None, nbins=4):
    assert model is not None and sample is not None
    X, Y = sample
    if step % save_every == 0:
        images, images1, img_amplitude, (gen_features, gen), (metrics_real, metrics_gen), chi2 = get_images(
            model, sample, metrics_real=metrics_real, features_names=features_names,
            return_raw_data=True, return_raw_metrics=True,
            calc_chi2=True
        )
        stats_uncond, stats_cond = get_ks_and_rstat(X, gen_features, metrics_real, metrics_gen,
                                                    features_names=features_names)
        stats_md = get_ksstat_multidimensional(X, gen_features, metrics_real, metrics_gen, nbins=nbins)

        with writer.as_default():
            tf.summary.scalar("chi2", chi2, step)
            if stats_cond.shape[-1] != 0:
                for k, stat_name in enumerate(['ks stat', 'rosenblatt stat']):
                    for i, feature in enumerate(features_names):
                        tf.summary.scalar(f"{stat_name}: {feature} multi dimentional", stats_md.sum(), step)
                        tf.summary.scalar(f"{stat_name}: {feature} total sum",
                                          stats_cond[i, :, k].sum() / len(features_names), step)
                        tf.summary.scalar(f"{stat_name}: total unconditional", stats_uncond[:, k].sum(), step)
                    for j in range(stats_cond.shape[1]):
                        tf.summary.scalar(f"{stat_name}: {metrics._METRIC_NAMES[j]} total",
                                          stats_cond[:, j, k].sum(), step)
                    for j in range(stats_uncond.shape[0]):
                        tf.summary.scalar(f"{stat_name}: {metrics._METRIC_NAMES[j]}", stats_uncond[j, k], step)

            for k, img in images.items():
                tf.summary.image(k, img, step)
            for k, img in images1.items():
                tf.summary.image("{} (amp > 1)".format(k), img, step)
            tf.summary.image("log10(amplitude + 1)", img_amplitude, step)


def get_ks_and_rstat(real_features, gen_features, metrics_real, metrics_gen, features_names, select_max=False):
    stats_cond = [[] for _ in range(len(features_names))]
    for i in range(len(features_names)):
        for j in range(metrics_gen.shape[-1]):
            ksstats, rstats = calc_bin_stats(
                real_features[:, i], metrics_real[:, j], gen_features[:, i], metrics_gen[:, j]
            )
            if select_max:
                stats_cond[i].append((ksstats.max(), rstats.max()))
            else:
                stats_cond[i].append((ksstats.mean(), rstats.mean()))
    stats_uncond = []
    for j in range(metrics_gen.shape[-1]):
        ksstat = metrics.get_kolmogorov_smirnov_stat(metrics_real[:, j], metrics_gen[:, j])[0]
        rstat = metrics.get_rosenblatt_stat(metrics_real[:, j], metrics_gen[:, j])[0]
        stats_uncond.append((ksstat, rstat))
    return np.array(stats_uncond), np.array(stats_cond)


def get_ksstat_multidimensional(real_features, gen_features, metrics_real, metrics_gen, nbins=4):
    stats_cond = [metrics.calc_bin_stats_multidimensional(
            real_features, metrics_real[:, j], gen_features, metrics_gen[:, j], nbins=nbins
        ).mean() for j in range(metrics_real.shape[-1])
    ]
    return np.array(stats_cond)
