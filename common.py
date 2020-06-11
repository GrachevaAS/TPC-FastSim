import numpy as np
import tensorflow as tf

import metrics
from metrics import make_metric_plots, make_histograms, calc_bin_stats


unscale = lambda x: 10 ** x - 1


def get_images(model, sample,
               return_raw_data=False, return_stats=False, calc_chi2=False,
               gen_more=None, batch_size=128,
               features_names=('crossing_angle', 'dip_angle', 'drift_length')):
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

    result = make_metric_plots(real, gen, features=features, return_raw_metrics=True, calc_chi2=calc_chi2)
    images = result[0]
    metrics_real, metrics_gen = result[1]
    if calc_chi2:
        chi2 = result[-1]

    images1 = make_metric_plots(real, gen1, features=features)

    img_amplitude = make_histograms(Y.flatten(), gen_scaled.flatten(), 'log10(amplitude + 1)', logy=True)

    result = [images, images1, img_amplitude]

    if return_raw_data:
        result += [(gen_features, gen)]

    if return_stats:
        stats_cond = [[] for _ in features_names]
        for i in range(X.shape[-1]):
            for j in range(metrics_real.shape[-1]):
                ksstats, rstats = calc_bin_stats(X[:, i], metrics_real[:, j], gen_features[:, i], metrics_gen[:, j])
                stats_cond[i].append((ksstats.mean(), rstats.mean()))
        stats_uncond = []
        for j in range(metrics_real.shape[-1]):
            ksstat = metrics.get_kolmogorov_smirnov_stat(metrics_real[:, j], metrics_gen[:, j])[0]
            rstat = metrics.get_rosenblatt_stat(metrics_real[:, j], metrics_gen[:, j])[0]
            stats_uncond.append((ksstat, rstat))
        result += [(np.array(stats_uncond), np.array(stats_cond))]

    if calc_chi2:
        result += [chi2]

    return result


def write_hist_summary(step, save_every, writer, model=None, sample=None,
                       features_names=('crossing_angle', 'dip_angle', 'drift_length')):
    assert model is not None and sample is not None
    if step % save_every == 0:
        images, images1, img_amplitude, (stats_uncond, stats_cond), chi2 = get_images(model, sample,
                                                                                  return_stats=True, calc_chi2=True,
                                                                                  features_names=features_names)
        with writer.as_default():
            tf.summary.scalar("chi2", chi2, step)
            for k, stat_name in enumerate(['KS stat', 'Rosenblatt stat']):
                for i, feature in enumerate(features_names):
                    tf.summary.scalar(f"{stat_name}: {feature} sum", stats_cond[i, :, k].sum(), step)
                for j in range(stats_cond.shape[-1]):
                    tf.summary.scalar(f"{stat_name}: {metrics._METRIC_NAMES[j]} sum", stats_cond[:, j, k].sum(), step)
                tf.summary.scalar(f"{stat_name}: sum of all", stats_cond[:, :, k].sum(), step)
                for j in range(len(stats_uncond)):
                    tf.summary.scalar(f"{stat_name}: {metrics._METRIC_NAMES[j]}", stats_uncond[j, k], step)

            for k, img in images.items():
                tf.summary.image(k, img, step)
            for k, img in images1.items():
                tf.summary.image("{} (amp > 1)".format(k), img, step)
            tf.summary.image("log10(amplitude + 1)", img_amplitude, step)
