import numpy as np
import tensorflow as tf

from metrics import make_metric_plots, make_histograms

unscale = lambda x: 10 ** x - 1


def get_images(model=None, sample=None, return_raw_data=False, calc_chi2=False, gen_more=None, batch_size=128):
    assert model is not None and sample is not None
    X, Y = sample
    assert X.ndim == 2
    assert X.shape[1] == 3

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

    features = {
        'crossing_angle': (X[:, 0], gen_features[:, 0]),
        'dip_angle': (X[:, 1], gen_features[:, 1]),
        'drift_length': (X[:, 2], gen_features[:, 2]),
        'time_bin_fraction': (X[:, 2] % 1, gen_features[:, 2] % 1),
    }

    images = make_metric_plots(real, gen, features=features, calc_chi2=calc_chi2)
    if calc_chi2:
        images, chi2 = images

    images1 = make_metric_plots(real, gen1, features=features)

    img_amplitude = make_histograms(Y.flatten(), gen_scaled.flatten(), 'log10(amplitude + 1)', logy=True)

    result = [images, images1, img_amplitude]

    if return_raw_data:
        result += [(gen_features, gen)]

    if calc_chi2:
        result += [chi2]

    return result


def write_hist_summary(step, save_every, writer, get_images=get_images):
    if step % save_every == 0:
        images, images1, img_amplitude, chi2 = get_images(calc_chi2=True)

        with writer.as_default():
            tf.summary.scalar("chi2", chi2, step)

            for k, img in images.items():
                tf.summary.image(k, img, step)
            for k, img in images1.items():
                tf.summary.image("{} (amp > 1)".format(k), img, step)
            tf.summary.image("log10(amplitude + 1)", img_amplitude, step)
