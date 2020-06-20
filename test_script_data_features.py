import os
from pathlib import Path
import argparse
import functools
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from data import preprocessing
from metrics import get_val_metric_v
from models.training_cgan import train
from models.baseline_10x10_cgan import BaselineModel


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256, required=False)
    parser.add_argument('--lr', type=float, default=1e-4, required=False)
    parser.add_argument('--num_disc_updates', type=int, default=3, required=False)
    parser.add_argument('--lr_schedule_rate', type=float, default=0.9998, required=False)
    parser.add_argument('--du_schedule_rate', type=float, default=1., required=False)
    parser.add_argument('--save_every', type=int, default=50, required=False)
    parser.add_argument('--num_epochs', type=int, default=10000, required=False)
    parser.add_argument('--latent_dim', type=int, default=32, required=False)
    parser.add_argument('--gpu_num', type=str, required=False)
    parser.add_argument('--kernel_init', type=str, default='glorot_uniform', required=False)
    parser.add_argument('--gp_lambda', type=float, default=1., required=False)
    parser.add_argument('--dropout_rate', type=float, default=0.02, required=False)
    parser.add_argument('--data_version', type=int, default=3, required=False)
    parser.add_argument('--gen_type', type=str, default='normal', required=False)
    parser.add_argument('--discrim_type', type=str, default='normal', required=False)
    parser.add_argument('--activation', type=str, default='relu', required=False)

    args = parser.parse_args()

    print("")
    print("----" * 10)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"    {k} : {v}")
    print("----" * 10)
    print("")

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if args.gpu_num is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    assert len(logical_devices) > 0, "Not enough GPU hardware devices available"

    model_path = Path('train_logs') / args.checkpoint_name / 'saved_models'
    model_path.mkdir(parents=True)
    with open(model_path.parent / 'arguments.txt', 'w+') as f:
        all_args = vars(args)
        f.write(''.join([f'--{argname}\n{argval}\n' for argname, argval in all_args.items()
                         if type(argval) != bool and argval is not None and argname != 'checkpoint_name'] +
                        [f'--{argname}\n' for argname, argval in all_args.items() if argval is True]
                        )
                )

    def save_model(step):
        if step % args.save_every == 0:
            print(f'Saving model on step {step} to {model_path}')
            model.generator.save(str(model_path.joinpath("generator_{:05d}.h5".format(step))))
            model.discriminator.save(str(model_path.joinpath("discriminator_{:05d}.h5".format(step))))

    params_dict = {2: ((39, 47), (267, 275)), 3: ((41, 47), (-7, 7))}
    pad_range, time_range = params_dict[args.data_version]
    data, features = preprocessing.read_csv_2d(version=f'data_v{args.data_version}',
                                               pad_range=pad_range, time_range=time_range)

    data_scaled = np.log10(1 + data).astype('float32')
    features = features.astype('float32')
    if args.data_version == 2:
        features = features[:, :1]

    Y_train, Y, X_train, X = train_test_split(data_scaled, features, test_size=0.3, random_state=42)
    Y_valid, Y_test, X_valid, X_test = train_test_split(Y, X, test_size=0.5, random_state=42)

    print("_" * 70)
    print("MEAN IS:", Y_train[Y_train > 0].mean())
    print('FEATURE MEAN:', X_train.mean(axis=0))
    print('FEATURE STD:', X_train.std(axis=0))
    print("_" * 70)

    def normalize_features(f, mean=X_train.mean(axis=0), std=X_train.std(axis=0)):
        return (f - mean) / std

    X_train, X_valid, X_test = normalize_features(X_train), normalize_features(X_valid), normalize_features(X_test)

    model = BaselineModel(kernel_init=args.kernel_init, lr=args.lr,
                          num_disc_updates=args.num_disc_updates, latent_dim=args.latent_dim,
                          gp_lambda=args.gp_lambda, dropout_rate=args.dropout_rate,
                          num_features=features.shape[1], shape=data.shape[1:],
                          generator_type=args.gen_type, discrim_type=args.discrim_type,
                          activation=args.activation)

    writer_train = tf.summary.create_file_writer(f'train_logs/{args.checkpoint_name}/train')
    writer_val = tf.summary.create_file_writer(f'train_logs/{args.checkpoint_name}/validation')

    def schedule_lr(step):
        model.disc_opt.lr.assign(model.disc_opt.lr * args.lr_schedule_rate)
        model.gen_opt.lr.assign(model.gen_opt.lr * args.lr_schedule_rate)
        with writer_val.as_default():
            tf.summary.scalar("discriminator learning rate", model.disc_opt.lr, step)
            tf.summary.scalar("generator learning rate", model.gen_opt.lr, step)

    def schedule_disc_updates(step):
        if 1000 - step % 1000 == 1:
            model.num_disc_updates = int(round(model.num_disc_updates * args.du_schedule_rate))
        with writer_val.as_default():
            tf.summary.scalar("num disc updates", model.num_disc_updates, step)

    from common import write_hist_summary, unscale
    metric_real = get_val_metric_v(unscale(Y_valid))
    write_hist_summary = functools.partial(write_hist_summary,
                                           save_every=args.save_every, writer=writer_val,
                                           model=model, sample=(X_valid, Y_valid), metrics_real=metric_real,
                                           )
    if args.data_version == 2:
        write_hist_summary = functools.partial(write_hist_summary, features_names=('crossing_angle',))

    train((Y_train, Y_valid, X_train, X_valid), model.training_step, model.calculate_losses,
          args.num_epochs, args.batch_size,
          train_writer=writer_train, val_writer=writer_val,
          callbacks=[write_hist_summary, save_model, schedule_lr, schedule_disc_updates]
          )


if __name__ == '__main__':
    main()
