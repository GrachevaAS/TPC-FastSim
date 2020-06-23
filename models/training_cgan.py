import tensorflow as tf
import numpy as np
from tqdm import trange


def train(data, train_step_fn, loss_eval_fn, num_epochs, batch_size,
          train_writer=None, val_writer=None, callbacks=None):
    data_train, data_val, param_train, param_val = data
    for i_epoch in range(num_epochs):
        print("Working on epoch #{}".format(i_epoch), flush=True)

        tf.keras.backend.set_learning_phase(1)  # training

        shuffle_ids = np.random.permutation(len(data_train))
        losses_train = {}

        for i_sample in trange(0, len(data_train), batch_size):
            batch = data_train[shuffle_ids][i_sample:i_sample + batch_size]
            batch_param = param_train[shuffle_ids][i_sample:i_sample + batch_size]

            losses_train_batch = train_step_fn(batch, batch_param)
            for k, l in losses_train_batch.items():
                losses_train[k] = losses_train.get(k, 0) + l.numpy() * len(batch)
        losses_train = {k: l / len(data_train) for k, l in losses_train.items()}

        tf.keras.backend.set_learning_phase(0)  # testing

        losses_val = {k: l.numpy() for k, l in loss_eval_fn(data_val, param_val).items()}
        for f in callbacks:
            f(i_epoch)

        if train_writer is not None:
            with train_writer.as_default():
                for k, l in losses_train.items():
                    tf.summary.scalar(k, l, i_epoch)

        if val_writer is not None:
            with val_writer.as_default():
                for k, l in losses_val.items():
                    tf.summary.scalar(k, l, i_epoch)

        print("", flush=True)
        print("Train losses:", losses_train)
        print("Val losses:", losses_val)
