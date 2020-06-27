import tensorflow as tf
import numpy as np


def get_generator_conv(activation, kernel_init, final_shape):
    padding = tuple((np.array(final_shape) % 4 == 0).astype(int).tolist())
    generator_conv = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.UpSampling2D(),  # 8x8 / 6x10 / 6x6

        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation=activation,
                               kernel_initializer=kernel_init),  # 6x6 / 4x8 / 4x4
        tf.keras.layers.UpSampling2D(),  # 12x12 / 8x16 / 8x8
        tf.keras.layers.ZeroPadding2D(padding=padding),  # / / 10x10

        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation=activation,
                               kernel_initializer=kernel_init),  # 10x10 / 6x14 / 8x8
        tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='valid', activation=tf.keras.activations.relu,
                               kernel_initializer=kernel_init),

        tf.keras.layers.Reshape(final_shape),
    ], name='generator_conv')
    return generator_conv


def get_generator_no_res(activation, kernel_init, final_shape):
    padding = tuple((np.array(final_shape) % 4 == 0).astype(int).tolist())
    generator_conv = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.UpSampling2D(),  # 8x8 / 6x10 / 6x6

        tf.keras.layers.Conv2D(filters=16, kernel_size=1, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=16, kernel_size=1, padding='same', activation=activation,
                               kernel_initializer=kernel_init),

        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.UpSampling2D(),  # 12x12 / 8x16 / 8x8
        tf.keras.layers.ZeroPadding2D(padding=padding),

        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation=activation,
                               kernel_initializer=kernel_init),  # 10x10 / 6x14 / 8x8
        tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='valid', activation=tf.keras.activations.relu,
                               kernel_initializer=kernel_init),

        tf.keras.layers.Reshape(final_shape),
    ], name='generator_conv')
    return generator_conv


def get_generator_residual(x, activation, kernel_init, final_shape):
    padding = tuple((np.array(final_shape) % 4 == 0).astype(int).tolist())
    block1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding='same', kernel_initializer=kernel_init),
        ])
    block2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=16, kernel_size=1, padding='same', kernel_initializer=kernel_init),
        ])

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init)(x)
    fx = block1(x)
    x = x + fx
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.UpSampling2D()(x)  # 8x8 / 6x10 / 6x6

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=1, padding='same', activation=activation,
                               kernel_initializer=kernel_init)(x)
    fx = block2(x)
    x = x + fx
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    final = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.UpSampling2D(),  # 12x12 / 8x16 / 8x8
        tf.keras.layers.ZeroPadding2D(padding=padding),

        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation=activation,
                               kernel_initializer=kernel_init),  # 10x10 / 6x14 / 8x8
        tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='valid', activation=tf.keras.activations.relu,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Reshape(final_shape),
    ])
    x = final(x)
    return x


def get_generator_conv_transposed(activation, kernel_init, final_shape):
    padding = tuple((np.array(final_shape) % 4 == 0).astype(int).tolist())
    generator_conv = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=(2, 2), padding='same', use_bias=False),  # 8x8 / 6x10 / 6x6

        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation=activation,
                               kernel_initializer=kernel_init),  # 6x6 / 4x8 / 4x4
        tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=(2, 2), padding='same', use_bias=False),  # 12x12 / 8x16 / 8x8
        tf.keras.layers.ZeroPadding2D(padding=padding),  # / / 10x10

        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation=activation,
                               kernel_initializer=kernel_init),  # 10x10 / 6x14 / 8x8
        tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='valid', activation=tf.keras.activations.relu,
                               kernel_initializer=kernel_init),

        tf.keras.layers.Reshape(final_shape),
    ], name='generator_conv')
    return generator_conv


gen_types = {
    'normal': get_generator_conv,
    'transposed': get_generator_conv_transposed,
    'no_residual': get_generator_no_res,
    'residual': get_generator_residual
}


def get_generator(activation, kernel_init, latent_dim, num_features, shape, type):
    input_random = tf.keras.Input(shape=(latent_dim,))
    input_params = tf.keras.Input(shape=(num_features,))
    x = tf.concat([input_random, input_params], axis=-1)

    input_shape = latent_dim + num_features
    inter_shape = ((np.array(shape) + 2) // 2 + 2) // 2
    init_linear = tf.keras.layers.Dense(units=4*np.prod(inter_shape), activation=activation, input_shape=(input_shape,))
    x = init_linear(x)
    x = tf.reshape(x, (-1, inter_shape[0], inter_shape[1], 4))
    get_conv_part = gen_types[type]
    if type == 'residual':
        x = get_conv_part(x, activation, kernel_init, final_shape=shape)
    else:
        generator_conv = get_conv_part(activation, kernel_init, final_shape=shape)
        x = generator_conv(x)
    generator = tf.keras.Model(
        inputs=[input_random, input_params],
        outputs=x,
        name='generator'
    )
    return generator


def get_discriminator_conv(activation, kernel_init, dropout_rate, final_shape):
    discriminator_conv = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init),  # 8x8
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.MaxPool2D(),  # 4x4

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.MaxPool2D(),  # 2x2

        tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='valid', activation=activation, kernel_initializer=kernel_init),  # 1x1
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Reshape((64 * np.prod(final_shape),))
    ], name='discriminator_conv')
    return discriminator_conv


def get_discriminator_no_res(activation, kernel_init, dropout_rate, final_shape):
    discriminator_conv = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),  # 8x8
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=16, kernel_size=1, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.MaxPool2D(),  # 4x4

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.MaxPool2D(),  # 2x2

        tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='valid', activation=activation, kernel_initializer=kernel_init),  # 1x1
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Reshape((64 * np.prod(final_shape),))
    ], name='discriminator_conv')
    return discriminator_conv


def get_discriminator_residual(x, activation, kernel_init, dropout_rate, final_shape):
    block1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=16, kernel_size=1, padding='same', kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
    ])
    block2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding='same', kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
    ])

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation=activation,
                               kernel_initializer=kernel_init)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    fx = block1(x)
    x = x + fx
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation,
                               kernel_initializer=kernel_init)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    fx = block2(x)
    x = x + fx
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    final = tf.keras.Sequential([
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='valid', activation=activation,
                               kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Reshape((64 * np.prod(final_shape),)),
    ])
    x = final(x)
    return x


def get_discriminator_conv_strided(activation, kernel_init, dropout_rate, final_shape):
    discriminator_conv = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', strides=(2, 2),
                               activation=activation, kernel_initializer=kernel_init),  # 4x4
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=(2, 2),
                               activation=activation, kernel_initializer=kernel_init),  # 2x2
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='valid', activation=activation, kernel_initializer=kernel_init),  # 1x1
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Reshape((64 * np.prod(final_shape),))
    ], name='discriminator_conv')
    return discriminator_conv


discrim_types = {
    'normal': get_discriminator_conv,
    'strided': get_discriminator_conv_strided,
    'no_residual': get_discriminator_no_res(),
    'residual': get_discriminator_residual
}


def get_discriminator(activation, kernel_init, dropout_rate, num_features, shape, type):
    input_img = tf.keras.Input(shape=shape)
    input_params = tf.keras.Input(shape=(num_features,))

    full_shape = (-1, shape[0], shape[1], 1)
    x = tf.reshape(input_img, full_shape)
    params_linear = tf.keras.layers.Dense(units=np.prod(shape), activation=activation, input_shape=(num_features,))
    params_x = tf.keras.layers.Dropout(dropout_rate)(params_linear(input_params))
    params_x = tf.reshape(params_x, full_shape)
    x = tf.concat([x, params_x], axis=-1)
    final_shape = (np.array(shape) - 2) // 4
    discriminator_conv_part = discrim_types[type]
    if type == 'residual':
        x = discriminator_conv_part(x, activation, kernel_init, dropout_rate, final_shape)
    else:
        discriminator_conv = discriminator_conv_part(activation, kernel_init, dropout_rate, final_shape)
        x = discriminator_conv(x)
    x = tf.concat([x, input_params], axis=-1)
    out_linear = tf.keras.layers.Dense(units=128, activation=activation)
    x = tf.keras.layers.Dropout(dropout_rate)(out_linear(x))
    last_linear = tf.keras.layers.Dense(units=1, activation=None)
    x = last_linear(x)
    discriminator = tf.keras.Model(
        inputs=[input_img, input_params],
        outputs=x,
        name='discriminator'
    )
    return discriminator


def disc_loss(d_real, d_fake):
    return tf.reduce_mean(d_fake - d_real)


def gen_loss(d_fake):
    return tf.reduce_mean(-d_fake)

activations_dict = {
    'relu': tf.keras.activations.relu,
    'leaky_relu': tf.nn.leaky_relu
}


class BaselineModel:
    def __init__(self, kernel_init, lr, num_disc_updates, latent_dim, gp_lambda, dropout_rate, num_features, shape,
                 generator_type, discrim_type, activation):
        self.activation = activations_dict[activation]
        self.disc_opt = tf.keras.optimizers.RMSprop(lr)
        self.gen_opt = tf.keras.optimizers.RMSprop(lr)
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.gp_lambda = gp_lambda
        self.num_disc_updates = num_disc_updates

        self.generator = get_generator(activation=self.activation, kernel_init=kernel_init, latent_dim=self.latent_dim,
                                       num_features=self.num_features, shape=shape, type=generator_type)
        self.discriminator = get_discriminator(activation=self.activation, kernel_init=kernel_init,
                                               dropout_rate=dropout_rate,
                                               num_features=self.num_features, shape=shape, type=discrim_type)

        self.step_counter = tf.Variable(0, dtype='int32', trainable=False)

    def make_fake(self, params):
        size = params.shape[0]
        return self.generator([tf.random.normal(shape=(size, self.latent_dim), dtype='float32'), params])

    def gradient_penalty(self, real, fake, params):
        alpha = tf.random.uniform(shape=[len(real), 1, 1])
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(interpolates)
            d_int = self.discriminator([interpolates, params])
        grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
        return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0)**2)

    @tf.function
    def calculate_losses(self, batch, batch_params):
        fake = self.make_fake(batch_params)
        d_real = self.discriminator([batch, batch_params])
        d_fake = self.discriminator([fake, batch_params])

        d_loss = disc_loss(d_real, d_fake) + self.gp_lambda * self.gradient_penalty(batch, fake, batch_params)
        g_loss = gen_loss(d_fake)
        return {'disc_loss': d_loss, 'gen_loss': g_loss}


    def disc_step(self, batch, batch_params):
        batch, batch_params = tf.convert_to_tensor(batch), tf.convert_to_tensor(batch_params)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(batch, batch_params)

        grads = t.gradient(losses['disc_loss'], self.discriminator.trainable_variables)
        self.disc_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return losses

    def gen_step(self, batch, batch_params):
        batch, batch_params = tf.convert_to_tensor(batch), tf.convert_to_tensor(batch_params)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(batch, batch_params)

        grads = t.gradient(losses['gen_loss'], self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        return losses

    @tf.function
    def training_step(self, batch, batch_params):
        if self.step_counter == self.num_disc_updates:
            result = self.gen_step(batch, batch_params)
            self.step_counter.assign(0)
        else:
            result = self.disc_step(batch, batch_params)
            self.step_counter.assign_add(1)
        return result
