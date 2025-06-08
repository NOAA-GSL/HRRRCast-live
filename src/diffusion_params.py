import numpy as np
import tensorflow as tf

USE_DIFFUSION = True

# number of steps in the forward phase
NUM_DIFFUSION_STEPS = 200
NUM_INFERENCE_STEPS = 50
INFERENCE_STEPS = range(
    0,
    NUM_DIFFUSION_STEPS,
    NUM_DIFFUSION_STEPS // NUM_INFERENCE_STEPS,
)


# linear beta schedule
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)


# quadratic beta schedule
def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


# sigmoid beta schedule
def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = np.linspace(-6, 6, timesteps)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    return sigmoid(betas) * (beta_end - beta_start) + beta_start


# cosine beta schedule
def cosine_beta_schedule(timesteps, s=0.008):
    beta_start = 0.0001
    beta_end = 0.9999

    # Define the function for alpha_bar (cumulative product of (1 - beta))
    def alpha_bar_fn(t):
        return np.cos((t / timesteps + s) / (1 + s) * np.pi / 2) ** 2

    # Calculate alpha_bar for each timestep
    alphas_bar = np.array([alpha_bar_fn(t) for t in range(timesteps + 1)])

    # Compute betas from alpha_bar
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = np.clip(betas, beta_start, beta_end)

    return betas


# create a fixed BETA schedule
BETA = cosine_beta_schedule(NUM_DIFFUSION_STEPS)

# constants in the reparameterization trick
ALPHA = 1 - BETA
ALPHA_BAR = np.cumprod(ALPHA, 0)
SQRT_ALPHA_BAR = np.sqrt(ALPHA_BAR)
SQRT_ONE_MINUS_ALPHA_BAR = np.sqrt(1 - ALPHA_BAR)


# generate random timestamps between [0,T)
def generate_timestamps(num):
    return np.random.randint(1, NUM_DIFFUSION_STEPS, size=(num,))


# forward noise
def forward_noise(x_0, t):
    epsilon = tf.random.normal(shape=x_0.shape)
    SQRT_ALPHA_BAR_t = SQRT_ALPHA_BAR[t].reshape((-1, 1, 1, 1))
    SQRT_ONE_MINUS_ALPHA_BAR_t = SQRT_ONE_MINUS_ALPHA_BAR[t].reshape((-1, 1, 1, 1))
    x_t = x_0 * SQRT_ALPHA_BAR_t + epsilon * SQRT_ONE_MINUS_ALPHA_BAR_t
    return x_t


# compute epsilon given x_0
def compute_epsilon(x_t, x_0, t):
    SQRT_ALPHA_BAR_t = SQRT_ALPHA_BAR[t].reshape((-1, 1, 1, 1))
    SQRT_ONE_MINUS_ALPHA_BAR_t = SQRT_ONE_MINUS_ALPHA_BAR[t].reshape((-1, 1, 1, 1))
    epsilon = (x_t - x_0 * SQRT_ALPHA_BAR_t) / SQRT_ONE_MINUS_ALPHA_BAR_t
    return epsilon


# ddpm
def ddpm(x_t, pred_noise, t_, seed=0):
    t = INFERENCE_STEPS[t_ - 1]

    ALPHA_t = ALPHA[t]
    ALPHA_BAR_t = ALPHA_BAR[t]

    eps_coef = (1 - ALPHA_t) / (1 - ALPHA_BAR_t) ** 0.5
    mean = (1 / (ALPHA_t**0.5)) * (x_t - eps_coef * pred_noise)

    var = BETA[t] if t > 0 else 0
    tf.random.set_seed(seed * 1000 + t)
    batch_size = x_t.shape[0]
    z = tf.random.normal(shape=tf.shape(x_t[0, :, :, :]))
    z = tf.tile(tf.expand_dims(z, axis=0), [batch_size, 1, 1, 1])

    return mean + (var**0.5) * z


# ddim
def ddim(x_t, pred_noise, t_, seed=0, sigma_t=0):
    t = INFERENCE_STEPS[t_]
    tm1 = INFERENCE_STEPS[t_ - 1]

    ALPHA_BAR_t = ALPHA_BAR[t]
    ALPHA_BAR_tm1 = ALPHA_BAR[tm1]

    pred = (x_t - ((1 - ALPHA_BAR_t) ** 0.5) * pred_noise) / (ALPHA_BAR_t**0.5)
    pred = (ALPHA_BAR_tm1**0.5) * pred + ((1 - ALPHA_BAR_tm1 - (sigma_t**2)) ** 0.5) * pred_noise

    if sigma_t > 0:
        tf.random.set_seed(seed * 1000 + t)
        batch_size = x_t.shape[0]
        z = tf.random.normal(shape=tf.shape(x_t[0, :, :, :]))
        z = tf.tile(tf.expand_dims(z, axis=0), [batch_size, 1, 1, 1])
        pred = pred + (sigma_t * z)

    return pred
