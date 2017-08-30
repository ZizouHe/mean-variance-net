# mean-variance network

This is a neural network structure to measure the uncertainty of our prediction. Specifically, we use two networks, mean and variance, to model the uncertainty in classification problem.

## 1. Model

### 1.1 Main Idea

Write our likelihood function in the following form:

$$
p(y | \mu; \sigma^2) = \int_{-\infty}^{\infty} p(y | \theta) \; p(\theta | \mu; \sigma^2) \; d\theta
$$

where $y$ is a Bernoulli with $p(y = 1 | x) = \theta$, $p(y = -1 | x) = 1 - \theta$ and $\theta$ follows a logit normal distribution with parameters $\mu(x), \sigma^2(x)$ given by:

$$
ln\left(\frac{\theta}{1 - \theta}\right) \sim N\left(\mu; \sigma^2\right)
$$

Therefore, the integral we are handling here has the following form:

$$
\mathscr{L}\left(\mu, \sigma^2 | y\right) = \int_{-\infty}^{\infty}  \frac{1}{1 + \mathrm{e}^{-yx}}\; \frac{1}{\sqrt{2\pi}\sigma}\mathrm{e}^{-\frac{(x - \mu)^2}{2\sigma^2}} \; dx
$$

### 1.2 Computation

One way to estimate the integral above efficiently is Monte-Carlo simulation, which is an unbiased way. We can do a Monte-Carlo from a standard normal distribution to calculate:
$$
\mathscr{L}\left(\mu(\cdot), \sigma^2(\cdot) | y\right) = E\left(\frac{1}{1 + \mathrm{e}^{-yx}} |\; x \sim N\left[\mu; \sigma^2\right]\right)
$$

Combine samples $\left\{(x_i, y_i)\right\}$ together, our cost function is the negative loglikelihood:
$$
\mathscr{C}\left(\mu(\cdot), \sigma^2(\cdot)\right | y) = \frac{1}{N}\sum_{i = 1}^N ln \left(\int_{-\infty}^{\infty}  \frac{1}{1 + \mathrm{e}^{-y_it}}\; \frac{1}{\sqrt{2\pi}\,\sigma(x_i)}exp\left\{-\frac{\left[t - \mu(x_i)\right]^2}{2\,\sigma^2(x_i)}\right\} \; dt\right)
$$

The derivative of weights and biases for two nets $w_{\mu}, w_{\sigma}$ are:

$$
\frac{\partial \mathscr{C}}{\partial w_\mu} = \frac{1}{K}\sum_{i = 1}^K \frac{E\left\{\frac{1}{1 + \mathrm{e}^{-y_it}} \frac{t - \mu_(x_i)}{\sigma^2(x_i)}|\; t \sim N\left[\mu(x_i), \sigma^2(x_i)\right]\right\}}{E\left\{\frac{1}{1 + \mathrm{e}^{-y_it}} |\; t \sim N\left[\mu(x_i), \sigma^2(x_i)\right]\right\}} \frac{\partial \mu(x_i)}{\partial w_\mu}
$$

$$
\frac{\partial \mathscr{C}}{\partial w_\sigma} = \frac{1}{K}\sum_{i = 1}^K \frac{E\left\{\frac{1}{1 + \mathrm{e}^{-y_it}} \left[\frac{\left(t - \mu_(x_i)\right)^2}{\sigma^3(x_i)} \, - \frac{1}{\sigma(x_i)}\right]|\; t \sim N\left[\mu(x_i), \sigma^2(x_i)\right]\right\}}{E\left\{\frac{1}{1 + \mathrm{e}^{-y_it}} |\; t \sim N\left[\mu(x_i), \sigma^2(x_i)\right]\right\}} \frac{\partial \sigma(x_i)}{\partial w_\sigma}
$$

here $k$ denotes the mini-batch size. The expectations above can be approximated by MC simulation easily.  

### 1.3 Problems

- Maximum likelihood estimation over $μ_\theta (x)$ and $\sigma^2_\theta (x)$ might overfit

- Still suffer the expensive computational cost, and given the Monte-Carlo simulation, the training procedure may take longer than normal gradient descent with closed form derivatives. 

- Variance net should be designed carefully so that variances do not have negative value. 

- Intialization is also important here, especially for variance net, if not carefully designed, the variance net output will go wild from the begining. 

## 2. Calibration Method of Probability Estimation

### 2.1 Main Idea

Given the fact that deep neural networks are not well-calibrated, we can find some calibration methods to give a better prediction on classification probability $p$. A proper score rule should be used here, and negative log-likelihood happens to be one. We just need a little modification on training process: insted of using classification error as criterion on validation set, we use NLL loss to determine hyperparameters. 

Several criterions can be used to justify the performance of our calibration methods, as described above:

1. Reliability Diagrams

2. Expected Calibration Error (ECE)

3. Maximum Calibration Error (MCE)

### 2.2 Problem

- How to estimate the performance of variance estimation has yet undetermined. Especially in the case of classification, we cannot observe the probability directly.

## 3. Estimation Variance

### 3.1 Main Idea

Following (Nix and Weigend, 1994), we use a network that outputs two values in the final layer, corresponding to the predicted mean $\mu(x)$ and variance $\sigma^2(x)$. We ensemble $M$ deep neural networks together, output $\mu_1, ..., \mu_M$, $\sigma^2_1(x), ..., \sigma^2_M(x)$. Therefore, the estimation of mean and variance is given by:
$$
\hat{\mu}(x) = \frac{1}{M} \sum_{i = 1}^M \mu_i(x)
$$
$$
\hat{\sigma}^2(x) = \frac{1}{M} \sum_{i = 1}^M \sigma^2_i(x)
$$
Also, we can estimate the variance of our estimation:
$$
\hat{\sigma}^2_{f} = \frac{1}{M - 1} \sum_{i = 1}^M \left(\mu_i(x) - \hat{\mu}(x)\right)^2
$$
Using the estimators above, we can construct the prediction interval on test set.

### 3.2 Problems

- Under the random initialization, the independency of networks is undetermined. 

- We may face overfitting problem here since we estimate mean and variance together. 

$$
\sum_{i = 1}^n ln (E(\frac{1}{1 + e^{-yx}} | x \sim N[\mu; \sigma^2]))
$$