# alpha-beta network

## Idea

We consider the basic situation, a two-class classification problem. We can model this problem with a Bernoulli distribution with data $(x,y)$
$$
y_i \sim \mathrm{Bernoulli}\left(p(x_i)\right)
$$
And we can model our paramter $p$ as a Beta distribution
$$
p(x_i) \sim \mathrm{Beta}(\alpha(x_i), \beta(x_i))
$$

Therefore, we can write our likelihood function for a single sample$(x_i, y_i)$
$$
\mathcal{L}\left(\alpha(x_i), \beta(x_i); y_i\right) = \int_0^1 p^y (1-p)^{1-y} \frac{1}{B(\alpha(x_i), \beta(x_i))} p^{\alpha(x_i) -1} (1-p)^{\beta(x_i)-1} \mathrm{d} p
$$
which has a close form
$$
\mathcal{L}\left(\alpha(x_i), \beta(x_i); y_i\right) = \frac{B(\alpha(x_i)+y, \beta(x_i)+1-y)}{B(\alpha(x_i), \beta(x_i))}
$$

where beta function $B(\cdot)$ can be expressed by gamma function $\Gamma(\cdot)$
$$
B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}
$$

The negative log likelihood function is:
$$
\mathcal{C}(\alpha(\cdot), \beta(\cdot)) = -\sum_{i = 1}^n ln \left\{\frac{B(\alpha(x_i)+y, \beta(x_i)+1-y)}{B(\alpha(x_i), \beta(x_i))}\right\}
$$

and gradient on two network output is:
$$
\frac{\partial}{\partial \alpha} \mathcal{L}(\alpha,\beta) = \psi(\alpha + y) - \psi(\alpha) + \psi(\alpha + \beta) - \psi(\alpha + \beta -1)
$$
$$
\frac{\partial}{\partial \beta} \mathcal{L}(\alpha,\beta) = \psi(\beta + 1 - y) - \psi(\beta) + \psi(\alpha + \beta) - \psi(\alpha + \beta -1)
$$

where $\psi(\cdot)$ is digamma function, the derivative of log gamma function. 