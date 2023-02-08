---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Pricing a European Call Option under Risk Neutrality

+++

#### Written for the CBC QuantEcon Workshop (September 2022)
#### Author: [John Stachurski](http://johnstachurski.net/)

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

We are going to price a European call option under the assumption of risk neutrality.  The price satisfies


$$ P = \beta^n \mathbb E \max\{ S_n - K, 0 \} $$

where

1. $\beta$ is a discount factor,
2. $n$ is the expiry date,
2. $K$ is the strike price and
3. $\{S_t\}$ is the price of the underlying asset at each time $t$.

(For example, if the call option is to buy stock in Amazon at strike price $K$, the owner has the right to buy 1 share in Amazon at price $K$ after $n$ days.  The price is the expectation of the return $\max\{S_n - K, 0\}$, discounted to current value.)

+++

### Exercise

Suppose that $S_n$ has the [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) distribution with parameters $\mu$ and $\sigma$.  Let $f$ denote the density of this distribution.  Then

$$ P = \beta^n \int_0^\infty \max\{x - K, 0\} f(x) dx $$

Plot the function 

$$g(x) = \beta^n  \max\{x - K, 0\} f(x)$$ 

over the interval $[0, 400]$ when

```{code-cell} ipython3
μ, σ, β, n, K = 4, 0.25, 0.99, 10, 40
```

Hint: From `scipy.stats` you can import `lognorm` and then use `lognorm(x, σ, scale=np.exp(μ)` to get the density $f$.

```{code-cell} ipython3
# Put your solution here
```

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

```{code-cell} ipython3
from scipy.integrate import quad
from scipy.stats import lognorm

def g(x):
    return β**n * np.maximum(x - K, 0) * lognorm.pdf(x, σ, scale=np.exp(μ))

x_grid = np.linspace(0, 400, 1000)
y_grid = g(x_grid) 

fig, ax = plt.subplots()
ax.plot(x_grid, y_grid, label="$g$")
ax.legend()
plt.show()
```

### Exercise

In order to get the option price, compute the integral of this function numerically using `quad` from `scipy.optimize`.

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

The integral and hence the price is

```{code-cell} ipython3
P, error = quad(g, 0, 1_000)
print(f"The numerical integration based option price is {P:3f}")
```

### Exercise

+++

Try to get a similar result using Monte Carlo to compute the expectation term in the option price, rather than `quad`.

+++

In particular, use the fact that if $S_n^1, \ldots, S_n^M$ are independent draws from the lognormal distribution specified above, then, by the law of large numbers,

$$ \mathbb E \max\{ S_n - K, 0 \} 
    \approx
    \frac{1}{M} \sum_{m=1}^M \max \{S_n^m - K, 0 \}
    $$
    
Set `M = 10_000_000`

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

```{code-cell} ipython3
M = 10_000_000
S = np.exp(μ + σ * np.random.randn(M))
return_draws = np.maximum(S - K, 0)
P = β**n * np.mean(return_draws) 
print(f"The Monte Carlo option price is {P:3f}")
```

### Exercise

In this exercise we investigate a more realistic model for the stock price $S_n$.

This comes from specifying the underlying dynamics.

+++

One common model for $\{S_t\}$ is

$$ \ln \frac{S_{t+1}}{S_t} = \mu + \sigma \xi_{t+1} $$

where $\{ \xi_t \}$ is IID and standard normal.  However, its predictions are counterfactual because volatility is not stationary but rather changes over time.  

Here is an improved version:

$$ \ln \frac{S_{t+1}}{S_t} = \mu + \sigma_t \xi_{t+1} $$

where 

$$ 
    \sigma_t = \exp(h_t), 
    \quad
        h_{t+1} = \rho h_t + \nu \eta_{t+1}
$$

Here $\{\eta_t\}$ is also IID and standard normal.

+++

Write a function that simulates the sequence $S_0, \ldots, S_n$, where the parameters are set to

```{code-cell} ipython3
μ  = 0.0001
ρ  = 0.1
ν  = 0.001
S0 = 10
h0 = 0
n  = 20
```

(Here `S0` is $S_0$ and `h0` is $h_0$.)

+++

Plot 50 paths of the form $S_0, \ldots, S_n$.

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

With $s_t := \ln S_t$, the price dynamics become

$$ s_{t+1} = s_t + \mu + \exp(h_t) \xi_{t+1} $$

Here is a function to simulate a path using this equation:

```{code-cell} ipython3
from numpy.random import randn

def simulate_asset_price_path(μ=μ, S0=S0, h0=h0, n=n, ρ=ρ, ν=ν):
    s = np.empty(n+1)
    s[0] = np.log(S0)

    h = h0
    for t in range(n):
        s[t+1] = s[t] + μ + np.exp(h) * randn()
        h = ρ * h + ν * randn()
        
    return np.exp(s)
```

Here we plot the paths and the log of the paths.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1)

titles = 'log paths', 'paths'
transforms = np.log, lambda x: x
for ax, transform, title in zip(axes, transforms, titles):
    for i in range(50):
        path = simulate_asset_price_path()
        ax.plot(transform(path))
    ax.set_title(title)
    
fig.tight_layout()
plt.show()
```

### Exercise 3

+++

Compute the price of the option $P_0$ by Monte Carlo, averaging over realizations $S_n^1, \ldots, S_n^M$ of $S_n$ and appealing to the law of large numbers:

$$ \mathbb E \max\{ S_n - K, 0 \} 
    \approx
    \frac{1}{M} \sum_{m=1}^M \max \{S_n^m - K, 0 \}
    $$
    
Use the values given below:

```{code-cell} ipython3
M = 10_000_000
K = 100
n = 10
β = 0.95
```

To the extend that you can, write fast, efficient code to compute the option price.  

In particular, try to speed up the code above using `jit` or `njit` from Numba.

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

```{code-cell} ipython3
from numba import njit, prange
```

```{code-cell} ipython3
@njit
def compute_call_price(β=β,
                       μ=μ,
                       S0=S0,
                       h0=h0,
                       K=K,
                       n=n,
                       ρ=ρ,
                       ν=ν,
                       M=M):
    current_sum = 0.0
    # For each sample path
    for m in range(M):
        s = np.log(S0)
        h = h0
        # Simulate forward in time
        for t in range(n):
            s = s + μ + np.exp(h) * randn()
            h = ρ * h + ν * randn()
        # And add the value max{S_n - K, 0} to current_sum
        current_sum += np.maximum(np.exp(s) - K, 0)
        
    return β**n * current_sum / M
```

```{code-cell} ipython3
%%time 
compute_call_price()
```

### Exercise 4

If you can, use `prange` from Numba to parallelize this code and make it even faster.

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

```{code-cell} ipython3
@njit(parallel=True)
def compute_call_price_parallel(β=β,
                                μ=μ,
                                S0=S0,
                                h0=h0,
                                K=K,
                                n=n,
                                ρ=ρ,
                                ν=ν,
                                M=M):
    current_sum = 0.0
    # For each sample path
    for m in prange(M):
        s = np.log(S0)
        h = h0
        # Simulate forward in time
        for t in range(n):
            s = s + μ + np.exp(h) * randn()
            h = ρ * h + ν * randn()
        # And add the value max{S_n - K, 0} to current_sum
        current_sum += np.maximum(np.exp(s) - K, 0)
        
    return β**n * current_sum / M
```

```{code-cell} ipython3
%%time
compute_call_price_parallel()
```

```{code-cell} ipython3
%%time
compute_call_price_parallel()
```

```{code-cell} ipython3

```
