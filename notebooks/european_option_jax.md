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

## Pricing a European Call Option (Numba vs JAX)

+++

#### Written for the CBC QuantEcon Workshop (September 2022)

#### Author: [John Stachurski](http://johnstachurski.net/)


Previously we computed the value of a European call option via Monte Carlo using Numba-based routines.

Here we try the same operations using JAX.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

Recall that we want to compute


$$ P = \beta^n \mathbb E \max\{ S_n - K, 0 \} $$

We suppose that

```{code-cell} ipython3
n = 20
β = 0.99
K = 100
```

The dynamics are

$$ \ln \frac{S_{t+1}}{S_t} = \mu + \sigma_t \xi_{t+1} $$

where 

$$ 
    \sigma_t = \exp(h_t), 
    \quad
        h_{t+1} = \rho h_t + \nu \eta_{t+1}
$$

Here $\{\xi_t\}$ and $\{\eta_t\}$ are IID and standard normal.

+++

With $s_t := \ln S_t$, the price dynamics become

$$ s_{t+1} = s_t + \mu + \exp(h_t) \xi_{t+1} $$

+++

We use the following defaults.

```{code-cell} ipython3
μ  = 0.0001
ρ  = 0.1
ν  = 0.001
S0 = 10
h0 = 0
```

(Here `S0` is $S_0$ and `h0` is $h_0$.)

+++

We used the following estimate of the price, computed via Monte Carlo and applying Numba and parallelization.

```{code-cell} ipython3
from numba import njit, prange
from numpy.random import randn
```

```{code-cell} ipython3
M = 10_000_000
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
from numba import get_num_threads, set_num_threads
get_num_threads()
```

```{code-cell} ipython3
%%time
compute_call_price_parallel()
```

```{code-cell} ipython3
%%time
compute_call_price_parallel()
```

### Exercise

Try to shift the whole operation to the GPU using JAX and test your speed gain.

+++

### Solution

```{code-cell} ipython3
!nvidia-smi
```

```{code-cell} ipython3
import jax
import jax.numpy as jnp
```

```{code-cell} ipython3
@jax.jit
def compute_call_price_jax(β=β,
                           μ=μ,
                           S0=S0,
                           h0=h0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=M,
                           key=jax.random.PRNGKey(1)):

    s = jnp.full(M, np.log(S0))
    h = jnp.full(M, h0)
    for t in range(n):
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, (2, M))
        s = s + μ + jnp.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
    expectation = jnp.mean(jnp.maximum(jnp.exp(s) - K, 0))
        
    return β**n * expectation
```

```{code-cell} ipython3
%%time 
compute_call_price_jax().block_until_ready()
```

```{code-cell} ipython3
%%time 
compute_call_price_jax().block_until_ready()
```

```{code-cell} ipython3

```
