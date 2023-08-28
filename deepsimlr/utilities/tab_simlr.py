def tab_simlr( x ):
    """
    linear alebraic simlr for tabular data

    x: list of matrices
    """
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    from jax import random
    from jax.nn import relu
    return x
    # dot of matrices
    # jnp.dot(x, x.T).block_until_ready()
    def selu(x, alpha=1.67, lmbda=1.05):
        return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
    # %timeit selu(x).block_until_ready()
    selu_jit = jit(selu)
    # %timeit selu_jit(x).block_until_ready()
    def sum_logistic(x):
      return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

    x_small = jnp.arange(3.)
    derivative_fn = grad(sum_logistic)
    print(derivative_fn(x_small))

    def forward( params, x ):
        *hidden, last = params
        for layer in hidden:
            x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
        return x @ last['weights'] + last['biases']

    def loss_fn(params, x, y):
        return jnp.mean((forward(params, x) - y) ** 2)

    LEARNING_RATE = 0.0001

    @jax.jit
    def update(params, x, y):
        grads = jax.grad(loss_fn)(params, x, y)
        # Note that `grads` is a pytree with the same structure as `params`.
        # `jax.grad` is one of the many JAX functions that has
        # built-in support for pytrees.
        # This is handy, because we can apply the SGD update using tree utils:
        return jax.tree_map(
                lambda p, g: p - LEARNING_RATE * g, params, grads
        )
            
