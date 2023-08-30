from jax import random
import jax
import deepsimlr
import functools
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import numpy as jnp
n=10
nev=2
x0 = random.normal(random.PRNGKey(0), (n,34))
x1 = random.normal(random.PRNGKey(1), (n,49))
x2 = random.normal(random.PRNGKey(3), (n,12))

# initial solution
u = random.normal(random.PRNGKey(n), (n,nev))
v0 = jnp.dot( u.T, x0 )
v1 = jnp.dot( u.T, x1 )
v2 = jnp.dot( u.T, x2 )

icatx = FastICA(n_components=nev,random_state=0,whiten='unit-variance')
pca = PCA(n_components=nev)

print( deepsimlr.simlr_low_rank_frobenius_norm_loss( [x0,x1,x2], [v0,v1,v2], icatx )  )
print( deepsimlr.simlr_low_rank_frobenius_norm_loss( [x0,x1,x2], [v0,v1,v2], pca )  )
print( deepsimlr.simlr_canonical_correlation_loss( [x0,x1,x2], [v0,v1,v2], icatx )  )
print( deepsimlr.simlr_canonical_correlation_loss( [x0,x1,x2], [v0,v1,v2], pca )  )


print( deepsimlr.simlr_low_rank_frobenius_norm_loss_pj( [x0,x1,x2], [v0,v1,v2] )  )
myf = deepsimlr.simlr_low_rank_frobenius_norm_loss_pj
myf = deepsimlr.simlr_canonical_correlation_loss_pj
parfun = jax.tree_util.Partial( myf, [x0,x1,x2] )
myfg = jax.grad( parfun )
print('testgrad with respect to v parameters')
params = [v0,v1,v2]
simlrgrad = myfg(  params )
print( simlrgrad[0].shape )


# see also 
# https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
# https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html
# https://github.com/google/flax
# https://flax.readthedocs.io/en/latest/guides/flax_basics.html

@jax.jit
def update(params,lr):
    grads = myfg(params)
    return jax.tree_map(
      lambda p, g: p - lr * g, params, grads
    )
lrlist = [1.0,0.1,0.01,0.001,0.0001,1e-5] # customize given frobenius norm or cca 
for LEARNING_RATE in lrlist:
    for _ in range(99):
        params = update(params,LEARNING_RATE)
        print( deepsimlr.simlr_canonical_correlation_loss_pj( [x0,x1,x2], params )  )


# now use optax to take advantage of adam
import optax
tx = optax.adam(learning_rate=0.001)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(parfun)

for i in range(101):
  loss_val, grads = loss_grad_fn(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)
