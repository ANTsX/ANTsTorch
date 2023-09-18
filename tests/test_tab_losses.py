from jax import random
import jax
import deepsimlr
import functools
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import numpy as jnp
import sys
import optax
n=10
nev=3
x0 = random.normal(random.PRNGKey(0), (n,34))
x1 = random.normal(random.PRNGKey(1), (n,49))
x2 = random.normal(random.PRNGKey(3), (n,12))


# def tab_simlr( matrix_list, correlation_threshold_list, quantile_list, loss_function, nev=2, learning_rate=1.e-4, max_iterations=5000, verbose=True ):
regmats = deepsimlr.correlation_regularization_matrices( [x0,x1,x2], [0.5,0.5,0.5] )


parfun = jax.tree_util.Partial( 
  deepsimlr.simlr_low_rank_frobenius_norm_loss_reg_sparse, 
  [x0,x1,x2], regmats, [0.9,0.9,0.9], False )

mysim = deepsimlr.tab_simlr( [x0,x1,x2], regmats, [0.9,0.9,0.9],   
  parfun, 
  nev=5, 
  simlr_optimizer=optax.optimistic_gradient_descent( 0.01 ), max_iterations=11, 
  positivity=False   )

parfun0 = jax.tree_util.Partial( 
  deepsimlr.simlr_absolute_canonical_covariance, 
  [x0,x1,x2], regmats, [0.9,0.9,0.9], False, 1e-4 )

parfun1 = jax.tree_util.Partial( 
  deepsimlr.simlr_absolute_canonical_covariance, 
  [x0,x1,x2], regmats, [0.9,0.9,0.9], False, 1e-6 )

nits=201
nev=3

mysimcc0 = deepsimlr.tab_simlr( [x0,x1,x2], regmats, [0.9,0.9,0.9],   
  parfun0, 
  nev=nev,   
  simlr_optimizer=optax.optimistic_gradient_descent( 10 ),
  max_iterations=nits, 
  positivity=True  )

mysimcc1 = deepsimlr.tab_simlr( [x0,x1,x2], regmats, [0.9,0.9,0.9],   
  parfun1, 
  nev=nev,   
  simlr_optimizer=optax.optimistic_gradient_descent( 10 ),
  max_iterations=nits, 
  positivity=True  )

derka
sys.exit(0)

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

derka
derka

# see also 
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
# https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
# https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html
# https://github.com/google/flax
# https://flax.readthedocs.io/en/latest/guides/flax_basics.html
# https://raw.githubusercontent.com/tuananhle7/ica/main/ica.py
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

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


