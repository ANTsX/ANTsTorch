
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import jax
import numpy as np
from sklearn.decomposition import NMF

def orthogonalize_and_q_sparsify(v, sparseness_quantile=0.5, positivity='positive',
                                 orthogonalize=False, soft_thresholding=True, unit_norm=False):

    if sparseness_quantile == 0:
        return v

    eps_val = 1.0e-16
    for vv in range(v.shape[0]):
        if np.var(v[vv, :]) > eps_val:
            if vv > 0 and orthogonalize:
                for vk in range(vv):
                    temp = v[vk,:]
                    denom = jnp.sum(temp * temp)
                    if denom > eps_val:
                        ip = jnp.sum(temp * v[vv,:]) / denom
                    else:
                        ip = 1
                    v[vv,:] = v[vv,:] - temp * ip

            local_v = v[vv,:]
            do_flip = False

            if jnp.sum(local_v > 0) < jnp.sum(local_v < 0):
                do_flip = True

            if do_flip:
               local_v = -local_v
 
            my_quant = jnp.quantile(local_v, sparseness_quantile)
            if positivity == 'positive':
                local_v = jnp.maximum(0, local_v )
            my_sign = jnp.sign(local_v)
            my_quant = jnp.quantile(jnp.abs(local_v), sparseness_quantile)
            temp = jnp.abs(local_v) - my_quant
            temp = jnp.maximum(0, temp )
            local_v = my_sign * temp

            if do_flip:
               local_v = -local_v
            v = v.at[vv,:].set(local_v)

    if unit_norm:
        v /= jnp.linalg.norm(v, axis=0)

    return v


def simlr_low_rank_frobenius_norm_loss_reg_sparse( xlist, reglist, qlist, vlist ):
    """
    implements a low-rank loss function (error) for simlr (pure jax)

    xlist : list of data matrices  ( nsamples by nfeatures )

    reglist : list of regularization matrices

    qlist : list of sparseness quantiles

    vlist : list of current solution vectors ( nev by nfeatures )
    """
    loss_sum = 0.0
    ulist = []
    nev = vlist[0].shape[0]
    for k in range(len(vlist)):
        # regularize vlist[k]
        vlist[k] = jnp.dot( vlist[k], reglist[k]  )
        # make sparse
        vlist[k] = orthogonalize_and_q_sparsify( vlist[k], qlist[k] )
        ulist.append( jnp.dot( xlist[k], vlist[k].T ) )    
    for k in range(len(vlist)):
        uconcat = []
        for j in range(len(vlist)):
            if k != j :
                uconcat.append( ulist[j] )
        uconcat = jnp.concatenate( uconcat, axis=1 )
        p1 = jax.numpy.linalg.svd( uconcat, full_matrices=False )[0][:,0:nev]
        p0 = jnp.dot( xlist[k], vlist[k].T )
        loss_sum = loss_sum + jax.numpy.linalg.norm(  p0 - p1 )
    return loss_sum


def simlr_canonical_correlation_loss_reg_sparse( xlist, reglist, qlist, vlist ):
    """
    implements a low-rank CCA-like loss function (error) for simlr (pure jax)

    xlist : list of data matrices  ( nsamples by nfeatures )

    reglist : list of regularization matrices

    qlist : list of sparseness quantiles

    vlist : list of current solution vectors ( nev by nfeatures )
    """
    loss_sum = 0.0
    ulist = []
    nev = vlist[0].shape[0]
    for k in range(len(vlist)):
        # regularize vlist[k]
        vlist[k] = jnp.dot( vlist[k], reglist[k]  )
        # make sparse
        vlist[k] = orthogonalize_and_q_sparsify( vlist[k], qlist[k] )
        ulist.append( jnp.dot( xlist[k], vlist[k].T ) )    
    for k in range(len(vlist)):
        uconcat = []
        for j in range(len(vlist)):
            if k != j :
                uconcat.append( ulist[j] )
        uconcat = jnp.concatenate( uconcat, axis=1 )
        p1 = jax.numpy.linalg.svd( uconcat, full_matrices=False )[0][:,0:nev]
        p0 = jnp.dot( xlist[k], vlist[k].T )
        mydot0 = jnp.dot( p0.T, p0 )
        mydot1 = jnp.dot( p1.T, p1 )
        normer = jnp.mean( jnp.diagonal( mydot0 ) ) * jnp.mean( jnp.diagonal( mydot1 ) )
        mydot = jnp.dot( p0.T, p1 )/jnp.sqrt(normer)
        loss_sum = loss_sum - jnp.mean( jnp.diagonal( mydot ) )
    return loss_sum

def simlr_low_rank_frobenius_norm_loss_pj( xlist, vlist ):
    """
    implements a low-rank loss function (error) for simlr (pure jax)

    xlist : list of data matrices  ( nsamples by nfeatures )

    vlist : list of current solution vectors ( nev by nfeatures )
    """
    loss_sum = 0.0
    ulist = []
    nev = vlist[0].shape[0]
    for k in range(len(vlist)):
        ulist.append( jnp.dot( xlist[k], vlist[k].T ) )    
    for k in range(len(vlist)):
        uconcat = []
        for j in range(len(vlist)):
            if k != j :
                uconcat.append( ulist[j] )
        uconcat = jnp.concatenate( uconcat, axis=1 )
        p1 = jax.numpy.linalg.svd( uconcat, full_matrices=False )[0][:,0:nev]
        p0 = jnp.dot( xlist[k], vlist[k].T )
        loss_sum = loss_sum + jax.numpy.linalg.norm(  p0 - p1 )
    return loss_sum

def simlr_canonical_correlation_loss_pj( xlist, vlist ):
    """
    implements a low-rank loss function (error) for simlr (pure jax)

    xlist : list of data matrices  ( nsamples by nfeatures )

    vlist : list of current solution vectors ( nev by nfeatures )
    """
    loss_sum = 0.0
    ulist = []
    nev = vlist[0].shape[0]
    for k in range(len(vlist)):
        ulist.append( jnp.dot( xlist[k], vlist[k].T ) )    
    for k in range(len(vlist)):
        uconcat = []
        for j in range(len(vlist)):
            if k != j :
                uconcat.append( ulist[j] )
        uconcat = jnp.concatenate( uconcat, axis=1 )
        p1 = jax.numpy.linalg.svd( uconcat, full_matrices=False )[0][:,0:nev]
        p0 = jnp.dot( xlist[k], vlist[k].T )
        mydot0 = jnp.dot( p0.T, p0 )
        mydot1 = jnp.dot( p1.T, p1 )
        normer = jnp.mean( jnp.diagonal( mydot0 ) ) * jnp.mean( jnp.diagonal( mydot1 ) )
        mydot = jnp.dot( p0.T, p1 )/jnp.sqrt(normer)
        loss_sum = loss_sum - jnp.mean( jnp.diagonal( mydot ) )
    return loss_sum

def simlr_low_rank_frobenius_norm_loss( xlist, vlist, simlrtransformer ):
    """
    implements a low-rank loss function (error) for simlr 

    xlist : list of data matrices  ( nsamples by nfeatures )

    vlist : list of current solution vectors ( nev by nfeatures )

    simlrtransformer : a scikitlearn transformer
    """
    loss_sum = 0.0
    ulist = []
    for k in range(len(vlist)):
        ulist.append( jnp.dot( xlist[k], vlist[k].T ) )    
    for k in range(len(vlist)):
        uconcat = []
        for j in range(len(vlist)):
            if k != j :
                uconcat.append( ulist[j] )
        uconcat = jnp.concatenate( uconcat, axis=1 )
        p1 = simlrtransformer.fit_transform( uconcat )
        p0 = jnp.dot( xlist[k], vlist[k].T )
        loss_sum = loss_sum + jax.numpy.linalg.norm(  p0 - p1 )
    return loss_sum

def simlr_canonical_correlation_loss( xlist, vlist, simlrtransformer ):
    """
    implements a CCA-like loss for simlr 

    xlist : list of data matrices  ( nsamples by nfeatures )

    vlist : list of current solution vectors ( nev by nfeatures )

    simlrtransformer : a scikitlearn transformer
    """
    loss_sum = 0.0
    ulist = []
    for k in range(len(vlist)):
        ulist.append( jnp.dot( xlist[k], vlist[k].T ) )    
    for k in range(len(vlist)):
        uconcat = []
        for j in range(len(vlist)):
            if k != j :
                uconcat.append( ulist[j] )
        uconcat = jnp.concatenate( uconcat, axis=1 )
        p1 = simlrtransformer.fit_transform( uconcat )
        p0 = jnp.dot( xlist[k], vlist[k].T )
        mydot = jnp.dot( p0.T, p1 )
        loss_sum = loss_sum - jnp.mean( jnp.diagonal( mydot ) )
    return loss_sum



from absl import app
from absl import flags

import jax.numpy as jnp

from jaxopt import BlockCoordinateDescent
from jaxopt import objective
from jaxopt import prox

import numpy as onp

from sklearn import datasets
from jax import random


flags.DEFINE_string("penalty", "l1", "Regularization type.")
flags.DEFINE_float("gamma", 1.e-31, "Regularization strength.")
FLAGS = flags.FLAGS


def nnreg(U, V_init, X, maxiter=150):
  """Regularized non-negative regression.

  We solve::

  min_{V >= 0} mean((U V^T - X) ** 2) + 0.5 * gamma * ||V||^2_2

  or

  min_{V >= 0} mean((U V^T - X) ** 2) +  gamma * ||V||_1
  """
  if FLAGS.penalty == "l2":
    block_prox = prox.prox_non_negative_ridge
  elif FLAGS.penalty == "l1":
    block_prox = prox.prox_non_negative_lasso
  else:
    raise ValueError("Invalid penalty.")

  bcd = BlockCoordinateDescent(fun=objective.least_squares,
                               block_prox=block_prox,
                               maxiter=maxiter)
  sol = bcd.run(init_params=V_init.T, hyperparams_prox=FLAGS.gamma, data=(U, X))
  return sol.params.T  # approximate solution V


def reconstruction_error(U, V, X):
  """Computes (unregularized) reconstruction error."""
  UV = jnp.dot(U, V.T)
  return 0.5 * jnp.mean((UV - X) ** 2)


def tab_simlrx( x0, x1, v0i, v1i, maxiter=10):
  v0, v1 = v0i, v1i
  u0 = jnp.dot( x0, v0.T)
  u1 = jnp.dot( x1, v1.T )
  error = reconstruction_error(u0, v0.T, x0) + reconstruction_error(u1, v1.T, x1)
  print(f"STEP: 0; Error: {error:.3f}")
  print()

  for step in range(1, maxiter + 1):
    print(f"STEP: {step}")
    v0 = nnreg(u1, v0.T, x0, maxiter=5).T
    error = reconstruction_error(u1, v0.T, x0)
    print(f"part 1-Error: {error:.3f} (V update)")

    v1 = nnreg(u0, v1.T, x1, maxiter=5).T
    error = reconstruction_error(u0, v1.T, x1)
    print(f"part 2-Error: {error:.3f} (V update)")


    u1 = jnp.dot( x1, v1.T)
    u0 = jnp.dot( x0, v0.T)

    # U = nnreg(V, U, X.T, maxiter=150) - smart!
    # error = reconstruction_error(U, V, X)
    # print(f"Error: {error:.3f} (U update)")
    # print()


def mainer(argv):
  del argv
  x0 = random.normal(random.PRNGKey(0), (10,3))
  x1 = random.normal(random.PRNGKey(1), (10,4))

  # initial solution
  u = random.normal(random.PRNGKey(10), (10,2))
  v0 = jnp.dot( u.T, x0 ).block_until_ready()
  v1 = jnp.dot( u.T, x1 ).block_until_ready()

  n_samples = x0.shape[0]
  n_features0 = x0.shape[1]
  n_features1 = x1.shape[1]
  n_components = u.shape[1]

  # Run the algorithm.
  print("penalty:", FLAGS.penalty)
  print("gamma", FLAGS.gamma)
  print()

  tab_simlr(x0, x1, v0, v1, maxiter=300)


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
            
