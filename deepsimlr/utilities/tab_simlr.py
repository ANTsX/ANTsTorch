
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import jax
from sklearn.decomposition import NMF


def whiten_slow(x, k=None):
    if x.size == 0:
        print("Usage: x_whitened = whiten(x)")
        return
    if k is None:
        k = min( x.shape )
    n, p = x.shape
    u, s, vh = jax.scipy.linalg.svd(jnp.dot(x, x.T), full_matrices=False)
    dd = 1.0 / jnp.sqrt(s)[:k]
    xw = jnp.dot(jnp.dot(u[:, :k], jnp.diag(dd)), vh[:k, :])
    xw = jnp.dot(xw, x)
    return xw

def whiten(X,fudge=1E-18):
   # the matrix X should be observations-by-components
   # get the covariance matrix
   Xcov = jnp.dot(X.T,X)
   # eigenvalue decomposition of the covariance matrix
   d, V = jnp.linalg.eigh(Xcov)
   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   d = jnp.maximum(fudge, d )
   D = jnp.diag(1. / jnp.sqrt(d))
   # whitening matrix
   W = jnp.dot(jnp.dot(V, D), V.T)
   # multiply by the whitening matrix
   X_white = jnp.dot(X, W)
   return X_white

def orthogonalize_and_q_sparsify(v, sparseness_quantile=0.5, positivity='positive',
                                 orthogonalize=False, soft_thresholding=True, 
                                 unit_norm=False):

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
                    v = v.at[vv,:].add( - temp * ip )

            local_v = v[vv,:]
            do_flip = False

            if jnp.quantile(local_v, 0.5) < 0:
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
        mynorm = jnp.linalg.norm(v)
        if mynorm > 0:
            v = jnp.divide( v, mynorm + eps_val )

    return v

def basic_q_sparsify(v, sparseness_quantile=0.5 ):

    if sparseness_quantile == 0:
        return v

    eps_val = 1.0e-16
    for vv in range(v.shape[0]):
        if np.var(v[vv, :]) > eps_val:
            local_v = v[vv,:]
            do_flip = False
            if jnp.quantile(local_v, 0.5) < 0:
                do_flip = True
            if do_flip:
                local_v = -local_v
 
            my_quant = jnp.quantile(local_v, sparseness_quantile)
            local_v = local_v - my_quant
            local_v = jnp.maximum(0, local_v )
            if do_flip:
               local_v = -local_v
            v = v.at[vv,:].set(local_v)
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
        vlist[k] = basic_q_sparsify( vlist[k], qlist[k] )
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
        vlist[k] = basic_q_sparsify( vlist[k], qlist[k] )
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

def correlation_regularization_matrices( matrix_list, correlation_threshold_list ):
    """
    matrix_list : list of matrices

    correlation_threshold_list : list of correlation values; regularization will be constructed from this
    """
    corl = []
    for k in range(len(matrix_list)):
        cor1 = jnp.corrcoef( matrix_list[k].T )
        cor1 = jnp.add( cor1, -correlation_threshold_list[k] )
        cor1 = jnp.maximum(0, cor1 )
        cor1sum = jnp.sum( cor1, axis=1 )
        for j in range( cor1.shape[0] ):
            if cor1sum[j] > 0:
                cor1 = cor1.at[j,:].divide( cor1sum[j] )
        corl.append( jax.numpy.asarray( cor1 ) )
    return corl

def tab_simlr( matrix_list, regularization_matrices, quantile_list, loss_function, simlr_optimizer, nev=2, max_iterations=5000, verbose=True ):
    """
    matrix_list : list of matrices

    regularization_matrices : list of regularization matrices

    quantile_list : list of quantiles that determine sparseness level

    loss_function : a deep_simlr loss function

    simlr_optimizer : optax optimizer to use with simlr (initialized with relevant parameters)

    nev : number of solution vectors per modality

    max_iterations : maximum number of optimization steps

    verbose: boolean

    returns:
        sparse solution parameters in form of a list (the v)
    """
    from jax import random
    parfun = jax.tree_util.Partial( loss_function, matrix_list, regularization_matrices, quantile_list )
    myfg = jax.grad( parfun )
    params = None
    if params is None:
        # initial solution
        n = matrix_list[0].shape[0]
        u = random.normal(random.PRNGKey(0), (n,nev))
        params = []
        for k in range(len(matrix_list)):
            params.append(  jax.numpy.asarray( jnp.dot( u.T, matrix_list[k] ) ) )

    import math
    best_params = params
    best_e = math.inf
    import optax
    opt_state = simlr_optimizer.init(params)
    loss_grad_fn = jax.value_and_grad(parfun)
    for i in range(max_iterations):
        loss_val, grads = loss_grad_fn(params)
        if loss_val < best_e:
            best_params = params
            best_e = loss_val
        updates, opt_state = simlr_optimizer.update(grads, opt_state, params=params )
        params = optax.apply_updates(params, updates)
        if i % 10 == 0 and verbose:
            print('Loss step {}: '.format(i), loss_val)

    params = best_params
    for k in range(len(params)):
        params[k] = jnp.dot( params[k], regularization_matrices[k]  )
        params[k] = basic_q_sparsify( params[k], quantile_list[k] )

    if verbose:
        for k in range(len(params)):
            temp = jnp.dot( matrix_list[k], params[k].T )
            print(jnp.corrcoef(temp.T))

    return params

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


def tab_simlr_old( x ):
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
            
