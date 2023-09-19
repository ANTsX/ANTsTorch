
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import jax
from sklearn.decomposition import NMF

import jax
import jax.numpy as jnp
import tqdm
import math

def icawhiten(x):
    # Calculate the covariance matrix
    coVarM = jnp.cov(x)
    # Single value decoposition
    U, S, V = jnp.linalg.svd(coVarM, full_matrices=False)
    # Calculate diagonal matrix of eigenvalues
    d = jnp.diag(1.0 / jnp.sqrt(S))
    # Calculate whitening matrix
    whiteM = jnp.dot(U, jnp.dot(d, U.T))
    # Project onto whitening matrix
    Xw = jnp.dot(whiteM, x)
    return Xw

def preprocess_signal(signal):
    """Center and whiten the signal
    x_preprocessed = A @ (x - mean)

    Args
        signal [num_samples, signal_dim]
    
    Returns
        signal_preprocessed [num_samples, signal_dim]
        preprocessing_params
            A [signal_dim, signal_dim]
            mean [signal_dim]
    """
    mean = jnp.mean(signal, axis=0)
    signal_centered = signal - jnp.mean(signal, axis=0)

    signal_cov = jnp.mean(jax.vmap(jnp.outer, (0, 0), 0)(signal_centered, signal_centered), axis=0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(signal_cov)
    A = jnp.diag(eigenvalues ** (-1 / 2)) @ eigenvectors.T

    return jax.vmap(jnp.matmul, (None, 0), 0)(A, signal_centered), (A, mean)


def fastIca(signals,  k=None, alpha = 1, thresh=1e-6, iterations=50 ):
    # calculate W but return the S matrix
    signals = icawhiten( signals )
    # signals = preprocess_signal( signals )
    m, n = signals.shape

    # Initialize random weights
    if k is None:
        k=m
    W = random.normal(random.PRNGKey(0), (k,m))

    for c in range(k):
            w = W[c, :].copy().reshape(m, 1)
            w = w / jnp.sqrt((w ** 2).sum())

            i = 0
            lim = 100
            while ((lim > thresh) & (i < iterations)):

                # Dot product of weight and signal
                ws = jnp.dot(w.T, signals)

                # Pass w*s into contrast function g
                wg = jnp.tanh(ws * alpha).T

                # Pass w*s into g prime
                wg_ = (1 - jnp.square(jnp.tanh(ws))) * alpha

                # Update weights
                wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

                # Decorrelate weights              
                wNew = wNew - jnp.dot( jnp.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / jnp.sqrt((wNew ** 2).sum())

                # Calculate limit condition
                lim = jnp.abs(jnp.abs((wNew * w).sum()) - 1)

                # Update weights
                w = wNew

                # Update counter
                i += 1

            W = W.at[c,:].set( w.T )
    return jnp.dot( W, signals )


# Calculate Kurtosis
def kurtosis(x):
    n = jnp.shape(x)[0]
    mean = jnp.sum((x**1)/n) # Calculate the mean
    var = jnp.sum((x-mean)**2)/n # Calculate the variance
    # skew = jnp.sum((x-mean)**3)/n # Calculate the skewness
    kurt = jnp.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = kurt/(var**2)-3
    return kurt

def get_signal(mixing_matrix, source):
    """Compute single signal from a single source
    Args
        mixing_matrix [signal_dim, source_dim]
        source [source_dim]
    
    Returns
        signal [signal_dim]
    """
    return jnp.dot(mixing_matrix, source)


def get_subgaussian_log_prob(source):
    """Subgaussian log probability of a single source.

    Args
        source [source_dim]

    Returns []
    """
    return jnp.sum(jnp.sqrt(jnp.abs(source)))


def get_supergaussian_log_prob(source):
    """Supergaussian log probability of a single source.
    log cosh(x) = log ( (exp(x) + exp(-x)) / 2 )
                = log (exp(x) + exp(-x)) - log(2)
                = logaddexp(x, -x) - log(2)
                   
    https://en.wikipedia.org/wiki/Hyperbolic_functions#Exponential_definitions
    https://en.wikipedia.org/wiki/FastICA#Single_component_extraction

    Args
        source [source_dim]

    Returns []
    """
    return jnp.sum(jnp.logaddexp(source, -source) - math.log(2))


def get_antisymmetric_matrix(raw_antisymmetric_matrix):
    """Returns an antisymmetric matrix
    https://en.wikipedia.org/wiki/Skew-symmetric_matrix

    Args
        raw_antisymmetric_matrix [dim * (dim - 1) / 2]: elements in the upper triangular
            (excluding the diagonal)

    Returns [dim, dim]
    """
    dim = math.ceil(math.sqrt(raw_antisymmetric_matrix.shape[0] * 2))
    zeros = jnp.zeros((dim, dim))
    indices = jnp.triu_indices(dim, k=1)
    upper_triangular = zeros.at[indices].set(raw_antisymmetric_matrix)
    return upper_triangular - upper_triangular.T


def get_orthonormal_matrix(raw_orthonormal_matrix):
    """Returns an orthonormal matrix
    https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map

    Args
        raw_orthonormal_matrix [dim * (dim - 1) / 2]

    Returns [dim, dim]
    """
    antisymmetric_matrix = get_antisymmetric_matrix(raw_orthonormal_matrix)
    dim = antisymmetric_matrix.shape[0]
    eye = jnp.eye(dim)
    return jnp.matmul(eye - antisymmetric_matrix, jnp.linalg.inv(eye + antisymmetric_matrix))


def get_source(signal, raw_mixing_matrix):
    """Get source from signal
    
    Args
        signal [signal_dim]
        raw_mixing_matrix [dim * (dim - 1) / 2]
    
    Returns []
    """
    return jnp.matmul(get_mixing_matrix(raw_mixing_matrix).T, signal)


def get_log_likelihood(signal, raw_mixing_matrix, get_source_log_prob):
    """Log likelihood of a single signal log p(x_n)
    
    Args
        signal [signal_dim]
        raw_mixing_matrix [dim * (dim - 1) / 2]
        get_source_log_prob [source_dim] -> []
    
    Returns []
    """
    return get_source_log_prob(get_source(signal, raw_mixing_matrix))


def get_mixing_matrix(raw_mixing_matrix):
    """Get mixing matrix from a vector of raw values (to be optimized)

    Args
        raw_orthonormal_matrix [dim * (dim - 1) / 2]

    Returns [dim, dim]
    """
    return get_orthonormal_matrix(raw_mixing_matrix)


def get_total_log_likelihood(signals, raw_mixing_matrix, get_source_log_prob):
    """Log likelihood of all signals ∑_n log p(x_n)
    
    Args
        signals [num_samples, signal_dim]
        raw_mixing_matrix [dim * (dim - 1) / 2]
        get_source_log_prob [source_dim] -> []
    
    Returns []
    """
    log_likelihoods = jax.vmap(get_log_likelihood, (0, None, None), 0)(
        signals, raw_mixing_matrix, get_source_log_prob
    )
    return jnp.sum(log_likelihoods)


def update_raw_mixing_matrix(raw_mixing_matrix, signals, get_source_log_prob, lr=1e-3):
    """Update raw mixing matrix by stepping the gradient

    Args:
        raw_mixing_matrix [signal_dim, source_dim]
        signals [num_samples, signal_dim]
        get_source_log_prob [source_dim] -> []
        lr (float)

    Returns
        total_log_likelihood []
        updated_raw_mixing_matrix [signal_dim, source_dim]
    """
    total_log_likelihood, g = jax.value_and_grad(get_total_log_likelihood, 1)(
        signals, raw_mixing_matrix, get_source_log_prob
    )
    return total_log_likelihood, raw_mixing_matrix + lr * g



def ica(signal, get_source_log_prob, num_iterations=1000, lr=1e-3):
    """Gradient-descent based maximum likelihood estimation of the independent component analysis
    (ICA) model

    Args
        signal [num_samples, signal_dim]
        get_source_log_prob [source_dim] -> []
        num_iterations (int)
        lr (float)
    
    Returns
        total_log_likelihoods: list of length num_iterations
        raw_mixing_matrices: list of length (num_iterations + 1)
        preprocessing_params
            A [signal_dim, signal_dim]
            mean [signal_dim]

            where the preprocessed signal is obtained by

            matmul(A, (signal - mean))
    """
    dim = signal.shape[1]

    # Preprocess
    signal_preprocessed, preprocessing_params = preprocess_signal(signal)

    # Optim
    key=0
    raw_mixing_matrix = jax.random.normal(key, (int(dim * (dim - 1) / 2),))

    total_log_likelihoods = []
    raw_mixing_matrices = [raw_mixing_matrix]
    for _ in tqdm.tqdm(range(num_iterations)):
        total_log_likelihood, raw_mixing_matrix = update_raw_mixing_matrix(
            raw_mixing_matrix, signal_preprocessed, get_source_log_prob, lr
        )
        total_log_likelihoods.append(total_log_likelihood.item())
        raw_mixing_matrices.append(raw_mixing_matrix)

    return total_log_likelihoods, raw_mixing_matrices, preprocessing_params

def corr2_coeff(A, B):
    # from stack overflow
    # Rowwise mean of input arrays & subtract from input arrays themselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    # Finally get corr coeff
    return jnp.dot(A_mA, B_mB.T) / jnp.sqrt(jnp.dot(ssA[:, None],ssB[None]))


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

def orthogonalize_and_q_sparsify(v, sparseness_quantile=0.5, positivity=False,
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
            if positivity:
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


def simlr_low_rank_frobenius_norm_loss_reg_sparse( xlist, reglist, qlist, positivity, vlist ):
    """
    implements a low-rank loss function (error) for simlr (pure jax)

    xlist : list of data matrices  ( nsamples by nfeatures )

    reglist : list of regularization matrices

    qlist : list of sparseness quantiles

    positivity : boolean

    vlist : list of current solution vectors ( nev by nfeatures )
    """
    loss_sum = 0.0
    ulist = []
    nev = vlist[0].shape[0]
    for k in range(len(vlist)):
        # regularize vlist[k]
        vlist[k] = jnp.dot( vlist[k], reglist[k]  )
        # make sparse
        vlist[k] = orthogonalize_and_q_sparsify( vlist[k], qlist[k], positivity=positivity )
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

def simlr_canonical_correlation_loss_reg_sparse( xlist, reglist, qlist, positivity, nondiag_weight, vlist ):
    """
    implements a low-rank CCA-like loss function (error) for simlr (pure jax)

    xlist : list of data matrices  ( nsamples by nfeatures )

    reglist : list of regularization matrices

    qlist : list of sparseness quantiles

    positivity : boolean

    nondiag_weight : scalar parameter penalizing offdiagonal correlations within a given modality

    vlist : list of current solution vectors ( nev by nfeatures )
    """
    loss_sum = 0.0
    ulist = []
    nev = vlist[0].shape[0]
    for k in range(len(vlist)):
        # regularize vlist[k]
        vlist[k] = jnp.dot( vlist[k], reglist[k]  )
        # make sparse
        vlist[k] = orthogonalize_and_q_sparsify( vlist[k], qlist[k],positivity=positivity )
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
        normer =  jnp.sqrt( jnp.linalg.norm( mydot0 ) * jnp.linalg.norm( mydot1 ))
        mydot = jnp.dot( p0.T, p1 )/normer
        offdiag = jnp.linalg.norm( jnp.eye(mydot.shape[0]) - mydot ) * nondiag_weight
        mycorr = jnp.mean( jnp.diagonal( mydot ) )
        loss_sum = loss_sum - mycorr + offdiag 
    return loss_sum

def simlr_absolute_canonical_covariance( xlist, reglist, qlist, positivity, nondiag_weight, merging, vlist ):
    """
    implements a low-rank CCA-like loss function (error) for simlr (pure jax)

    xlist : list of data matrices  ( nsamples by nfeatures )

    reglist : list of regularization matrices

    qlist : list of sparseness quantiles

    positivity : boolean

    nondiag_weight : scalar parameter penalizing offdiagonal correlations within a given modality

    merging : string svd, ica or average

    vlist : list of current solution vectors ( nev by nfeatures )
    """
    loss_sum = 0.0
    ulist = []
    nev = vlist[0].shape[0]
    for k in range(len(vlist)):
        # regularize vlist[k]
        vlist[k] = jnp.dot( vlist[k], reglist[k]  )
        # make sparse
        vlist[k] = orthogonalize_and_q_sparsify( vlist[k], qlist[k],positivity=positivity )
        ulist.append( jnp.dot( xlist[k], vlist[k].T ) )    
    for k in range(len(vlist)):
        uconcat = []
        for j in range(len(vlist)):
            if k != j :
                uconcat.append( ulist[j] )
        if merging == 'svd':
            uconcat = jnp.concatenate( uconcat, axis=1 )
            p1 = jax.numpy.linalg.svd( uconcat, full_matrices=False )[0][:,0:nev]
        elif merging == 'ica':
            uconcat = jnp.concatenate( uconcat, axis=1 )
            p1 = fastIca( uconcat.T, nev ).T
        else:
            p1 = uconcat[0]/jnp.linalg.norm(uconcat[0])
            if len(uconcat) > 1:
                for m in range(1,len(uconcat)):
                    p1 = p1 + uconcat[m]/jnp.linalg.norm(uconcat[m])
            p1 = p1 / len(uconcat)
        p0 = jnp.dot( xlist[k], vlist[k].T )
        intramodalityCor = corr2_coeff( p0.T, p0.T )
        intermodalityCor = corr2_coeff( p0.T, p1.T )
        offdiag = 0.0
        offdiagcount = ( nev * nev - nev ) / 2
        for q0 in range(1,intramodalityCor.shape[0]):
            for q1 in range(q0+1,intramodalityCor.shape[1]):
                lcov = intramodalityCor.at[q0,q1].get()
                offdiag = offdiag + jnp.abs(lcov)/offdiagcount
        mycorr = jnp.trace( jnp.abs( intermodalityCor ) )/nev
        loss_sum = loss_sum - mycorr + offdiag * nondiag_weight
    return loss_sum/len(xlist)

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
        p0 = p0 / jnp.linalg.norm( p0 )
        p1 = p1 / jnp.linalg.norm( p1 )
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

def tab_simlr( matrix_list, regularization_matrices, quantile_list, loss_function, simlr_optimizer, nev=2, max_iterations=5000, positivity=False, verbose=True ):
    """
    matrix_list : list of matrices

    regularization_matrices : list of regularization matrices

    quantile_list : list of quantiles that determine sparseness level

    loss_function : a partial deep_simlr loss function where its only parameter is the current solution value (params)

    simlr_optimizer : optax optimizer to use with simlr (initialized with relevant parameters)

    nev : number of solution vectors per modality

    max_iterations : maximum number of optimization steps

    positivity : constrains solution to be positive if set to True

    verbose: boolean

    returns:
        sparse solution parameters in form of a list (the v)
    """
    from jax import random
    myfg = jax.grad( loss_function )
    params = None
    n = matrix_list[0].shape[0]
    u = random.normal(random.PRNGKey(0), (n,nev))
    if params is None:
        # initial solution
        params = []
        for k in range(len(matrix_list)):
            params.append(  jax.numpy.asarray( jnp.dot( u.T, matrix_list[k] ) ) )

    import math
    best_params = params
    best_e = math.inf
    import optax
    opt_state = simlr_optimizer.init(params)
    loss_grad_fn = jax.value_and_grad(loss_function)
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
        params[k] = orthogonalize_and_q_sparsify( params[k], quantile_list[k],positivity=positivity )

    if verbose:
        print("Within modality")
        for k in range(len(params)):
            temp = jnp.dot( matrix_list[k], params[k].T )
            print(corr2_coeff( temp.T, temp.T ))

        print("Between modality")
        for k in range(len(params)):
            temp = jnp.dot( matrix_list[k], params[k].T )
            for j in range(k+1,len(params)):
                temp2 = jnp.dot( matrix_list[j], params[j].T )
                print( jnp.diag( corr2_coeff( temp.T, temp2.T )) )

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
            
