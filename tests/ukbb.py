import jax
from jax.config import config
config.update("jax_debug_nans", False)
import deepsimlr
import functools
import numpy as jnp
import sys
import pandas as pd
import optax as optax
optlist = [
  optax.adabelief,
  optax.adagrad,
  optax.adafactor,
  optax.adam,
  optax.adamax,
  optax.adamaxw,
  optax.adamw,
  optax.amsgrad,
  optax.fromage,
  optax.lamb,
  optax.lars,
  optax.lion,
  optax.sm3,
  optax.noisy_sgd,
  optax.novograd,
  optax.optimistic_gradient_descent,
  optax.radam,
  optax.rmsprop,
  optax.sgd,
  optax.yogi]
# x0=deepsimlr.whiten( pd.read_csv("/tmp/ukt1x.csv").to_numpy() )
# x1=deepsimlr.whiten( pd.read_csv("/tmp/ukdtix.csv").to_numpy() )
# x2=deepsimlr.whiten( pd.read_csv("/tmp/ukrsfx.csv").to_numpy() )
x0=pd.read_csv("/tmp/ukt1x.csv").to_numpy()
x1=pd.read_csv("/tmp/ukdtix.csv").to_numpy()
x2=pd.read_csv("/tmp/ukrsfx.csv").to_numpy()
x0 = x0 / jax.numpy.linalg.norm( x0 )
x1 = x1 / jax.numpy.linalg.norm( x1 )
x2 = x2 / jax.numpy.linalg.norm( x2 )
xx=jnp.dot( x0.T, x0 )
xxw=deepsimlr.whiten( xx )
print(type(xxw))
qq = 0.9 # regularization
sp = 0.99 # quantile sparseness
sparseness = [sp,sp,sp]
simlrdata = [jnp.asarray( x0 ), jnp.asarray( x1 ), jnp.asarray( x2 ) ]
regmats = deepsimlr.correlation_regularization_matrices( simlrdata, [qq,qq,qq] )

# lion and rmsprop looks best for this
lr=0.01
parfun = jax.tree_util.Partial( 
  deepsimlr.simlr_low_rank_frobenius_norm_loss_reg_sparse, 
  simlrdata, regmats, sparseness, False )

# rmsprop looks best for this
lr=1.0
mypos=True
# (29936, 20) - uconcat
# (10, 20) - ica W matrix
parfun = jax.tree_util.Partial( 
  deepsimlr.simlr_absolute_canonical_covariance, 
  simlrdata, regmats, sparseness, mypos, 0.001, 'ica' )

for myopt in [optax.rmsprop]:
  print(myopt)
  mysim = deepsimlr.tab_simlr(
    simlrdata,
    regmats,
    sparseness,
    parfun,
    simlr_optimizer=myopt(lr,centered=False, momentum=None, nesterov=False),
    nev=10, max_iterations=111, positivity=mypos )
##############################
# write the features out
pd.DataFrame(mysim[0]).to_csv("/tmp/ukt1ev_b.csv")
pd.DataFrame(mysim[1]).to_csv("/tmp/ukdtiev_b.csv")
pd.DataFrame(mysim[2]).to_csv("/tmp/ukrsfev_b.csv")

derka

for k in range(len(mysim)):
  temp = jnp.dot( simlrdata[k], mysim[k].T )
  for j in range(k+1,len(mysim)):
    temp2 = jnp.dot( simlrdata[j], mysim[j].T )
    print( jnp.diag(deepsimlr.corr2_coeff(temp.T,temp2.T)) )

k=0
j=0
temp = jnp.dot( simlrdata[k], mysim[k].T )
temp2 = jnp.dot( simlrdata[j], mysim[j].T )
deepsimlr.corr2_coeff(temp.T,temp2.T)
# print( jnp.trace( jnp.abs( mydot ) )/nev )

