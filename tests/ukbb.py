import jax
from jax.config import config
config.update("jax_debug_nans", True)
import deepsimlr
import functools
import numpy as jnp
import sys
import pandas as pd
import optax as optax
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
sp = 0.95 # quantile sparseness
sparseness = [sp,sp,sp]
simlrdata = [jnp.asarray( x0 ), jnp.asarray( x1 ), jnp.asarray( x2 ) ]
regmats = deepsimlr.correlation_regularization_matrices( simlrdata, [qq,qq,qq] )
mysim = deepsimlr.tab_simlr( 
  simlrdata, 
  regmats,
  sparseness,
  deepsimlr.simlr_canonical_correlation_loss_reg_sparse, 
#   deepsimlr.simlr_low_rank_frobenius_norm_loss_reg_sparse,
#  simlr_optimizer=optax.sgd( 0.001, momentum=0.8, nesterov=True ),
  simlr_optimizer=optax.optimistic_gradient_descent( 0.0001 ),
  nev=22, max_iterations=91, positivity=False )
##############################
# write the features out
pd.DataFrame(mysim[0]).to_csv("/tmp/ukt1ev_b.csv")
pd.DataFrame(mysim[1]).to_csv("/tmp/ukdtiev_b.csv")
pd.DataFrame(mysim[2]).to_csv("/tmp/ukrsfev_b.csv")
