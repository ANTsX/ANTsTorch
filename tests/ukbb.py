import jax
import deepsimlr
import functools
import numpy as jnp
import sys
import pandas as pd
x0=pd.read_csv("/tmp/ukt1x.csv")
x1=pd.read_csv("/tmp/ukdtix.csv")
x2=pd.read_csv("/tmp/ukrsfx.csv")
qq = 0.8
sp = 0.9
sparseness = [sp,sp,sp]
simlrdata = [jnp.asarray( x0 ), jnp.asarray( x1 ), jnp.asarray( x2 ) ]
regmats = deepsimlr.correlation_regularization_matrices( simlrdata, [qq,qq,qq] )
mysim = deepsimlr.tab_simlr( 
  simlrdata, 
  regmats, 
  sparseness,   
  deepsimlr.simlr_low_rank_frobenius_norm_loss_reg_sparse, 
  nev=10, learning_rate=1, max_iterations=5000 )

# mysimcc = deepsimlr.tab_simlr( [x0,x1,x2], regmats, [0.9,0.9,0.9],   
#   deepsimlr.simlr_canonical_correlation_loss_reg_sparse, 
#  nev=5, learning_rate=0.0005, max_iterations=33 )


# write the features out
