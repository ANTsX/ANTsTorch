# from .tab_simr import tab_simr
# from .tab_simr import whiten
# from .tab_simr import simr_low_rank_frobenius_norm_loss
# from .tab_simr import simr_canonical_correlation_loss
# from .tab_simr import simr_low_rank_frobenius_norm_loss_pj
# from .tab_simr import simr_canonical_correlation_loss_pj
# from .tab_simr import simr_low_rank_frobenius_norm_loss_reg_sparse
# from .tab_simr import simr_canonical_correlation_loss_reg_sparse
# from .tab_simr import simr_absolute_canonical_covariance
# from .tab_simr import orthogonalize_and_q_sparsify
# from .tab_simr import basic_q_sparsify
# from .tab_simr import correlation_regularization_matrices
# from .tab_simr import correlation_regularization_matrices
# from .tab_simr import corr2_coeff
# from .tab_simr import preprocess_signal_for_ica

from .normalizing_simr_flows_whitener import normalizing_simr_flows_whitener
from .apply_normalizing_simr_flows_whitener import apply_normalizing_simr_flows_whitener

from .latent_alignment import pearson_multi
from .latent_alignment import info_nce_multi
from .latent_alignment import barlow_twins_multi
from .latent_alignment import vicreg_multi
from .latent_alignment import hsic_biased
from .latent_alignment import hsic_multi
