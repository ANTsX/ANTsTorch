
# Par ceci :
from .latent_alignment import (
    LatentAlignmentLossManager,
    Projector,
    ScreenState,
    flatten_latents,
)

# Et si vous voulez aussi exposer les pertes historiques d'ANTsTorch :
from .alignment_losses import (
    pearson_multi,
    info_nce_multi,
    barlow_twins_multi,
    vicreg_multi,
    hsic_biased,
    hsic_multi, 
    lpnorm_multi
)


from .scripts.train_lamnr_flows_tabular import TabularLAMNrTrainer
