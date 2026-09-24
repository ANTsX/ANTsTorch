"""DiReCT cortical-thickness estimation.

The tensor implementation of DiReCT lives in this namespace so its numerical
core can be used independently of the higher-level segmentation workflows in
``antstorch.utilities.cortical_thickness``.  Public functions will be exported
here as the ITK-compatible port is added.
"""

from .bridge import kelly_kapowski
from .core import DiReCTResult, direct_cortical_thickness

__all__ = ["DiReCTResult", "direct_cortical_thickness", "kelly_kapowski"]
