from enum import Enum
import numpy as np

class ViewTransposeConfig(Enum):
        CORONAL = 0
        SAGITTAL = 1
        AXIAL = 2

class ViewTransformer:
    """
    A transformer that transposes the volume and mask to a specific view.
    
    Args:
        view: The view to transpose the volume and mask to
            - CORONAL: Transpose the volume and mask to the coronal view
            - SAGITTAL: Transpose the volume and mask to the sagittal view
            - AXIAL: Transpose the volume and mask to the axial view
    
    Returns:
        The transformed volume and mask
    
    Example:
        >>> transformer = ViewTransformer(view=View.AXIAL)
        >>> volume, mask = transformer(volume, mask)
        or
        >>> transformer = ViewTransformer(view=View.CORONAL)
        >>> dataset = TDSC(transforms=[transformer])
        >>> volume, mask, label, bbx = dataset[0] # the volume and mask are now in the axial view
        
    Note:
        The default view is the axial view.
    """
    
    @property
    def TRANSPOSE_CONFIGS(self) -> dict[ViewTransposeConfig, tuple[int, int, int]]:
        return {
        ViewTransposeConfig.AXIAL: (0, 1, 2),
        ViewTransposeConfig.CORONAL: (1, 2, 0),
        ViewTransposeConfig.SAGITTAL: (2, 0, 1)
    }
    
    def __init__(self, view: ViewTransposeConfig):
        self.transpose_axes = self.TRANSPOSE_CONFIGS[view]
    
    def __call__(self, vol: np.ndarray, mask: np.ndarray):
        transformed_vol = np.transpose(vol, self.transpose_axes)
        transformed_mask = np.transpose(mask, self.transpose_axes)
        return transformed_vol, transformed_mask