"""Makeup simulation library"""
from .processor import MakeupProcessor, ProcessorConfig
from .context import ImageContext
from .face import Face

__version__ = "0.1.0"
__all__ = [
    'MakeupProcessor', 
    'ProcessorConfig',  
    'ImageContext',    
    'Face'             
]