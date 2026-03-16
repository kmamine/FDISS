"""
FDISS - Foveal Disc IoU Scanpath Score

A biologically grounded scanpath similarity metric that models fixations
as foveal discs and computes similarity via bidirectional nearest-neighbour
IoU matching.
"""

from .fdiss import FDISS

__version__ = "0.1.0"
__author__ = "Mohammed Amine Kerkouri, Marouane Tliba, Aladine Chetouani"
__license__ = "MIT"

__all__ = ["FDISS"]
