"""
FDISS - Foveal Disc IoU Scanpath Score

A biologically grounded scanpath similarity metric that models fixations
as foveal discs and computes similarity via bidirectional nearest-neighbour
IoU matching.
"""

import numpy as np


class FDISS:
    """
    Foveal Disc IoU Scanpath Score (FDISS).

    Represents fixations as foveal discs and computes scanpath
    similarity via bidirectional nearest-neighbour IoU matching.

    FDISS is symmetric, bounded in [0,1], equals 1 iff the scanpaths
    are identical, equals 0 iff all pairwise disc distances exceed 2r.
    """

    @staticmethod
    def _disc_iou(d: np.ndarray, r: float) -> np.ndarray:
        """
        Closed-form IoU between pairs of equal-radius discs.

        Parameters
        ----------
        d : np.ndarray
            Euclidean distances between fixation centres.
        r : float
            Foveal disc radius in pixels.

        Returns
        -------
        np.ndarray
            IoU values in [0, 1], same shape as d.
        """
        iou = np.zeros_like(d, dtype=float)

        # Identical centres -> full overlap
        iou[d == 0] = 1.0

        # Partial overlap
        mask = (d > 0) & (d < 2 * r)
        d_m = d[mask]
        intersection = 2 * r**2 * np.arccos(d_m / (2 * r)) - (d_m / 2) * np.sqrt(
            4 * r**2 - d_m**2
        )
        union = 2 * np.pi * r**2 - intersection
        iou[mask] = intersection / union

        # d >= 2r -> no overlap, iou stays 0

        return iou

    @staticmethod
    def _iou_matrix(S1: np.ndarray, S2: np.ndarray, r: float) -> np.ndarray:
        """
        Build the (n x m) IoU matrix between two scanpaths.

        Parameters
        ----------
        S1 : np.ndarray, shape (n, 2)
            Fixation (x, y) coordinates for scanpath 1.
        S2 : np.ndarray, shape (m, 2)
            Fixation (x, y) coordinates for scanpath 2.
        r : float
            Foveal disc radius in pixels.

        Returns
        -------
        np.ndarray, shape (n, m)
        """
        diff = S1[:, None, :] - S2[None, :, :]
        D = np.sqrt((diff**2).sum(axis=-1))
        return FDISS._disc_iou(D, r)

    @staticmethod
    def compute_foveal_radius(px_per_degree: float, degrees: float = 1.0) -> float:
        """
        Compute foveal radius from pixels-per-degree.

        Parameters
        ----------
        px_per_degree : float
            Number of pixels per degree of visual angle.
            For 45 px/degree, this is the standard screen assumption.
        degrees : float
            Foveal extent in degrees (default: 1.0)

        Returns
        -------
        float
            Foveal radius in pixels.
        """
        return px_per_degree * degrees

    def evaluate(
        self,
        S1: np.ndarray,
        S2: np.ndarray,
        r: float,
        return_components: bool = False,
    ) -> dict:
        """
        Compute FDISS between two scanpaths.

        Parameters
        ----------
        S1 : np.ndarray, shape (n, 2)
            Fixation (x, y) coordinates for scanpath 1.
        S2 : np.ndarray, shape (m, 2)
            Fixation (x, y) coordinates for scanpath 2.
        r : float
            Foveal disc radius in pixels.
            Use compute_foveal_radius(45.0) for 1 degree (45px).
        return_components : bool
            If True, also return the per-fixation score arrays.

        Returns
        -------
        dict with keys:
            'fdiss'     : float — harmonic mean of P and R
            'precision' : float — S1 -> S2 coverage
            'recall'    : float — S2 -> S1 coverage
            'scores_S1' : np.ndarray (n,), only if return_components=True
            'scores_S2' : np.ndarray (m,), only if return_components=True
        """
        S1 = np.asarray(S1, dtype=float)
        S2 = np.asarray(S2, dtype=float)

        if S1.ndim != 2 or S1.shape[1] != 2:
            raise ValueError("S1 must be shape (n, 2)")
        if S2.ndim != 2 or S2.shape[1] != 2:
            raise ValueError("S2 must be shape (m, 2)")
        if r <= 0:
            raise ValueError("r must be positive")

        if len(S1) == 0 or len(S2) == 0:
            return {
                "fdiss": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        M = self._iou_matrix(S1, S2, r)

        scores_S1 = M.max(axis=1)
        scores_S2 = M.max(axis=0)

        precision = scores_S1.mean()
        recall = scores_S2.mean()

        denom = precision + recall
        fdiss = (2 * precision * recall / denom) if denom > 0 else 0.0

        result = {
            "fdiss": fdiss,
            "precision": precision,
            "recall": recall,
        }

        if return_components:
            result["scores_S1"] = scores_S1
            result["scores_S2"] = scores_S2

        return result
