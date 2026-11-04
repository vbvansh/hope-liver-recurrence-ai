"""3D connected components of a binary mask, measured in mm (MCT blob check now, lesion measurement in Step 7).

Sizes use the outer surface of the voxels (their corners), close to pyradiomics' mesh-based diameters: a single
voxel has the size of one voxel, not zero.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

FULL = np.ones((3, 3, 3), bool)  # 26-connectivity, as in the MCT-LTDiag authors' VOI_extraction.py


def max_distance(points):
    """Largest distance between any two points; the convex hull keeps this cheap for big point sets."""
    if len(points) < 2:
        return 0.0
    if len(points) > 50:
        points = points[ConvexHull(points, qhull_options="QJ").vertices]  # QJ also copes with flat point sets
    return float(pdist(points).max())


def corners(idx, spacing):
    """Corner coordinates (mm) of the voxels or pixels at integer indices idx (n, d)."""
    d = idx.shape[1]
    offsets = np.array(np.meshgrid(*[[-0.5, 0.5]] * d, indexing="ij")).reshape(d, -1).T
    return np.unique(((idx[:, None, :] + offsets[None]) * spacing).reshape(-1, d), axis=0)


def measure(idx, spacing, slice_axis):
    """Voxels, volume, slices, in-plane / 3D max and equivalent diameter of the voxels at indices idx (n, 3)."""
    spacing = np.asarray(spacing, float)
    plane = [a for a in range(3) if a != slice_axis]
    slices = np.unique(idx[:, slice_axis])
    inplane = max(max_distance(corners(idx[idx[:, slice_axis] == z][:, plane], spacing[plane])) for z in slices)
    volume = len(idx) * float(np.prod(spacing))
    return {"voxels": len(idx), "volume_mm3": round(volume, 1),
            "slices": len(slices), "slice_first": int(slices[0]), "slice_last": int(slices[-1]),
            "inplane_max_diameter_mm": round(inplane, 1),
            "max3d_diameter_mm": round(max_distance(corners(idx, spacing)), 1),
            "equivalent_diameter_mm": round((6 * volume / np.pi) ** (1 / 3), 1)}


def components(mask, spacing, slice_axis):
    """Label array and one dict per component (see measure).

    spacing is given per array axis (mm); slice_axis is the array axis across the slices.
    """
    labels, n = ndimage.label(mask > 0, structure=FULL)
    out = []
    for k, box in enumerate(ndimage.find_objects(labels), 1):
        idx = np.argwhere(labels[box] == k) + [s.start for s in box]
        out.append({"component": k, **measure(idx, spacing, slice_axis)})
    return labels, out
