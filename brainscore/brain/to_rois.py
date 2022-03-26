
from pathlib import Path

import mne
import nibabel as nib
import numpy as np

from .. import paths


def split_parc(xyz, label, n, axis="y"):

    axes = dict(x=0, y=1, z=2)

    m = xyz[:, axes[axis]].min()
    M = xyz[:, axes[axis]].max()
    bounds = (M - m) * np.linspace(0, 1.0, 1 + n) + m

    groups = np.digitize(xyz[:, axes[axis]], bounds)

    labels = list()
    for group_id, _ in enumerate(bounds):
        label_ = label.copy()
        label_.name = f"{group_id}_" + label.name
        label_.vertices = label.vertices[groups == group_id]
        labels.append(label_)
    return labels


def split_labels(
    all_labels,
    subjects_dir=paths.base_dir / "derivatives/freesurfer/",
    fsaverage="fsaverage6",
    surf="pial",
    max_vox=250,
):

    areas = [
        "-".join(k.name.split("-")[:-1]) for k in all_labels
    ]  # if "Networks" in i]
    areas = np.unique(areas)

    surf = Path(subjects_dir) / fsaverage / "surf" / f"%s.{surf}"
    xyz = {
        "rh": nib.freesurfer.read_geometry(str(surf) % "rh")[0],
        "lh": nib.freesurfer.read_geometry(str(surf) % "lh")[0],
    }
    new_labels = []
    for area in areas:
        labels = [k for k in all_labels if k.name in [
            area + "-lh", area + "-rh"]]
        assert len(labels) == 2
        n = max([len(k.vertices) for k in labels])
        if n > max_vox:
            n = n // max_vox
            for hemi, k in zip(["lh", "rh"], labels):
                new = split_parc(xyz[hemi][k.vertices], k, n)
                new_labels.extend(new)
        else:
            new_labels.extend(labels)
    return new_labels


def get_rois(hemi="lh", max_vox=None, fsaverage="fsaverage6"):
    subjects_dir = paths.base_dir / "derivatives/freesurfer/"
    all_labels = mne.read_labels_from_annot(
        fsaverage,
        parc="aparc.a2009s",  # parc='Yeo2011_17Networks_N1000',
        subjects_dir=subjects_dir,
        verbose=False,
    )

    if max_vox:
        all_labels = split_labels(
            all_labels, subjects_dir=subjects_dir, max_vox=max_vox
        )

    rois = {k.name: k.vertices for k in all_labels}
    rois_colors = {
        "-".join(k.name.split("-")[: -1]): k.color for k in all_labels
        if "lh" in k.name}

    # areas = [i.split("-")[0] for i in rois.keys() if "Networks" in i]
    areas = ["-".join(i.split("-")[:-1])
             for i in rois.keys()]  # if "Networks" in i]
    areas = np.unique(areas)

    mapping_rois = {k: rois[f"{area}-{hemi}"]
                    for (k, area) in enumerate(areas)}
    return areas, rois, mapping_rois, rois_colors


def to_rois(x, mapping_rois):
    y = np.zeros((*x.shape[:-1], len(mapping_rois)))
    for j, vert in mapping_rois.items():
        if len(vert):
            y[..., j] = np.nanmean(x[..., vert], -1)
    return y


def to_vox(y, mapping_rois, nvox=40962):
    x = np.zeros((*y.shape[:-1], nvox))
    for j, vert in mapping_rois.items():
        if y[..., j].any():
            x[..., vert] = y[..., j][..., None]
    return x


def bold_to_rois(vox_bold, hemi="L", max_vox=None, space="fsaverage6"):
    hemi = f"{hemi.lower()}h"
    _, _, mapping_rois, _ = get_rois(hemi=hemi, max_vox=max_vox, fsaverage=space)
    roi_bold = to_rois(vox_bold, mapping_rois)
    return roi_bold
