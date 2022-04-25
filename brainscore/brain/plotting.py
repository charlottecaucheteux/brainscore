import numpy as np

from .. import paths


def plot_brain_map_stats(
    r,
    title="",
    h=0,
    cmap="cold_hot",
    view="lateral",
    thresh=0.001,
    vmax=None,
    symmetric_cbar=True,
):

    from nilearn import plotting

    hemis = ["lh", "rh"]
    hemi = "left" if (h == 0) else "right"
    if vmax is None:
        vmax = np.nanmax(r[np.isfinite(r)])
    plotting.plot_surf_stat_map(
        surf_mesh=str(paths.surf_dir / f"{hemis[h]}.inflated"),
        stat_map=r,
        hemi=hemi,
        # axes=ax,
        # vmax = 1,
        view=view,
        vmax=vmax,
        cmap=cmap,
        symmetric_cbar=symmetric_cbar,
        threshold=thresh,
        bg_map=str(paths.surf_dir / f"{hemis[h]}.sulc"),
        colorbar=True,
        title=title,
    )
