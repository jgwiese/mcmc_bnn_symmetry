import jax.numpy as jnp
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def gram_schmidt(basis):
    orthogonal_basis = []
    for v in basis:
        if len(orthogonal_basis) == 0:
            orthogonal_basis.append(v.clone() / jnp.linalg.norm(v))
        else:
            v_orthogonal = v.clone()
            for w_orthogonal in orthogonal_basis:
                v_orthogonal -= (w_orthogonal.T @ v) / (w_orthogonal.T @ w_orthogonal) * w_orthogonal
            v_orthogonal /= jnp.linalg.norm(v_orthogonal)
            orthogonal_basis.append(v_orthogonal)
    return jnp.array(orthogonal_basis)


@dataclass
class SettingsCutFigure:
    ax_width: int = 4.0
    ax_height: int = 4.0
    relative_limits = jnp.array([-0.2, 1.2])


class CutFigure:
    def __init__(self, settings: SettingsCutFigure = SettingsCutFigure()):
        self._settings = settings
        self._figure = None
    
    def __del__(self):
        plt.close(self._figure)
    
    def plot(self, log_density, basis, resolution: int = 16):
        if self._figure is not None:
            self._figure.clf()
        self._figure = plt.figure(
            figsize = (
                self._settings.ax_width,
                self._settings.ax_height
            )
        )
        ax = self._figure.add_subplot(1, 1, 1)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        #ax.tick_params(direction="in")

        # data generation
        d = basis.shape[-1]

        u = (basis[1] - basis[0])
        v = (basis[2] - basis[0])
        
        a = basis[0]
        u = u
        v = v - ((u.T @ v) / (u.T @ u)) * u

        axis = jnp.linspace(self._settings.relative_limits[0], self._settings.relative_limits[1], resolution)
        xx, yy = jnp.meshgrid(axis, axis)
        parameters_grid = a + xx[..., jnp.newaxis] @ u[jnp.newaxis, ...] + yy[..., jnp.newaxis] @ v[jnp.newaxis, ...]
        values = log_density(parameters_grid.reshape((-1, d))).reshape((resolution, resolution))
        mappable = ax.contourf(axis, axis, values, 64, cmap="jet")
        #mappable = ax.pcolormesh(axis, axis, values, shading="gouraud", cmap="jet")

        points = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])
        ax.scatter(points[:, 0], points[:, 1], marker="x", c="black")
        for i, point in enumerate(points):
            ax.annotate(r"$\theta^{{*, {}}}$".format(i), point + jnp.array([0.04, 0.0]))
        
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        self._figure.add_axes(ax_cb)

        plt.colorbar(mappable=mappable, cax=ax_cb, format="%.2e")

        return self._figure
