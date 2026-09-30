import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3
import numpy as np

######################### Functions for Plotting in 3D ################################


def draw_3D_vertices(vertices, surfaces=None, surf_color=None, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = plt3.Axes3D(fig)

    if surfaces is None:
        surfaces = [range(len(vertices))]

    for i, surf in enumerate(surfaces):
        tri = plt3.art3d.Poly3DCollection(vertices[surf])

        if surf_color is None:
            face_color = np.random.rand(3)
        else:
            face_color = surf_color[i]

        tri.set_facecolor(face_color)
        tri.set_edgecolor("k")
        ax.add_collection3d(tri)

    x = np.concatenate(vertices[:, :, 0])
    y = np.concatenate(vertices[:, :, 1])
    z = np.concatenate(vertices[:, :, 2])

    set_limits_3D(ax, x, y, z)

    plt.show()
    return ax


def set_limits_3D(ax, x, y, z):

    xlim = np.array((np.amin(x) - 0.01, np.amax(x) + 0.01))
    ylim = np.array((np.amin(y) - 0.01, np.amax(y) + 0.01))
    zlim = np.array((np.amin(z) - 0.01, np.amax(z) + 0.01))

    xy_dif = np.max((np.diff(xlim), np.diff(ylim)))

    xlim = xlim.mean() + (np.array([-0.5, 0.5]) * xy_dif)
    ylim = ylim.mean() + (np.array([-0.5, 0.5]) * xy_dif)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)


try:
    import napari

    NAPARI_AVAILABLE = True
except ImportError:
    napari = None
    NAPARI_AVAILABLE = False


def render_models_napari(models):
    if not NAPARI_AVAILABLE:
        print("napari is not installed. Install with: pip install napari[all]")
        return None
    print("opening - Napari")
    v = napari.current_viewer()
    if v is None:
        v = napari.Viewer()
    v.layers.clear()

    for key in models:
        print(key)

        vertices, faces = models[key]
        print(len(faces))
        surface = (vertices, faces)
        s = v.add_surface(surface)
        s.wireframe.visible = len(faces) < 2000000
        s.name = key
        s.opacity = 1
        s.blend_mode = "translucent"
