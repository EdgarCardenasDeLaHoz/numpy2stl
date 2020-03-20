
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3

######################### Functions for Plotting in 3D ################################

def plot_edges_3d(edges,ax=None):

    if ax is None:
        fig = plt.figure()
        ax = plt3.Axes3D(fig)

    for n,e in enumerate(edges):
        color = np.random.rand(3)
        ax.plot3D([e[0,0],e[1,0]], [e[0,1],e[1,1]], [e[0,2],e[1,2]], color=color)
        ax.plot3D([e[1,0]], [e[1,1]], [e[1,2]], 'o', color=color)
        #ax.text(e[1,0], e[1,1], e[1,2], str(n), [0, 0, 1])

    x = edges[:,:,0].ravel()
    y = edges[:,:,1].ravel()
    z = edges[:,:,2].ravel()
    
    set_limits_3D(ax,x,y,z)
    
    return ax

def plot_perimeters(perimeter,ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for p in perimeter:
        color = np.random.rand(3)

        p = np.concatenate([p,[p[0]]])
        ax.plot(p[:,0],p[:,1], color=color)

    return ax

def plot_perimeters_3d(perimeter,ax=None):

    if ax is None:
        fig = plt.figure()
        ax = plt3.Axes3D(fig)

    for p in perimeter:
        color = np.random.rand(3)
        ax.plot3D(p[:,0],p[:,1],p[:,2], color=color)

    x = perimeter[0][:,0].ravel()
    y = perimeter[0][:,1].ravel()
    z = perimeter[0][:,2].ravel()

    set_limits_3D(ax,x,y,z)
    
    return ax

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
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)

    x = np.concatenate(vertices[:,:,0])
    y = np.concatenate(vertices[:,:,1])
    z = np.concatenate(vertices[:,:,2])

    set_limits_3D(ax,x,y,z)
    
    plt.show()
    return ax

def set_limits_3D(ax,x,y,z):
    
    xlim = np.array((np.amin(x)-1, np.amax(x)+1))
    ylim = np.array((np.amin(y)-1, np.amax(y)+1))
    zlim = np.array((np.amin(z)-1, np.amax(z)+1))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)  

