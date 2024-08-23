import numpy as np
import matplotlib.pyplot as plt

def plot_points(axes, points, color='black', s = 10, label=None):
    axes.scatter(points[:,0], points[:,1], points[:,2], color=color, s= s, label=label)

def plot_traj(axes, traj, color='blue', label=None, axis_label=None):
    axes.plot(traj[:,0], traj[:,1], traj[:,2], color=color, label=label)
    if axis_label is not None:
        axes.set_xlabel(axis_label[0])
        axes.set_ylabel(axis_label[1])
        axes.set_zlabel(axis_label[2])


def plot_robot_spheres(axes, sphere_list, color='blue', alpha=0.5):
    for i in range(len(sphere_list)):
            for j in range(len(sphere_list[i])):
                u = np.linspace(0, 2 * np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                center = sphere_list[i][j].pos
                radius = sphere_list[i][j].radius
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                axes.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_env_spheres(axes, sphere_list, color='blue', alpha=0.5):
    for i in range(len(sphere_list)):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        center = sphere_list[i].pos
        radius = sphere_list[i].radius
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        axes.plot_surface(x, y, z, color=color, alpha=alpha)