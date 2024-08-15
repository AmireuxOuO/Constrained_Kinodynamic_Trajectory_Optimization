import yaml
import torch
from typing import Dict, List, Optional, Tuple, Union
from pinocchio import casadi as cpin
import casadi
import numpy as np
from curobo.geom.types import Sphere, Mesh, Cuboid, Capsule, Cylinder
from curobo.geom.sphere_fit import fit_spheres_to_mesh
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from dataclasses import dataclass

@dataclass
class sphere:
    radius: float
    pos: np.ndarray


class ObsConfig:
    def __init__(self, yaml_file_path):
        if isinstance(yaml_file_path, str):
            with open(yaml_file_path, 'r') as file:
                config = yaml.safe_load(file)
            if "robot" in config:
                self.robot_config = config["robot"]
            if "env" in config:
                self.env_config = config["env"]
        else:
            raise ValueError(f'Invalid file path: {yaml_file_path}')

    def spheres_from_robot(self, q0=None):
        robot_config = self.robot_config['curobo_config']
        if isinstance(robot_config, str):
            robot_config = load_yaml(join_path(get_robot_configs_path(), robot_config))["robot_cfg"]
        if isinstance(robot_config, Dict):
            if "robot_cfg" in robot_config:
                robot_config = robot_config["robot_cfg"]
        
        if robot_config["kinematics"].get("collision_spheres") is not None:
            if isinstance(robot_config["kinematics"]["collision_spheres"],Dict):
                dict_collision_spheres = robot_config["kinematics"]["collision_spheres"]
            elif isinstance(robot_config["kinematics"]["collision_spheres"],str):
                dict_collision_spheres = load_yaml(join_path(get_robot_configs_path(), 
                                                        robot_config["kinematics"]["collision_spheres"]))["collision_spheres"]
            else:
                raise ValueError(f'Invalid format for collision_spheres')

            for _, link in dict_collision_spheres.items():
                print(link)

            robot_sphere_list =[
                [
                    sphere(
                        radius = sphere_i['radius'],
                        pos = np.array(sphere_i['center'])
                    )
                    for sphere_i in link  # sphere_i is a dict
                ]
                for _, link in dict_collision_spheres.items() # link is a list
            ]

        else:
            robot_config = RobotConfig.from_dict(robot_config, TensorDeviceType())
            kinematics = CudaRobotModel(robot_config.kinematics)
            
            robot_mesh_list = kinematics.get_robot_as_mesh(torch.tensor(q0).to('cuda'))  # List[Mesh], type Mesh in cuRobo
            robot_sphere_list = [None]*len(robot_mesh_list)

            for i in range(len(robot_mesh_list)):
                trimesh = robot_mesh_list[i].get_trimesh_mesh()
                n_pts, n_radius = fit_spheres_to_mesh(mesh=trimesh, n_spheres=2)
                print(n_radius)
                robot_sphere_list[i] = [sphere(radius = n_radius[i],pos = np.array(n_pts[i]))
                        for i in range(2)]  # sphere_i is a dict
            
        self.robot_sphere_list = robot_sphere_list
        return robot_sphere_list
        

    def update_robot_spheres(self, q, cmodel, cdata):
        Frames = get_link_kinematics(q, cmodel, cdata)        
        spheres = [
            [
                sphere(
                    radius = self.robot_sphere_list[i][j].radius,
                    pos = (Frames[i].rotation @ np.array(self.robot_sphere_list[i][j].pos[:3])) + Frames[i].translation
                )
                for j in range(len(self.robot_sphere_list[i]))
            ]
            for i in range(len(Frames))
        ]
        return spheres

    def spheres_from_env(self):
        env_sphere_list = []
        for i, ob in enumerate(self.env_config):
            if "pose" in ob:   # format [List[float]] = [x y z qw qx qy qz]
                ob_pose = ob['pose']
            else:
                ob_pose = [0, 0, 0, 1, 0, 0, 0] 

            if ob['type'] == 'Sphere':
                obj = sphere(pos = np.array(ob_pose[:3]), radius = ob['dim'])  # radius: float
                env_sphere_list.append(obj)
            else:
                raise ValueError(f'Unsupported type {ob["type"]}')
            
            """
            Use cuRobo dataclass
            if ob['type'] == 'Sphere':
                obj = Sphere(name = 'sph_'+str(i), pose = ob_pose, radius = ob['dim'])  # radius: float
                env_sphere_list.append(obj)
            elif ob['type'] == 'Mesh':
                obj = Mesh(name = 'mesh_'+str(i), pose = ob_pose, file_path = ob['file_path'])
                env_sphere_list.append( new_sphere for new_sphere in obj.get_bounding_spheres(n_spheres=1) )
            elif ob['type'] == 'Cuboid':
                obj = Cuboid(name = 'cub_'+str(i), pose = ob_pose, dims = ob['dim'])  # dims: List[float]
                env_sphere_list.append( new_sphere for new_sphere in obj.get_bounding_spheres(n_spheres=1) )
            elif ob['type'] == 'Capsule':
                obj = Capsule(name = 'cap_'+str(i), pose = ob_pose, radius = ob['dim'][0], base = ob['dim'][1], tip = ob['dim'][2]) # radius: float, base: List[float], tip: List[float]
                env_sphere_list.append( new_sphere for new_sphere in obj.get_bounding_spheres(n_spheres=1) )
            elif ob['type'] == 'Cylinder':
                obj = Cylinder(name = 'cyl_'+str(i), pose = ob_pose, radius = ob['dim'][0], height = ob['dim'][1]) # radius: float, height: float
                env_sphere_list.append( new_sphere for new_sphere in obj.get_bounding_spheres(n_spheres=1) )
            else:
                raise ValueError(f'Unsupported type {ob["type"]}')
            """
        self.env_sphere_list = env_sphere_list
        return env_sphere_list

    def update_env_spheres(self, _):
        spheres = [
            type(sphere)(
                    radius =  self.env_sphere_list[i].radius,
                    pos = np.array(self.env_sphere_list[i].pose[:3])
            )  
            for i in range(len(self.env_sphere_list))
        ]

        return spheres
            

def get_link_kinematics(q, cmodel, cdata):
    cpin.framesForwardKinematics(cmodel, cdata, q)
    Frames = [None] * cmodel.nq
    for i in range(0, cmodel.nq):
        Frames[i] = cdata.oMi[i]
    return Frames


def get_obstacle_distances(env_spheres, robot_spheres):
    res = [ [None] * len(robot_spheres[i]) for i in range(len(robot_spheres)) ]
    for i in range(len(robot_spheres)):
        for j in range(len(robot_spheres[i])):
            robot_sphere = robot_spheres[i][j]
            dis = []
            for k in range(len(env_spheres)):
                env_sphere = env_spheres[k]
                dis.append((robot_sphere.pos - env_sphere.pos).T @ (robot_sphere.pos - env_sphere.pos) -  (robot_sphere.radius + env_sphere.radius)**2)
            
            res[i][j] = casadi.vertcat(*dis)
    return res
            


