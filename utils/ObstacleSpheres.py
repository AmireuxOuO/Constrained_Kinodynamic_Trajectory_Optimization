import yaml
import torch
from typing import Dict, List, Optional, Tuple, Union
from pinocchio import casadi as cpin
import casadi
import numpy as np
from curobo.geom.types import Sphere, Mesh, Cuboid, Capsule, Cylinder
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel


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
                self.robot_config = config["env"]
        else:
            raise ValueError(f'Invalid file path: {yaml_file_path}')


    def spheres_from_env(self):
        env_sphere_list = []
        for i, ob in enumerate(self.env_config):
            if "pose" in ob:   # format [List[float]] = [x y z qw qx qy qz]
                ob_pose = ob['pose']
            else:
                ob_pose = [0, 0, 0, 1, 0, 0, 0] 

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
        self.env_sphere_list = env_sphere_list
        return env_sphere_list
    

    def spheres_from_robot(self, q0):
        robot_config = self.robot_config['curobo_config']
        if isinstance(robot_config, str):
            robot_config = load_yaml(join_path(get_robot_configs_path(), robot_config))["robot_cfg"]
        if isinstance(robot_config, Dict):
            if "robot_cfg" in robot_config:
                robot_config = robot_config["robot_cfg"]
            robot_config = RobotConfig.from_dict(robot_config, TensorDeviceType())
        
        if isinstance(robot_config.kinematics.collision_spheres, Dict):
            robot_sphere_list =[
                [
                    type(sphere)(
                        radius = sphere_i['radius'],
                        pos = np.array(sphere_i['center'])
                    )
                    for sphere_i in link
                ]
                for link in robot_config.kinematics.collision_spheres
            ]
        else:
            kinematics = CudaRobotModel(robot_config.kinematics)
            robot_sphere_list = kinematics.get_robot_as_spheres(q0)
        self.robot_sphere_list = robot_sphere_list
        return robot_sphere_list
        

    def update_env_spheres(self, _):
        spheres = [
            type(sphere)(
                    radius =  self.env_sphere_list[i].radius,
                    pos = np.array(self.env_sphere_list[i].pose[:3])
            )  
            for i in range(len(self.env_sphere_list))
        ]

        return spheres
            
    def update_robot_spheres(self, q, cmodel, cdata):
        Frames = get_link_kinematics(cmodel, cdata, q)        
        spheres = [
            [
                type(sphere)(
                    radius = self.robot_sphere_list[i][j].radius,
                    pos = (Frames[i].rotation @ np.array(self.robot_sphere_list[i][j].pose[:3])) + Frames[i].translation
                )
                for j in range(len(self.robot_sphere_list[i]))
            ]
            for i in range(len(Frames))
        ]
        
        return spheres
        

def get_link_kinematics(q, cmodel, cdata):
    cpin.framesForwardKinematics(cmodel, cdata, q)
    Frames = [None] * cmodel.nq
    for i in range(1, cmodel.nq+1):
        Frames[i] = cdata.oMi[i]
    return Frames


def get_obstacle_distances(env_spheres, robot_spheres):
    return [
        [
            casadi.vertcat([
                (robot_sphere.pos - env_sphere.pos).T @ (robot_sphere.pos - env_sphere.pos) - 
                (robot_sphere.radius + env_sphere.radius)**2
                for env_sphere in env_spheres
            ])
            for robot_sphere in robot_spheres[i]
        ]
        for i in range(len(robot_spheres))
    ]
            


