import yaml
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import example_robot_data as robex
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class sphere:
    radius: float
    pos: np.ndarray

@dataclass
class RobotConfig:
    model: pin.pinocchio_pywrap_default.Model
    data: pin.pinocchio_pywrap_default.Data
    cmodel: pin.pinocchio_pywrap_casadi.Model
    cdata: pin.pinocchio_pywrap_casadi.Data

    ndof: int = None
    homePosition: np.ndarray = None
    upperPositionLimit: np.ndarray = None
    lowerPositionLimit: np.ndarray = None
    upperVelocityLimit: np.ndarray = None
    lowerVelocityLimit: np.ndarray = None
    upperTorqueLimit: np.ndarray = None
    lowerTorqueLimit: np.ndarray = None

    def __init__(self, yaml_file_path):
        self.model, self.data, self.cmodel, self.cdata = PinConfig(yaml_file_path)
        self.ndof = self.model.nq
        self.upperPositionLimit = self.model.upperPositionLimit
        self.lowerPositionLimit = self.model.lowerPositionLimit

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        Rot = self.data.oMi[-1].rotation
        p = self.data.oMi[-1].translation
        Tran = np.eye(4)
        Tran[:3, :3] = Rot
        Tran[:3, 3] = p
        return Tran, Rot, p

    def error(self, q: np.ndarray, oMdes, JOINT_ID:int = 6) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        dMi = oMdes.actInv(self.data.oMi[JOINT_ID])
        err = pin.log(dMi).vector
        grad = pin.computeJointJacobian(self.model, self.data, q, JOINT_ID)
        return err, grad

    def inverse_kinematics(self, q: np.ndarray, p_target: np.ndarray, Rot_target: np.ndarray) -> np.ndarray:
        eps    = 1e-4
        IT_MAX = 1000
        DT     = 1e-1
        damp   = 1e-12
        oMdes = pin.SE3(Rot_target, p_target)
        i=0
        while True:
            err, J = self.error(q, oMdes)
            if np.linalg.norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v*DT)
            if not i % 10:
                print('%d: error = %s' % (i, err.T))
            i += 1
        if success:
            pass
            #print("Convergence achieved!")
        else:
            raise ValueError('Inverse kinematics has not reached convergence.')
        
        return q
    
    def link_kinematics(self, q):
        if type(q) == np.ndarray:    # numerical
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            JointFrames = [None] * self.model.nq
            for i in range(0, self.model.nq):
                JointFrames[i] = self.data.oMi[i+1]
        else:                        # symbolic
            cpin.forwardKinematics(self.cmodel, self.cdata, q)
            cpin.updateFramePlacements(self.cmodel, self.cdata)
            JointFrames = [None] * self.cmodel.nq
            for i in range(0, self.cmodel.nq):
                JointFrames[i] = self.cdata.oMi[i+1]
        return JointFrames


    def update_robot_spheres(self, robot_spheres, q) -> List[List[sphere]]:
        JointFrames = self.link_kinematics(q)        
        sphere_list = [
            [
                sphere(
                    radius = robot_spheres[i][j].radius,
                    pos = (JointFrames[i].rotation @ np.array(robot_spheres[i][j].pos[:3])) + JointFrames[i].translation
                )
                for j in range(len(robot_spheres[i]))
            ]
            for i in range(len(JointFrames))
        ]
        return sphere_list
    
    def obstacle_distances(self, q, env_spheres, robot_spheres):
        robot_spheres = self.update_robot_spheres(robot_spheres, q)
        dis = []
        for i in range(len(robot_spheres)):
            for j in range(len(robot_spheres[i])):
                robot_sphere = robot_spheres[i][j]
                for k in range(len(env_spheres)):
                    env_sphere = env_spheres[k]
                    dis.append((robot_sphere.pos - env_sphere.pos).T @ (robot_sphere.pos - env_sphere.pos) -  (robot_sphere.radius + env_sphere.radius)**2)
        return dis

    def collision_status(self, q:np.ndarray, env_spheres, robot_spheres) -> bool:
        dis = self.obstacle_distances(q, env_spheres, robot_spheres)
        return any([d < 0 for d in dis])



def PinConfig(yaml_file_path):
    if isinstance(yaml_file_path, str):
            with open(yaml_file_path, 'r') as file:
                config = yaml.safe_load(file)
    else:
        raise ValueError(f'Invalid file path: {yaml_file_path}')
    
    config = config["robot"]["pinocchio_config"]
    if "urdf_path" in config:
        model = pin.buildModelFromUrdf(config["urdf_path"])
        data = model.createData()
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
    elif "robex" in config:
        robot = robex.load(config["robex"])
        model = robot.model
        data = robot.data
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
    else:
        raise ValueError('Invalid configuration.')
    
    return model, data, cmodel, cdata