import yaml
import pinocchio as pin
from pinocchio import casadi as cpin
import example_robot_data as robex

def PinConfig(yaml_file_path):
    if isinstance(yaml_file_path, str):
            with open(yaml_file_path, 'r') as file:
                config = yaml.safe_load(file)
    else:
        raise ValueError(f'Invalid file path: {yaml_file_path}')
    
    config = config["robot"]["pinocchio_config"]
    if "robex" in config:
        robot = robex.load(config["robex"])
        model = robot.model
        data = robot.data
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
    elif "urdf_path" in config:
        robot = pin.buildModelFromUrdf(config["urdf_path"])
        model = robot.model
        data = robot.data
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
    else:
        raise ValueError(f'Invalid configuration.')
    
    return robot, model, data, cmodel, cdata