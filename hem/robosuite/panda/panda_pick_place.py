from robosuite.environments.panda_pick_place import PandaPickPlace as DefaultPandaPickPlace
from robosuite.environments.panda import PandaEnv
from hem.robosuite.arena.bin_arena import BinsArena
from hem.robosuite.objects.custom_xml_objects import BreadObject, CerealObject, MilkObject, CanObject
from hem.robosuite.tasks.pick_place import PickPlaceTask
from robosuite.models.objects import (
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject,
)
from collections import OrderedDict
from robosuite.utils.mjcf_utils import string_to_array
import robosuite.utils.transform_utils as T
import numpy as np
from einops import repeat
import xml.etree.ElementTree as ET

def mujoco_depth_to_meters(sim, depth):
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image

def process_depth(env, x):
    meters = mujoco_depth_to_meters(env.sim, x)
    return repeat(np.clip(meters, 0, 3) * 255 / 3, "w h -> w h c", c=3)

class PandaPickPlace(DefaultPandaPickPlace):
    def __init__(self, randomize_goal=False, single_object_mode=0, no_clear=False, default_bin=3, force_success=False, hand_depth=False,**kwargs):
        self._randomize_goal = randomize_goal
        self._no_clear = no_clear
        self._default_bin = default_bin
        self.hand_depth = hand_depth
        assert force_success == False, "force not implemented for panda env"
        if randomize_goal:
            assert single_object_mode == 2, "only  works with single_object_mode==2!"
        super().__init__(single_object_mode=single_object_mode, **kwargs)

    def clear_objects(self, obj):
        if self._no_clear:
            return
        super().clear_objects(obj)

    def _get_reference(self):
        super()._get_reference()
        self.target_bin_placements[0,0] -= 0.1
        self.target_bin_placements[0,1] -= 0.02
        self.target_bin_placements[2,0] += 0.025
        self.target_bin_placements[3,0] += 0.015
        if self.single_object_mode == 2:
            self.target_bin_placements = self.target_bin_placements[self._bin_mappings]

    def _reset_internal(self):
        if self.single_object_mode  == 2:
            # randomly target bins if in single_object_mode==2
            self._bin_mappings = np.arange(len(self.object_to_id.keys()))
            if self._randomize_goal:
                np.random.shuffle(self._bin_mappings)
            else:
                self._bin_mappings[:] = self._default_bin
        super()._reset_internal()
    
    def reward(self, action=None):
        if self.single_object_mode == 2:
            return float(self._check_success())
        return super().reward(action)
    
    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        if self.single_object_mode == 2:
            obj_str = str(self.item_names[self.object_id]) + "0"
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            return not self.not_in_bin(obj_pos, self._bin_mappings[self.object_id])
        return super()._check_success()
    
    def _load_model(self):
        PandaEnv._load_model(self)
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = BinsArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([.5, -0.3, 0])

        self.ob_inits = [MilkObject, BreadObject, CerealObject, CanObject]
        self.vis_inits = [
            MilkVisualObject,
            BreadVisualObject,
            CerealVisualObject,
            CanVisualObject,
        ]
        self.item_names = ["Milk", "Bread", "Cereal", "Can"]
        self.item_names_org = list(self.item_names)
        self.obj_to_use = (self.item_names[0] + "{}").format(0)

        lst = []
        for i in range(len(self.ob_inits)):
            ob = self.ob_inits[i]()
            lst.append((str(self.item_names[i]) + "0", ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = PickPlaceTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            []
        )
        for i in range(7):
            el = ET.Element('body',attrib=dict(name=f'indicator{i}',pos="0.45 0 -2"))
            el.append(ET.Element('geom',attrib=dict(pos="0 0 0",size='0.02',rgba=f"0 0 0 1",conaffinity='0')))
            el.append(ET.Element('site',attrib=dict(name=f'indicator{i}',pos="0 0 0",size='0.02',rgba=f"{1-0.1*i} 0 0 1")))
            el.append(ET.Element('joint',attrib=dict(name=f'indicator{i}',type="free",damping='100000')))
            self.model.worldbody.append(el)
            el = ET.Element('body',attrib=dict(name=f'indicator{i}g',pos="0.45 0 -2"))
            el.append(ET.Element('geom',attrib=dict(pos="0 0 0",size='0.02',rgba=f"0 0 0 1",conaffinity='0')))
            el.append(ET.Element('site',attrib=dict(name=f'indicator{i}g',pos="0 0 0",size='0.02',rgba=f"0 {1-0.1*i} 0 1")))
            el.append(ET.Element('joint',attrib=dict(name=f'indicator{i}g',type="free",damping='100000')))
            self.model.worldbody.append(el)
        self.model.place_objects()
        self.model.place_visual()
        self.bin_pos = string_to_array(self.model.bin2_body.get("pos"))
        self.bin_size = self.model.bin_size
        self.waypoints = []

    def set_waypoints(self,waypoints):
        self.waypoints = waypoints

    def render(self,mode='rgb'):
        for wi,waypoint in enumerate(self.waypoints):
            if waypoint[-1] > 0.1:
                self.sim.data.site_xpos[self.sim.model.site_name2id(f'indicator{wi}g')] = waypoint[:3]
            else:
                self.sim.data.site_xpos[self.sim.model.site_name2id(f'indicator{wi}')] = waypoint[:3]
        im = self.sim.render(camera_name=self.camera_name, width=320, height=320)
        im = im[80:,:]
        return im
        # return self.sim.render(camera_name=self.camera_name, width=224, height=224)


    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        
        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
                di['depth'] = di['depth'][:,::-1].copy()
                di['depth'] = ((di['depth'] - 0.95) / 0.05 * 255).astype(np.uint8)
            else:
                di["image"] = camera_obs
            di['image'] = di['image'][:,::-1].copy()

            if self.hand_depth:
                depth = self.sim.render(
                    camera_name="gripperPOV",
                    width=224,
                    height=224,
                    depth=True,
                )[1]
                depth_im = process_depth(self,depth)
                di['hand_cam'] = depth_im[:,:,0]

        # low-level object information
        if self.use_object_obs:

            # remember the keys to collect into object info
            object_state_keys = []

            # for conversion to relative gripper frame
            gripper_pose = T.pose2mat((di["eef_pos"], di["eef_quat"]))
            world_pose_in_gripper = T.pose_inv(gripper_pose)

            for i in range(len(self.item_names_org)):

                # add all obs for now
                # if self.single_object_mode == 2 and self.object_id != i:
                    # Skip adding to observations
                    # continue

                obj_str = str(self.item_names_org[i]) + "0"
                obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_str]])
                obj_quat = T.convert_quat(
                    self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
                )
                di["{}_pos".format(obj_str)] = obj_pos
                di["{}_quat".format(obj_str)] = obj_quat

                # get relative pose of object in gripper frame
                object_pose = T.pose2mat((obj_pos, obj_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(object_pose, world_pose_in_gripper)
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                di["{}_to_eef_pos".format(obj_str)] = rel_pos
                di["{}_to_eef_quat".format(obj_str)] = rel_quat

                object_state_keys.append("{}_pos".format(obj_str))
                object_state_keys.append("{}_quat".format(obj_str))
                object_state_keys.append("{}_to_eef_pos".format(obj_str))
                object_state_keys.append("{}_to_eef_quat".format(obj_str))

            if self.single_object_mode == 1:
                # Zero out other objects observations
                for obj_str, obj_mjcf in self.mujoco_objects.items():
                    if obj_str == self.obj_to_use:
                        continue
                    else:
                        di["{}_pos".format(obj_str)] *= 0.0
                        di["{}_quat".format(obj_str)] *= 0.0
                        di["{}_to_eef_pos".format(obj_str)] *= 0.0
                        di["{}_to_eef_quat".format(obj_str)] *= 0.0

            di["object-state"] = np.concatenate([di[k] for k in object_state_keys])
        
        if self.single_object_mode == 2:
            di['target-box-id'] = self._bin_mappings[self.object_id]

        return di

    def initialize_time(self, control_freq):
        self.sim.model.vis.quality.offsamples = 8
        super().initialize_time(control_freq)


class PandaPickPlaceDistractor(PandaPickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object = None, randomize_goal=True, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        obj = np.random.choice(items) if force_object is None else force_object
        obj = items[obj] if isinstance(obj, int) else obj
        super().__init__(single_object_mode=2, object_type=obj, no_clear=True, randomize_goal=randomize_goal, **kwargs)
        # super().__init__(single_object_mode=0, object_type=obj, no_clear=True, randomize_goal=randomize_goal, **kwargs)
