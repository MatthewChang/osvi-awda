import numpy as np
import math
import numpy as np
from matplotlib import pyplot as plt 
from einops import rearrange, reduce, repeat
from pyutil import argmin
import cv2 as cv
from hem.robosuite.custom_ik_wrapper import normalize_action

num_points = 50
point_spacing = 1
def point_to_width(im,point,vector):
    width = num_points*point_spacing
    for i in range(num_points):
        ind = (point+vector*i*point_spacing).astype(int)
        if (ind >= im.shape).any() or (ind < 0).any() or im[ind[0],ind[1]] == 0:
            width = i*point_spacing
            break
    return width

# takes an object mask and returns the best pixel to grasp at
def find_best_grasp_point(objim):
    pr,pc = np.where(objim > 0)
    all_inds = np.stack((pr,pc),axis=1)
    keep = np.all(all_inds/4 == (all_inds/4).astype(int),axis=1)
    points = all_inds[keep]
    out_left = np.zeros_like(objim)
    out_right = np.zeros_like(objim)
    out_top = np.zeros_like(objim)
    out_bottom = np.zeros_like(objim)
    for point in points:
        # widths.append(point_to_width(objim,point,vector))
        out_left[point[0],point[1]] += point_to_width(objim,point,np.array([0,-1]))
        out_right[point[0],point[1]] += point_to_width(objim,point,np.array([0,1]))
        out_top[point[0],point[1]] += point_to_width(objim,point,np.array([1,0]))
        out_bottom[point[0],point[1]] += point_to_width(objim,point,np.array([-1,0]))
    out = out_left + out_right
    # mask = (out > 0) & (out < 30) & (out_top > 6) & (out_bottom > 6) & (out_left > 0) & (out_left< 10) & (out_right > 0) & (out_right< 10)
    mask = (out_top > 3) & (out_bottom > 3) & (out_left > 0) & (out_left< 20) & (out_right > 0) & (out_right< 20)
    if mask.sum() == 0:
        return None
    center = np.array(objim.shape[:2]) / 2
    inds = np.stack(np.where(mask)).T
    min_ind = np.linalg.norm(inds-center,axis=1).argmin()
    best_point = inds[min_ind]
    return best_point

    # val = np.abs(out_left - out_right)
    # val[~mask] = 100
    # best_point = np.unravel_index(val.argmin(), val.shape)
    bpv = val[best_point[0],best_point[1]]
    if bpv < 100:
        return best_point
    else:
        return None

def estimate_object_pos(state,vis=False,panda=False,mosaic=False):
    fovy=90
    imheight,imwidth = state['img'].shape[:2]
    # imheight = 224
    # imwidth=224
    f = 0.5 * imheight / math.tan(fovy * math.pi / 360)
    cammatrix = np.array(((f, 0, imwidth / 2), (0, f, imheight / 2), (0, 0, 1)))
    center = np.array(state['img'].shape[:2]) / 2
    # center = np.array([224,224])/2

    # Older version passed in all image channels
    # meters = state['img'][:,:,-1]*3.0/255
    # Full eval script passes in depth channel only
    meters = state['img']*3.0/255
    bg_limit = 0.33 if mosaic else 1
    bg = meters >= bg_limit
    median = np.median(meters[bg == False])
    diff = meters-median

    if panda:
        diff_adjusted = diff
    else:
        # adjust for tilted camera
        bottom = -1.54563040e-02
        tip = 2.60747820e-02
        tilt = repeat(np.arange(224),'b -> b s',s=224)/224
        tilt = tilt * (tip-bottom) + bottom
        diff_adjusted = diff-tilt

    # floor_offset = -0.000 if mosaic else -0.01
    floor_offset = -0.01
    floor = diff_adjusted > floor_offset
    gripper_limit = 3 if mosaic else 10
    gripper = meters < gripper_limit*3/255
    objmask = np.logical_not(np.logical_or(floor,gripper))

    # find connected regions and take closest that isn't too big or too small
    num,objim = cv.connectedComponents(objmask.astype(np.uint8),)
    potential_objects = []
    for i in range(1,num+1):
        if (objim == i).sum() < 5000 and (objim == i).sum() > 50:
            potential_objects.append(i)
    if vis:
        # sampled tilt range
        # plt.clf()
        # plt.imshow(diff_adjusted)
        # plt.colorbar()
        # plt.savefig('vis/test.png')
        # from IPython import embed; embed()
        # import pdb; pdb.set_trace()
        plt.clf()
        plt.imshow(meters)
        plt.colorbar()
        plt.savefig('vis/gripping/depth.png')
        plt.clf()
        plt.imshow(np.concatenate((gripper,bg,floor,objmask,objim),axis=0))
        plt.savefig('vis/gripping/objs.png')

    # no objects found
    visim = repeat(objim*255,'h w -> h w 3')
    if mosaic:
        obj_loc = find_best_grasp_point(objim)
        if obj_loc is None:
            if vis:
                return None, visim
            else:
                return None
        else:
            obj_loc = np.array(obj_loc)
    else:
        if len(potential_objects) == 0: 
            if vis:
                return None, objim
            else:
                return None
        obj_locs = [np.stack(np.where(objim == po),axis=1).mean(axis=0) for po in potential_objects]
        ind,el,value = argmin(obj_locs,lambda x: np.linalg.norm(x-center))
        obj_loc = el
    p1,p2 = obj_loc
    d = meters[int(p1),int(p2)]
    u,v = (obj_loc-center)
    pos = state['state'][:3]
    obj = state['state'][4:7]
    x_over_z = u / f
    y_over_z = v / f
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    point3d = np.array((x,y,z))
    if mosaic:
        toworld = np.array([
            [-1,0,0],
            [0,1,0],
            [0,0,-1],
            ])
    else:
        toworld = np.array([
            [0,1,0],
            [1,0,0],
            [0,0,-1],
            ])

    cam_offset = [0,0,-0.01] if mosaic else [0,0,0]
    pos += cam_offset
    if vis:
        visim = cv.circle(visim, (obj_loc[1],obj_loc[0]), 3, (255,0,0),-1)
        cent = center.astype(int)
        visim = cv.circle(visim, (cent[1],cent[0]), 2, (0,255,0),-1)
        return pos + toworld@point3d,visim
    else:
        return pos + toworld@point3d

def action_toward(pos,dest,scale=10):
    diff = dest-pos
    actions = np.clip(diff*scale,-1,1)
    return np.concatenate((actions,[-1]),axis=-1)

class GraspPolicy:
    def __init__(self,panda=False,verbose=False,mosaic=False):
        super().__init__()
        self.stable_frames = 0
        self.centering_frames = 0
        self.desc_frames = 0
        self.close_frames = 0
        self.lift_frames = 0
        self.total_frames = 0
        self.panda=panda
        self.obj_pos = None
        self.tolerance = 0.005 if self.panda else 0.02
        self.verbose = verbose
        self.mosaic=mosaic
        self.centering_frames_limit = 25 if mosaic else 500
        self.stable_frames_limit = 10
        self.prev_diff = 100
        self.lift_lim = 3 if self.panda else 20
        if self.mosaic:
            self.lift_lim = 0
        if panda and not mosaic:
            self.stable_frames_limit = 3

    def act(self,state):
        scale = 10
        if self.panda:
            scale = 1
        if self.mosaic:
            scale = 1
        self.total_frames += 1
        if self.total_frames > 200:
            if self.verbose: print("out of time")
            return [0,0,0,0],True
        pos = state['state'][:3]
        if self.stable_frames < self.stable_frames_limit and self.centering_frames < self.centering_frames_limit:
            # if not self.mosaic or self.obj_pos is None:
            self.obj_pos = estimate_object_pos(state,panda=self.panda,mosaic=self.mosaic)
            if self.verbose:
                print('obj loc ', self.obj_pos)
            if self.obj_pos is None:
                meters = state['img']*3.0/255
                floor_estimate = np.median(meters)
                if floor_estimate < 0.15:
                    return [0,0,0,0],True
                else:
                    diff = 0.14-floor_estimate
                    return [0,0,diff,0],False
            above_obj_pos = self.obj_pos + [0,0,0.15]
            dist = (pos-above_obj_pos)[:2]
            if self.verbose: print(dist)
            if np.linalg.norm(dist) >= self.tolerance:
                if self.verbose: print("centering")
                self.centering_frames += 1
                return action_toward(pos,above_obj_pos,scale=scale),False
            else:
                if self.verbose: print("waiting")
                self.stable_frames += 1
                return action_toward(pos,above_obj_pos,scale=scale),False
        elif self.desc_frames < 20:
            if self.verbose: print("desc")
            self.desc_frames += 1
            if self.panda:
                diff = self.obj_pos[-1]-pos[-1]
                # if mosaic, stop descending on collision
                diff_thresh =0.03 if self.mosaic else 0.03
                if abs(diff) < diff_thresh or (self.mosaic and abs(self.prev_diff) <= abs(diff)):
                    self.desc_frames = 20
                self.prev_diff = diff
                return [0,0,diff,-1],False
            else:
                return [0,0,-1,-1],False
        elif self.close_frames < (1 if self.panda else 20):
            if self.verbose: print("close")
            self.close_frames += 1
            return [0,0,0,1],False
        elif self.lift_frames < self.lift_lim:
            if self.verbose: print("lift")
            self.lift_frames += 1
            return [0,0,1,1],self.lift_frames >= self.lift_lim
        else:
            if self.verbose: print("done")
            return [0,0,0,0],True

class WaypointsPolicy:
    def __init__(self,waypoints,panda=False,mosaic=False,base_actions=False,verbose=False):
        super().__init__()
        self.steps = 0
        self.current_waypoint = 0
        self.try_frames = 0
        self.waypoints = waypoints.copy()
        self.dests = self.waypoints[:,:3].copy()
        self.grasp_endpoint = self.waypoints[:,-1] > 0.1
        #primittive acts as a function of params and next primitives desired start configuration
        # moves destination up if start of grasp
        for i,(se,ee) in enumerate(zip(self.grasp_endpoint[:-1],self.grasp_endpoint[1:])):
            if ee and not se:
                self.dests[i] += [0,0,0.15]
        # self.dests += [0,0,0.5]
        self.held=False
        self.grasping = False
        self.panda = panda
        self.verbose=verbose
        self.grasp_policy = GraspPolicy(panda=self.panda,verbose=self.verbose,mosaic=mosaic)
        self.tol = 0.02 if self.panda else 0.01
        self.try_limit = 10 if self.panda else 500/len(waypoints)
        self.mosaic = mosaic
        self.base_actions = base_actions

    def process_action(self,cur_pos,action):
        if not self.base_actions: 
                return action
        action = np.concatenate(
            [action[:3] + cur_pos, [0.296875, 0.703125, 0.703125, 0.0], action[-1:]]
        )
        action = normalize_action(action)
        # mimic hardcoding from ost paper, fixes orientation of gripper
        action[3:7] = [0.296875, 0.703125, 0.703125, 0.0]
        return action

    def act(self,state):
        pos = state['state'][:3]
        done = False
        if self.current_waypoint >= len(self.waypoints):
            act = [0]*4
            done = True
        else:
            act = self._act(state)
            done = False
        return self.process_action(pos,act), done

    def _act(self,state):
        if self.current_waypoint >= len(self.waypoints):
            return ([0]*3)+[-1]
        if self.grasping:
            action,done = self.grasp_policy.act(state)
            if done:
                self.grasping = False
                self.held = True
            return action
        # get current waypoint and compute destination
        dest = self.dests[self.current_waypoint]
        grip = self.grasp_endpoint[self.current_waypoint]
        if grip and not self.held:
            self.grasping = True
            self.grasp_policy = GraspPolicy(panda=self.panda,mosaic=self.mosaic,verbose=self.verbose)
            return self._act(state)
        # if at destination, start next action, else move to destination
        pose = state['state'][:3]
        if np.linalg.norm(dest-pose) <= self.tol or self.try_frames >= self.try_limit:
            self.try_frames = 0
            self.current_waypoint += 1
            return self._act(state)
        self.try_frames += 1
        action = action_toward(pose,dest,scale=1 if self.panda else 10)
        action[-1] = 1 if grip else -1
        return action
