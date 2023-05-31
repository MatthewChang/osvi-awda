from collections import OrderedDict
import cv2
import copy
import numpy as np

class LazyDecompressionDict(dict):
    def __init__(self,*args,**kwargs):
        self.decoded = {}
        super().__init__(*args,**kwargs)
    def __getitem__(self, __k):
        if __k in self and __k == 'image':
            if 'image' not in self.decoded:
                item = super().__getitem__(__k)
                item = cv2.imdecode(item, cv2.IMREAD_COLOR)
                self['image'] = item
                self.decoded['image'] = True
        return super().__getitem__(__k)

def _compress_obs(obs):
    if 'image' in obs:
        okay, im_string = cv2.imencode('.jpg', obs['image'])
        assert okay, "image encoding failed!"
        obs['image'] = im_string
    if 'depth' in obs:
        assert len(obs['depth'].shape) == 2 and obs['depth'].dtype == np.uint8, "assumes uint8 greyscale depth image!"
        depth_im = np.tile(obs['depth'][:,:,None], (1, 1, 3))
        okay, depth_string = cv2.imencode('.jpg', depth_im)
        assert okay, "depth encoding failed!"
        obs['depth'] = depth_string
    return obs


# def _decompress_obs(obs):
    # if 'image' in obs:
        # obs['image'] = cv2.imdecode(obs['image'], cv2.IMREAD_COLOR)
    # if 'depth' in obs:
        # obs['depth'] = cv2.imdecode(obs['depth'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    # return obs


def _decompress_obs(obs):
    return LazyDecompressionDict(obs)
    return obs


class Trajectory:
    def __init__(self, config_str=None):
        self._data = []
        self._raw_state = []
        self.set_config_str(config_str)
    
    def append(self, obs, reward=None, done=None, info=None, action=None, raw_state=None):
        """
        Logs observation and rewards taken by environment as well as action taken
        """
        obs, reward, done, info, action, raw_state = [copy.deepcopy(x) for x in [obs, reward, done, info, action, raw_state]]

        obs = _compress_obs(obs)
        self._data.append((obs, reward, done, info, action))
        self._raw_state.append(raw_state)

    @property
    def T(self):
        """
        Returns number of states
        """
        return len(self._data)
    
    def __getitem__(self, t):
        return self.get(t)

    def get(self, t, decompress=True):
        assert 0 <= t < self.T or -self.T < t <= 0, "index should be in (-T, T)"
        
        obs_t, reward_t, done_t, info_t, action_t = copy.deepcopy(self._data[t])
        if decompress:
            obs_t = _decompress_obs(obs_t)
        ret_dict = dict(obs=obs_t, reward=reward_t, done=done_t, info=info_t, action=action_t)

        # for k in list(ret_dict.keys()):
            # if ret_dict[k] is None:
                # ret_dict.pop(k)
        return ret_dict

    def __len__(self):
        return self.T

    def __iter__(self):
        for d in range(self.T):
            yield self.get(d)

    def get_raw_state(self, t):
        assert 0 <= t < self.T or -self.T < t <= 0, "index should be in (-T, T)"
        return copy.deepcopy(self._raw_state[t])

    def set_config_str(self, config_str):
        self._config_str = config_str

    @property
    def config_str(self):
        return self._config_str
