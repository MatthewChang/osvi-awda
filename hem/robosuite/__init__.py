def get_env(env_name, action_repeat=10, **kwargs):
    if env_name == 'SawyerPickPlaceDistractor':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceDistractor
        env = SawyerPickPlaceDistractor
    elif env_name == 'SawyerPickPlaceMilk':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceMilk
        env = SawyerPickPlaceMilk
    elif env_name == 'SawyerPickPlaceBread':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceBread
        env = SawyerPickPlaceBread
    elif env_name == 'SawyerPickPlaceCereal':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceCereal
        env = SawyerPickPlaceCereal
    elif env_name == 'SawyerPickPlaceCan':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceCan
        env = SawyerPickPlaceCan
    elif env_name == 'BaxterPickPlaceDistractor':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceDistractor
        env = BaxterPickPlaceDistractor
    elif env_name == 'BaxterPickPlaceMilk':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceMilk
        env = BaxterPickPlaceMilk
    elif env_name == 'BaxterPickPlaceBread':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceBread
        env = BaxterPickPlaceBread
    elif env_name == 'BaxterPickPlaceCereal':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceCereal
        env = BaxterPickPlaceCereal
    elif env_name == 'BaxterPickPlaceCan':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceCan
        env = BaxterPickPlaceCan
    elif env_name == 'PandaPickPlaceDistractor':
        from hem.robosuite.panda.panda_pick_place import PandaPickPlaceDistractor
        env = PandaPickPlaceDistractor
    elif env_name == 'PandaPickPlaceTrainObjs':
        from hem.robosuite.panda.multi_objects import PandaPickPlaceDiverseTrain
        env = PandaPickPlaceDiverseTrain
    elif env_name == 'SawyerPickPlaceTrainObjs':
        from hem.robosuite.sawyer.multi_objects import SawyerPickPlaceDiverseTrain
        env = SawyerPickPlaceDiverseTrain
    elif env_name == 'PandaPickPlaceTestObjs':
        from hem.robosuite.panda.multi_objects import PandaPickPlaceDiverseTest
        env = PandaPickPlaceDiverseTest
    elif env_name == 'SawyerPickPlaceTestObjs':
        from hem.robosuite.sawyer.multi_objects import SawyerPickPlaceDiverseTest
        env = SawyerPickPlaceDiverseTest
    elif env_name == 'PandaPickPlaceTrainPartialObjs':
        from hem.robosuite.panda.multi_objects import PandaPickPlacePartialDiverseTrain
        env = PandaPickPlacePartialDiverseTrain
    elif env_name == 'SawyerPickPlaceTrainPartialObjs':
        from hem.robosuite.sawyer.multi_objects import SawyerPickPlacePartialDiverseTrain
        env = SawyerPickPlacePartialDiverseTrain
    elif env_name == 'PandaPickPlaceTestPartialObjs':
        from hem.robosuite.panda.multi_objects import PandaPickPlacePartialDiverseTest
        env = PandaPickPlacePartialDiverseTest
    elif env_name == 'SawyerPickPlaceTestPartialObjs':
        from hem.robosuite.sawyer.multi_objects import SawyerPickPlacePartialDiverseTest
        env = SawyerPickPlacePartialDiverseTest
    else:
        raise NotImplementedError
    
    from hem.robosuite.custom_ik_wrapper import CustomIKWrapper
    return CustomIKWrapper(env(**kwargs), action_repeat=action_repeat)


from hem.robosuite.gym_wrapper import GymWrapper
from hem.robosuite.mjc_util import postprocess_model_xml
