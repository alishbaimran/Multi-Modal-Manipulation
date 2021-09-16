# Multi-Modal-Manipulation

We use self-supervision to learn a compact and multimodal representation of our sensory inputs, which can then be used to improve the sample efficiency of our policy learning. We train a policy in PyBullet (on the a Kuka LBR iiwa robot arm) using PPO for peg-in-hole tasks. This implementation can also be used to understand force-torque (F/T) control for contact-rich manipulation tasks as at each step the force-torque (F/T) reading is captured at the joint connected with the end-effector.

How it works: it uses self-supervision to learn a compact and multimodal representation of our sensory inputs. This can improve the sample efficiency of our policy learning. We train a policy in #PyBullet (on a Kuka LBR iiwa robot arm) using PPO for peg-in-hole tasks.



This implementation can also be used to understand force-torque (F/T) control for contact-rich manipulation tasks as at each step the force-torque (F/T) reading is captured at the joint connected with the end-effector.

# Instructions 

To add Robotiq see:
https://github.com/Alchemist77/pybullet-ur5-equipped-with-robotiq-140/blob/master/urdf/robotiq_140_gripper_description/urdf/robotiq_140.urdf

To add S-RL Toolbox see: https://s-rl-toolbox.readthedocs.io/en/latest/

1. Download the project-master folder. Note: you will need anaconda to run this program. I recommend installing it, creating/initializing an environment, installing python 3.6 in it as that is the version you'll need.
2. cd into the folder on your local laptop
3. Run `pip install -r requirements.txt`. You will also have to install pybullet `pip install pybullet`, gym `pip install gym`, opencv `pip install opencv-python`, pytorch  `pip install torch torchvision`. 
4. Run `python train_peg_insertion.py` to train the agent. If you get any errors, you might have to change any paths that are specified for me to your own.
5. To collect the multimodal dataset for encoder pre-train run `python environments/kuka_peg_env.py`.You will be able to get more data by changing the random seed.
6. To pre-train the fusion encoder run `python multimodal/train_my_fusion_model.py` You have to specify the path to the root directory of multimodal dataset.

**Quick Notes:**
- This code was built on from the implementation here https://github.com/Henry1iu/ierg5350_rl_course_project
- DDPG code implementation: https://github.com/ghliu/pytorch-ddpg
