# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

<<<<<<< HEAD
In `imitation/`, the provided you can find expert policies and it's tnn-run conterparts

In 'DAgger_results/', the dagger results as set in task and relative preformance graphs of dagger vs expert vs expert imitation by  (input,64,64,output) nn with tanh activation function

=======
The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.
>>>>>>> e82c0ba0166126de9dfb8be3bc5a2670e178714d
