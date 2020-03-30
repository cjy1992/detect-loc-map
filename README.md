# End-to-end Autonomous Driving Perception
[[Project webpage]](https://sites.google.com/berkeley.edu/e2e-percep/) [[Paper]](https://arxiv.org/abs/2003.12464)

This repo contains code for [End-to-end Autonomous Driving Perception with Sequential Latent Representation Learning](https://arxiv.org/abs/2003.12464). This work introduces a novel end-to-end approach for autonomous driving perception. A latent space is introduced to capture all relevant features useful for perception, which is learned through sequential latent representation learning. The learned end-to-end perception model is able to solve the detection, tracking, localization and mapping problems altogether with only minimum human engineering efforts and without storing any maps online. The proposed method is evaluated in a realistic urban driving simulator (CARLA simulator), with both camera image and lidar point cloud as sensor inputs.

## System Requirements
- Ubuntu 16.04
- NVIDIA GPU with CUDA 10. See [GPU guide](https://www.tensorflow.org/install/gpu) for TensorFlow.

## Installation
1. Setup conda environment
```
$ conda create -n env_name python=3.6
$ conda activate env_name
```

2. Install the interpretable end-to-end driving package following the installation steps 2-4 in [https://github.com/cjy1992/interp-e2e-driving](https://github.com/cjy1992/interp-e2e-driving).

3. Clone this git repo to an appropriate folder
```
$ git clone https://github.com/cjy1992/detect-loc-map.git
```

4. Enter the root folder of this repo and install the package:
```
$ pip install -r requirements.txt
$ pip install -e .
```

## Usage
1. Enter the CARLA simulator folder and launch the CARLA server by:
```
$ ./CarlaUE4.sh -windowed -carla-port=2000
```
You can use ```Alt+F1``` to get back your mouse control.
Or you can run in non-display mode by:
```
$ DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000
```
It might take several seconds to finish launching the simulator.

2. Enter the root folder of this repo and run:
```
$ ./run_train_eval.sh
```
It will then connect to the CARLA simulator, collect driving data, then train and evaluate the agent. Main Parameters are stored in ```params.gin```. 

3. Run `tensorboard --logdir logs` and open http://localhost:6006 to view training and evaluation information.

## Trouble Shootings
1. If out of system memory, change the parameter ```replay_buffer_capacity``` and ```initial_collect_steps``` the function ```tran_eval``` smaller.

2. If out of CUDA memory, set parameter ```model_batch_size``` or ```sequence_length``` of the function ```tran_eval``` smaller.

## Citation
If you find this useful for your research, please use the following.

```
@article{chen2020perception,
  title={End-to-end Autonomous Driving Perception with Sequential Latent Representation Learning},
  author={Chen, Jianyu and Xu, Zhuo and Tomizuka, Masayoshi},
  journal={arXiv preprint arXiv:2003.12464},
  year={2020}
}
```
