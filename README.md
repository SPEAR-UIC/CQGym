# CQSim Reinforcement Learning Module implementation using OpenAi Gym Environment

### Environment :

Python 3.7.x

### Dependencies : 
    gym==0.18.0
    h5py==2.10.0
    Keras==2.0.6
    matplotlib==3.4.1
    numpy==1.19.5
    pandas==1.2.3
    tensorflow==1.14.0

### Install dependencies:

```
pip install -r requirements.txt
```

### Run Module using PG Training:

Model Training can be simply executed using the following command:
```
python cqsim.py
```
This command automatically takes the required metadata from the configuration files in config module. 

You can change the data required for the Simulator and RL model by changing the config file or pass externally along with the python command line call:

```
python cqsim.py -j theta_data.swf -n theta_data.swf -R 1000
```

All the config commands which are applicable in CqSim for managing the simulator are also available in the current implementation.

Along with the commands for simulator, the configuration arguments needed for RL model are as follows :

* **input_dim** : [int] Same as DRAS PG
* **job_info_siz** : [int] Same as DRAS PG
* **is_training** : [int] 1 - if the model should be trained, 0 - otherwise.
* **input_weight_file** : [str] Name of the weights file to be loaded. Just
as in DRAS PG, only the common part of the 2 weights file is
required. “_policy_.h5” and “_predict_.h5” are concatenated
automatically. Weights are not loaded if the parameter is empty or
not provided.
* **output_weight_file** : [str] Name of the weights file to
be saved. Just as in DRAS PG, only the common part of the 2
weights file is required. “_policy_.h5” and “_predict_.h5” are
concatenated automatically. Weights are not saved if the parameter
is empty or not provided.
* **do_render** : [int] 1 - if the rendering graphics should be displayed, 0 - otherwise.

These arguments can be managed externally in the following way :

```
python cqsim.py -j theta_data.swf -n theta_data.swf -R 1000 --is_training 1 --output_weight_file C:/path_to_saving_weights
```
