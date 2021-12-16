# CQGym: Gym Environment for Reinforcement-Learned Batch Job scheduling 

All necessary packages can be installed with
```
pip install -r requirements.txt
```
There are many command line options available in the CQGym environment. These can be viewed with
```
python cqsim.py -h
```
The following outlines the most common use cases for CQGym.


## Training and testing
The common options for training a model from scratch. 
```
python cqsim.py -n [str] -j [str] --rl_alg [str] --is_training [1] --output_weight_file [str]
```
The common options for testing a model.
```
python cqsim.py -n [str] -j [str] --rl_alg [str] --is_training [0] --input_weight_file [str]
```
* **-n:** the name of the job trace file present in /data/InputFiles/, such as test.swf.
* **-j:** the same file name used for **-n**. 
* **--rl_alg:** the name of the training algorithm to use. Either PG, DQL, A2C, PPO, or FCFS. Defaults to FCFS.
* **--is_training:** 1 = perform optimization. 0 = No optimization.
* **--output_weight_file:** the file name model weights are saved under. Can be found under /data/Fmt/ at the end of execution.
* **--input_weight_file:** [str]. Specify a file name to load in existing model weights. Should be present in /data/Fmt.

### Other environment options
These options are useful for making a custom training routine using CQGym calls. 
* **-R:** [int]. Specify the number of traces to simulate before stopping. Defaults to 8000.
* **-r:** [int]. Specify job trace starting point as a line number. Defaults to 0.
* **--do_render** : [int] 1 = display graphics, 0 - do not display graphics. Rendered graphics reports training performance within the episode. 

### Training testing example script
Training for two episodes over 1500 job traces.
```
python cqsim.py -j train.swf -n train.swf -R 1500 --is_training 1 --output_weight_file pg0 --rl_alg PG
python cqsim.py -j train.swf -n train.swf -r 1501 -R 1500 --is_training 1 --input_weight_file pg0 --output_weight_file pg1 --rl_alg PG
```
Testing on validation file for 5000 job traces.
```
python cqsim.py -j validate.swf -n validate.swf -R 5000 --is_training 0 --input_weight_file pg0 --rl_alg PG
python cqsim.py -j validate.swf -n validate.swf -R 5000 --is_training 0 --input_weight_file pg1 --rl_alg PG
```

### Learning parameters
Model hyperparameters can be modified using these options.
* **--learning_rate:** [float]. Defaults to 0.000021.
* **--batch_size:** [int]. The number of state-action-value sequences recorded by the agent before performing optimization. Defaults to 70.
* **--window_size:** [int]. Input size. How many jobs from the queue considered by the agent for scheduling. Defaults to 50.
* **--reward_discount:** [float]. Between [0, 1]. Designates the importance of future rewards in future states. Corresponds to gamma in the Bellman Optimality equation. Defaults to 0.95

### Config/
Additionally, all default values can be found and modified in src_fc/Config/.

## Data Collection
Output from training and testing episodes goes to /data/Results.
* **.rst:** Job scheduling results.

| Column | Description     |
| ------ | -----------     |
| 1      | Job ID          | 
| 2      | Processor count |
| 3      | Requested time  |
| 4      | Actual runtime  |
| 5      | Wait time       |
| 6      | Submission time |
| 7      | Start time      |
| 8      | End time        |

* **.ult:** Changes to system utilization.

| Column | Description   |
| ------ | -----------   |
| 1      | Time          |
| 2      | Utilization % |

* **.rwd:** Reward results.

| Column | Description  |
| :----  | :----        |
| 1      | Reward value |
