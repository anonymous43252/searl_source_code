# searl_sourcce_code

CSource code for SEARL with TD3.

##### Easy setup:

```
git clone https://github.com/anonymous43252/searl_source_code.git
cd searl_source_code
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Eventually, adjust PyTorch version for your personal Cuda setup. To repeat experiments a mujoco license is required. 
A free 30 days test license is available at https://www.roboti.us/license.html.  


##### Start training:

```
python searl/train_population.py --config=searl/config.yml --expt=experiment
```

All results will be written in the folder "experiment".
To change seed, environment or hyperparameters please edit `searl/config.yml`

Due to the parallel execution of the training is an exact reproduction of a performance trajectory not possible. By direct execution of the command above, SEARL runs on HalfCheetah environment and achieves a performance after 1 million environment interaction of 8000 to 10000 with a mean performance of 9500 according to our results reported in Figure 2 in our paper. 

We run each of our experiments on 20 server CPUs with 5 parallel workers for evaluation and RL-training. 




