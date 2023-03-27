# extra files for hacking stable-baselines3

### current file for up-to-date algorithm:
`stable_baselines3/common/research_method_8.py` <br>
* boundary sampling
* easy -> hard and hard -> easy transitions
* additional logging
* test environment evaluation
* updated boundary thresholds

### scripts
`eval_levels.py`: evaluating performance on # of levels <br>
`finetune_procgen.py`: finetuning pretrained agents on specific levels <br>
`train_procgen_augmented.py`: apply latest algorithm for training procgen agents <br>
`train_procgen.py`: training vanilla procgen agents <br>

### stable_baselines3/common
* research_method_1.py
* research_method_2.py
* research_method_3.py

### stable_baselines3/ppo
* ppo_research_method_1.py
* ppo_research_method_2.py
* ppo_research_method_3.py

