# Extra files for hacking stable-baselines3

# How to run

## Global args (used for most/all scripts)
```
# PPO parameters (defaults are used by openai)
parser.add_argument('--n-steps', type=int, default=256)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--n-epochs', type=int, default=3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--clip-range', type=float, default=0.2)
parser.add_argument('--clip-range-vf', type=bool, default=None)
parser.add_argument('--ent-coef', type=float, default=0.01)
parser.add_argument('--vf-coef', type=float, default=0.5)
parser.add_argument('--target-kl', type=float, default=0.01)
parser.add_argument('--normalize-advantage', type=bool, default=True)
parser.add_argument('--max-grad-norm', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--device', type=int, default=0)

# environment args
parser.add_argument('--env-name', type=str, default='coinrun')         # environment name
parser.add_argument('--num-envs', type=int, default=64)                # number of vectorized environments
parser.add_argument('--num-levels', type=int, default=500)             # number of levels to train on
parser.add_argument('--start-level', type=int, default=0)              # start level
parser.add_argument('--distribution-mode', type=str, default='hard')   # easy, hard, extreme, etc
parser.add_argument('--max-global-steps', type=int, default=200000000) # number of training steps
```

## Regular training
**train_procgen.py**
```
# saving args
parser.add_argument('--log', type=bool, default=False) # for tensorboard 
parser.add_argument('--logging-path', type=str, default=None) # logging folder path for tensorboard
parser.add_argument('--save-path', type=str, default='agent.pth') # save path for saving final weights
```
**example**
```
nohup python train_procgen.py --device=0 --env-name='jumper' --log=True --logging-path=regular_ppo_jumper --save-path=weights/regular_ppo_jumper_1
```

## Finetuning 
**finetune_procgen.py**
```
parser.add_argument('--log', type=bool, default=False)
parser.add_argument('--logging-path', type=str, default=None)
parser.add_argument('--save-path', type=str, default='agent.pth')
parser.add_argument('--pretrained-weights-path', type=str, default='weights/') # path to pretrained weights to load before finetuning
parser.add_argument('--hard-levels-path', type=str, default=None)              # path to a pickle file, which is a list of hard levels to finetune on
```
*note: finetune_procgen.py will automatically assign a prefix to all weights when saving, i.e a save-path=weights/jumper_finetune/ppo_jumper will give files in the form of weights/jumper_finetune/ppo_jumper_1_finetune, weights/jumper_finetune/ppo_jumper_2_finetune,...* <br>
*note2: creating the initial hard_levels list, you can run eval_levels.py then iterate through the resulting metrics list and filtering by <= 5.0 average reward and saving it into a pickle file* <br>
**example**
```
nohup python finetune_procgen.py --save-path=weights/jumper_finetune/ppo_jumper --pretrained-weights-path=weights/jumper_pretrain_500 --hard-levels-path=helpers/jumper_pretrain_500_hard_levels --log=True --logging-path=jumper_pretrain_500_finetune
```

## Evaluating performance on levels
**eval_levels.py**
```
parser.add_argument('--pretrained-weights-path', type=str, default='weights/')   # path to pretrained weights
parser.add_argument('--n', type=int, default=100)                                # number of times to evaluate a level (number of playthroughs)
parser.add_argument('--device', type=int, default=0)                           
parser.add_argument('--num-eval-levels', type=int, default=1000)                 # number of levels to evaluate on
parser.add_argument('--start-level', type=int, default=0)                        
parser.add_argument('--distribution-mode', type=str, default='hard') 
parser.add_argument('--num-envs', type=int, default=64) 
parser.add_argument('--save-path', type=str, default='metrics/')                 # save path for evaluation; pickle file 
parser.add_argument('--env-name', type=str, default='coinrun')                  
```
**example (eval on 500 levels)**
```
nohup python eval_levels.py --pretrained-weights-path=weights/jumper_pretrain_500 --num-eval-levels=500 --start-level=0 --save-path=metrics/jumper_pretrain_500_eval_levels --env-name=jumper
```
**example (eval on 10000 ood levels)**
```
nohup python eval_levels.py --pretrained-weights-path=weights/jumper_pretrain_500 --num-eval-levels=9500 --start-level=500 --save-path=metrics/jumper_pretrain_500_ood_eval_levels --env-name=jumper
```

## Setting up methods for training with research algorithm
**init_research_methods.py**
```
parser.add_argument('--pretrained-weights-path', type=str, default=None)       # file path for pretrained agent weights
parser.add_argument('--finetuned-weights-path', type=str, default=None)        # directory path to finetuned agent weights

parser.add_argument('--device', type=int, default=0)                    
parser.add_argument('--env-name', type=str, default=None)
parser.add_argument('--irm-save-path', type=str, default=None)                 # save path for all-trajectory-dict (pickle file)
parser.add_argument('--unplayable-levels-save-path', type=str, default=None)   # save path for unplayable levels (pickle file)
parser.add_argument('--revised-hard-levels-save-path', type=str, default=None) # save path for revised hard levels (pickle file)
```
*note: init_research_methods will automatically append the prefix that finetune_procgen.py applies when saving finetuned weights, so just use the same path that you set for the save-path argument in finetune_procgen.py script* <br>
**example**
```
nohup python init_research_methods.py --pretrained-weights-path=weights/jumper_pretrain_500 --finetuned-weights-path=weights/jumper_finetune/ppo_jumper --env-name=jumper --irm-save-path=helpers/jumper_pretrain_500_all_level_trajectories --unplayable-levels-save-path=helpers/jumper_pretrain_500_unplayable_levels --revised-hard-levels-path=helpers/jumper_pretrain_500_revised_hard_levels
```
## Training with algorithm
**train_procgen_algorithm8.py**
```
# saving/logging args
parser.add_argument('--log', type=bool, default=False) 
parser.add_argument('--logging-path', type=str, default=None)
parser.add_argument('--save-path', type=str, default='agent.pth')         # save path for weights
parser.add_argument('--csv-path', type=str, default=None)                 # filepath for csv logging 
parser.add_argument('--checkpoint-path', type=str, default=None)          # save path for weights checkpoints

# algorithm specific args 
parser.add_argument('--hard-levels-path', type=str, default=None)         # file path for revised hard levels list (obtained from init_research_methods.py)
parser.add_argument('--load-existing-alt', type=str, default=None)        # file path for all level trajectories dict  
parser.add_argument('--load-unplayable-levels', type=str, default=None)   # file path for unplayable levels
parser.add_argument('--pretrained-weights-path', type=str, default=None)  # file path for pretrained weights path to continue training on

parser.add_argument('--p', type=float, default=0.5)                       # probability sampling parameter
```
**example**
```
nohup python train_procgen_algorithm8.py --save-path=weights/jumper_rm8 --csv-path=csvs/jumper_rm8 --checkpoint-path=weights/jumper_rm8_checkpoints --hard-level-path=helpers/jumper_pretrain_500_revised_hard_levels --load-existing-alt=helpers/jumper_pretrain_500_all_level_trajectories --load-unplayable-levels=helpers/jumper_pretrain_unplayable_levels --pretrained-weights-path=weights/jumper_pretrain_500 --p=0.5 
```
**train_procgen_algorithm9.py**
```
# only additional arg (rest is the same as algorithm8)
# multiplier for determining max length an agent can play for some level depending on length of demonstration trajectory
parser.add_argument('--max-steps-multiplier', type=int, default=2) 
```



