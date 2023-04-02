import argparse
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import procgen
import numpy as np
import pandas as pd
import pickle

from utils.method_utils import *
from models.impala import *


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pretrained-weights-path', type=str, default=None)
    parser.add_argument('--finetuned-weights-path', type=str, default=None)
    
    parser.add_argument('--device', type=int, default=0)                    
    parser.add_argument('--env-name', type=str, default=None)
    parser.add_argument('--irm-save-path', type=str, default=None)
    parser.add_argument('--unplayable-levels-save-path', type=str, default=None)
    parser.add_argument('--revised-hard-levels-save-path', type=str, default=None)
                        
    args = parser.parse_args()
    return args

    
def main(args):
    # intialize dummy env for initializing model (inefficient, prob change the model init method later)
    model = SB_Impala(dummy_env.observation_space['rgb'], dummy_env.action_space.n, 5e-4)
    
    device = torch.device('cuda:' + str(args.device))
    
    with open(args.all_hard_levels + '.pkl', 'rb') as f:
        all_hard_levels = pickle.load(f)
        
    easy_levels = [i for i in range(args.num_levels) if i not in all_hard_levels]
    
    irm = init_all_level_trajectories(
        easy_levels=easy_levels,
        hard_levels=all_hard_levels,
        env_name=args.env_name,
        pretrained_weights_path=args.pretrained_weights_path,
        finetuned_weights_path=args.finetuned_weights_path,
        model=model,
        device=device
    )
                        
    unprocessed_irm = copy.deepcopy(irm)
    with open(args.irm_save_path + '_unprocessed.pkl', 'wb') as f:
        pickle.dump(unprocessed_irm, f)
    
    # some postprocessing + adding boundary sampling since all of our methods so far use boundary sampling and compute 'unplayable levels'
    postprocessed_irm, unplayable_levels = postprocess_dict(irm)
    boundary_sampling_postprocessed_irm = add_boundary_sampling(postprocessed_irm)
    
    revised_hard_levels = [i for i in all_hard_levels if i not in unplayable_levels]
    
    with open(args.irm_save_path + '_bsp.pkl', 'wb') as f: # bsp: boundary sampling + postprocessed
        pickle.dump(boundary_sampling_postprocessed_irm, f)
        
    with open(args.unplayable_levels_save_path + '.pkl', 'wb') as f:
        pickle.dump(unplayable_levels, f)
        
    with open(args.revised_hard_levels_save_path + '.pkl', 'wb') as f:
        pickle.dump(revised_hard_levels, f)
        
        
if __name__ == '__main__':
    args = parse_args()
    main(args)