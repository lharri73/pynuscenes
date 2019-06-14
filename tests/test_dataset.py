################################################################################
## Date Created  : Fri Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : Fri Jun 14 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################
from context import pynuscenes
import os
import logging
from tqdm import tqdm
from pynuscenes.utils.visualize import *
def test_dataset():
    dataset = pynuscenes.NuscenesDataset(nusc_path='../data/datasets/nuscenes', 
                                        nusc_version='v1.0-mini', 
                                        split='mini_train',
                                        db_file=None,
                                        coordinates='vehicle',
                                        nsweeps_lidar=1,
                                        nsweeps_radar=1)    
    for sample in tqdm(dataset):
        show_sample_data(sample)


if __name__ == "__main__":
    test_dataset()