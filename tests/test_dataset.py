################################################################################
## Date Created  : Fri Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : August 26th, 2019                                          ##
## Copyright (c) 2019                                                         ##
################################################################################

from context import pynuscenes
import os
import logging
from tqdm import tqdm
from pynuscenes.utils.visualize import show_sample_data

def test_dataset():
    dataset_location = '../../../data/datasets/nuscenes'
    fig = None
    
    ## test vehicle coordinates
    mini_dataset_vehicle = pynuscenes.NuscenesDataset(nusc_path=dataset_location, 
                                        nusc_version='v1.0-mini', 
                                        split='mini_train',
                                        db_file=None,
                                        coordinates='vehicle',
                                        nsweeps_lidar=1,
                                        nsweeps_radar=1)    
    for sample in tqdm(mini_dataset_vehicle):
        fig = show_sample_data(sample, coordinates='vehicle', fig = fig)
        input('press enter to continue')
    
    ## test global coordinates
    mini_dataset_global = pynuscenes.NuscenesDataset(nusc_path=dataset_location, 
                                        nusc_version='v1.0-mini', 
                                        split='mini_train',
                                        db_file=None,
                                        coordinates='global',
                                        nsweeps_lidar=1,
                                        nsweeps_radar=1)  
    for sample in tqdm(mini_dataset_global):
        fig = show_sample_data(sample, coordinates='global', fig=fig)
        input('press enter to continue')

##------------------------------------------------------------------------------
if __name__ == "__main__":
    test_dataset()