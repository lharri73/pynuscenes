################################################################################
## Date Created  : Fri Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : Sat Jun 26 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

from context import pynuscenes
import os
import logging
from tqdm import tqdm
from pynuscenes.utils.visualize import *

def test_dataset():
    fig = None
    ## test vehicle coordinates
    dataset = pynuscenes.NuscenesDataset(nusc_path='../data/datasets/nuscenes', 
                                        nusc_version='v1.0-mini', 
                                        split='mini_train',
                                        db_file=None,
                                        coordinates='vehicle',
                                        nsweeps_lidar=1,
                                        nsweeps_radar=1)    
    for sample in tqdm(dataset):
        fig = show_sample_data(sample, coordinates='vehicle', fig = fig)
        input('press enter to continue')
    ## test global coordinates

    dataset = pynuscenes.NuscenesDataset(nusc_path='../data/datasets/nuscenes', 
                                        nusc_version='v1.0-mini', 
                                        split='mini_train',
                                        db_file=None,
                                        coordinates='global',
                                        nsweeps_lidar=1,
                                        nsweeps_radar=1)    
    for sample in tqdm(dataset):
        fig = show_sample_data(sample, coordinates='global', fig=fig)
        input('press enter to continue')


if __name__ == "__main__":
    test_dataset()