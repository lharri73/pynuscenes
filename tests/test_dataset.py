################################################################################
## Date Created  : Fri Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : July 10th, 2019                                            ##
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
    dataset = pynuscenes.NuscenesDataset(nusc_path=dataset_location, 
                                        nusc_version='v1.0-trainval', 
                                        split='train',
                                        db_file=None,
                                        coordinates='vehicle',
                                        nsweeps_lidar=5,
                                        nsweeps_radar=5,
                                        verbose=True)    
    for sample in tqdm(dataset):
        pass
        # fig = show_sample_data(dataset[30], coordinates='vehicle', fig = fig)
        # input('press enter to continue')
    ## test global coordinates

    dataset = pynuscenes.NuscenesDataset(nusc_path=dataset_location, 
                                        nusc_version='v1.0-mini', 
                                        split='mini_train',
                                        db_file=None,
                                        coordinates='global',
                                        nsweeps_lidar=1,
                                        nsweeps_radar=1,
                                        verbose=True)    
    for sample in tqdm(dataset):
        print(sample)
        print('here')
        fig = show_sample_data(sample, coordinates='global', fig=fig)
        input('press enter to continue')


if __name__ == "__main__":
    test_dataset()