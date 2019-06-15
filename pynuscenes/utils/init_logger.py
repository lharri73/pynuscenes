################################################################################
## Date Created  : Sat Jun 15 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : Sat Jun 15 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

def initialize_logger(name, verbose=False):
    ## Set up logger
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        if verbose:
            ch.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(filename)s:%(lineno)d %(levelname)s:: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger