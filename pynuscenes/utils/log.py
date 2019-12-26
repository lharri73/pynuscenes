import logging
import time

def getLogger(name, console_level='INFO', file_level=None):
    """
    Generate logger with custom formatting. Console and file levels and formatting
    if different. Log is saved to a file only if file_level is provided.
    """
    ## Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel('DEBUG')

    ## Create console handler and formatter
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    c_formatter = ConsoleFormatter()
    ch.setFormatter(c_formatter)
    
    ## Create file handler and formatter if file_level provided
    if file_level:
        fmt = '%(filename)s:%(lineno)d - %(levelname)s - %(message)s'
        fh = logging.FileHandler('file.log')
        fh.setLevel(file_level)
        f_formatter = logging.Formatter(fmt=fmt)
        fh.setFormatter(f_formatter)

    ## Add handlers to the logger
    logger.addHandler(ch)
    if file_level:
        logger.addHandler(fh)

    return logger


class ConsoleFormatter(logging.Formatter):
    """
    Create colorful log messages for the console. Not to be used for file handlers.
    """
    DEBUG     = "\033[34;1mDEBUG\033[0;34;49m: "
    INFO     =  "\033[1;32mINFO\033[0;32;49m: "
    WARN     = "\033[1;33mWARNING\033[0;33;49m: "
    ERROR    = "\033[103;31mERROR\033[31;49m: "
    CRITICAL = "\033[41;97mCRITICAL\033[91;49;1m: "
    RESET = "\033[0m"

    def __init__(self, fmt='%(filename)s:%(lineno)d %(levelname)s:: %(message)s'):
        logging.Formatter.__init__(self, fmt=fmt, datefmt=None, style='%')
    
    def format(self, record):
        curTime = time.localtime()
        curTime = time.strftime("%m/%d/%Y %X", curTime)

        long_prefix = '[{time} {fileName}:{line}] '.format(
            time=curTime,
            fileName=record.filename,
            line=record.lineno
        )
        short_prefix = ''

        if record.levelno == logging.DEBUG:
            ret_s =  self.DEBUG + long_prefix + record.msg + self.RESET
        
        elif record.levelno == logging.INFO:
            ret_s = self.INFO + short_prefix +  record.msg + self.RESET
        
        elif record.levelno == logging.WARN:
            ret_s = self.WARN + long_prefix + record.msg + self.RESET
        
        elif record.levelno == logging.ERROR:
            ret_s = self.ERROR + long_prefix + record.msg + self.RESET
        
        elif record.levelno == logging.CRITICAL:
            ret_s = self.CRITICAL + long_prefix + record.msg + self.RESET
        
        return ret_s


def test_logger():
    logger = getLogger(__name__, console_level='DEBUG')
    logger.debug('This is debug')
    logger.info('This is info')
    logger.warning('This is warning')
    logger.error('This is error')
    logger.critical('This is critical')

if __name__ == '__main__':
    test_logger()