################################################################################
# Filename    : eval_f1.py
# Description : Calculate F1 score
# Author      : Taeseung Hahn @ LLM Tech
# Exec CMD    : python3 eval_f1.py --ref_path='sample-f1-ref.json' --cand_path='sample-f1-cand.json'
# History     : None
################################################################################

# %% ###########################################################################
# Load Libraries
################################################################################
# Basics
import sys, os, time, argparse, logging
from datetime import datetime
from pathlib import Path
from functools import wraps

# Libraries
import json


# %% ###########################################################################
# Set Global Variables
################################################################################
filename = sys.argv[0].replace(".py", "")
current_dtm = datetime.now().strftime("%y%m%d_%H%M%S")
log_path = Path(f'./logs/{filename}-{current_dtm}.log')


# %% ###########################################################################
# Utility functions
################################################################################
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        logger.info(f'1b[31;20m======== EXECUTED: {func.__name__}{args} {kwargs}')
        time_stt = time.perf_counter()
        wrapped = func(*args, **kwargs)
        elasped = time.perf_counter() - time_stt
        logger.info(f'Elapsed time for function {func.__name__}{args} {kwargs}: {elasped:.4f} seconds')
        return wrapped
    return timeit_wrapper

def pprint(text):
    print(text)
    try: logger.info(text)
    except: pass


# %% ###########################################################################
# Define Functions
################################################################################
def calculate_f1(ref, cand):
    ref, cand = set(ref), set(cand)
    inter = ref & cand
    union = ref | cand

    precision = len(inter) / len(cand)
    recall = len(inter) / len(ref)

    f1 = 0 if len(inter) == 0 else 2*precision*recall / (precision+recall)
    
    return f1

@timeit
def main(args):
    with open(args.ref_path, 'r') as f:
        ref_list = json.load(f)

    with open(args.cand_path, 'r') as f:
        cand_list = json.load(f)

    f1_list = [ calculate_f1(ref, cand) for ref, cand in zip(ref_list, cand_list) ]
    macro_f1 = sum(f1_list)/len(f1_list)
    print(f'Macro F1: {100*macro_f1:5.4f}')



# %% ###########################################################################
# Main
################################################################################
if __name__ == '__main__':
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path' , type=str, required=True, help='')
    parser.add_argument('--cand_path', type=str, required=True, help='')
    args = parser.parse_args()

    # LOGGING
    if not log_path.parent.exists():
        os.makedirs(log_path.parent, exist_ok=True)
    logger = logging.getLogger(f'logger-{filename}-{current_dtm}')
    logger.addHandler(logging.FileHandler(log_path))
    logger.setLevel(logging.INFO)

    # Run main()
    main(args)