################################################################################
# Filename    : eval_rouge.py
# Description : Calculate ROUGE metric
# Author      : Taeseung Hahn @ LLM Tech
# Exec CMD    : python3 eval_rouge.py --ref_path='sample-rouge-ref.json' --cand_path='sample-rouge-cand.json'
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
from rouge_metric import Rouge


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
def prepare_results(metric, p, r, f):
    pprint(f'\t{metric}:\tP: {100.0*p:5.2f}\tR: {100.0 * r:5.2f}\tF: {100.0*f:5.2f}')

@timeit
def main(args):
    with open(args.ref_path, 'r', encoding='utf8') as f:
        ref = json.load(f)

    with open(args.cand_path, 'r', encoding='utf8') as f:
        cand = json.load(f) 

    # evaluator = Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
    evaluator = Rouge(metrics=['rouge-n'],
        max_n=1, # 4
        limit_length=True,
        length_limit=100,
        length_limit_type='words',
        use_tokenizer=True,
        apply_avg=True,
        apply_best=False,
        alpha=0.5, # Default F1_score
        weight_factor=1.2,
    )

    scores = evaluator.get_scores(cand, ref)
    print(f'ROUGE-1 F1: {100.0*scores["rouge-1"]["f"]:5.4f}')

    # for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    #     prepare_results(metric, results['p'], results['r'], results['f'])



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