import argparse

from models.evaluator import Evaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["spider", "bird"], default="spider")
    parser.add_argument("--level", type=str, choices=["L0", "L1", "L2", "L3"], default="L0")
    parser.add_argument("--model", type=str, choices=["gpt-5.2", "llama-3.3-70B"], default="gpt-5.2")
    args = parser.parse_args()

    DATASET = args.dataset
    LEVEL = args.level
    MODEL = args.model

    ev = Evaluator(dataset=DATASET, level=LEVEL, model=MODEL)
    
    # calculate exa scores
    ev.score_sql()

    # print results
    ev.analyze_exa()
    
