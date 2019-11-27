from glob import glob
import sys
import pickle

from imitation.util.rollout import rollout_stats

def main(path="data/expert_models/mountain_car_0/rollouts/final.pkl"):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(path)
    print(rollout_stats(obj))

def main_all():
    for path in sorted(glob("data/expert_models/*/rollouts/final.pkl")):
        main(path)
        print()

if __name__ == "__main__":
    main_all()
