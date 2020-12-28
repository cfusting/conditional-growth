import argparse
import pandas as pd
from scipy.stats import spearmanr, pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--progress-report")
args = parser.parse_args()

df = pd.read_csv(args.progress_report)
steps = df.index.to_series().values

a = df['episode_reward_min'].values
b = df['episode_reward_mean'].values
c = df['episode_reward_max'].values

print(f"Pearson min cor / p: {pearsonr(steps, a)}")
print(f"Pearson mean cor / p: {pearsonr(steps, b)}")
print(f"Pearson max cor / p: {pearsonr(steps, c)}")
print(f"Spearman min cor / p: {spearmanr(steps, a)}")
print(f"Spearman mean cor / p: {spearmanr(steps, b)}")
print(f"Spearman max cor / p: {spearmanr(steps, c)}")
