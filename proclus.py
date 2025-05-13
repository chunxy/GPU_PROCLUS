import argparse
import numpy as np
import time
import torch

from python.GPU_PROCLUS import PROCLUS_parallel, GPU_PROCLUS

datasets = {
  "gist": 960,
  "crawl": 300,
  "glove100": 100,
  "audio": 128,
  "video": 1024,
  "sift": 128,
}

parser = argparse.ArgumentParser(description="Clustering script")
parser.add_argument("--name", type=str, required=True, help="Dataset name to process")
parser.add_argument("--nlist", type=int, required=True, help="Number of clusters")
parser.add_argument("--dnew", type=int, required=True, help="Intrinsic dimension")
parser.add_argument("--device", type=str, required=True, help="Using GPU or CPU")
args = parser.parse_args()

if args.name not in datasets:
    raise ValueError(f"Invalid dataset name: {args.name}. Must be one of {list(datasets.keys())}")
if args.device not in ["GPU", "CPU"]:
    raise ValueError(f"Invalid device: {args.device}. Must be one of ['GPU', 'CPU']")

name = args.name
nlist = args.nlist
dold = datasets[name]
dnew = args.dnew

PROCLUS = PROCLUS_parallel if args.device == "GPU" else PROCLUS_parallel

print(f"Running {name}: {dold} -> {dnew}")

file = f"/research/d1/gds/cxye23/datasets/data/{name}_base.float32"
X = np.fromfile(file, dtype=np.float32).reshape((-1, dold))
X = torch.from_numpy(X).to(torch.float32)

elapsed_time = 0
rounds = 1
ks = [nlist]
ls = [dnew]
a = 10
b = 5
min_deviation = 0.7
termination_rounds = 30
print(f"Using {args.device}")
print(X.shape)
for k in ks:
    for l in ls:
        print("k:", k, "l:", l)
        t0 = time.time()
        rs = PROCLUS(X, k, l, a, b, min_deviation, termination_rounds)
        medoids = rs.to("cpu").to_numpy()
        medoids.tofile(f"/research/d1/gds/cxye23/datasets/data/{name}.{k}.{l}.proclus.medoids")
        elapsed_time += time.time() - t0
print("Elapsed time: %.4fs" % elapsed_time)
