source $MYDISK/miniconda3/etc/profile.d/conda.sh
conda activate system

date && python proclus_gpu.py --name video --nlist 10000 --dnew 32
date && python proclus_gpu.py --name crawl --nlist 10000 --dnew 32
date && python proclus_gpu.py --name gist --nlist 10000 --dnew 32
date && python proclus_gpu.py --name glove100 --nlist 10000 --dnew 32