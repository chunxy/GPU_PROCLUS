source $MYDISK/miniconda3/etc/profile.d/conda.sh
conda activate system

date && python proclus.py --name video --nlist 10000 --dnew 32
date && python proclus.py --name crawl --nlist 10000 --dnew 32
date && python proclus.py --name gist --nlist 10000 --dnew 32
date && python proclus.py --name glove100 --nlist 10000 --dnew 32