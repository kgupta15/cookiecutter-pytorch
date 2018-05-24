all: data main

data:
	bash data/download.sh

main:
	python main.py --gpu=1 --resume=0 --chkpts=./checkpoints/ --config=./experiments/config.yaml

profile:
	python -m cProfile -s cumtime main.py --gpu=1 --resume=0 --chkpts=./checkpoints/ --config=./experiments/config.yaml

mem_profile:
	python -m memory_profiler main.py --gpu=1 --resume=0 --chkpts=./checkpoints/ --config=./experiments/config.yaml
