# cookiecutter-pytorch
CookieCutter Template for PyTorch

### Usage

Install cookiecutter using following command before following the rest.
```bash
sudo apt install cookiecutter
```

I am using default values for directory/file names. If edited, replace accordingly.

```bash
cookiecutter https://github.com/daemonslayer/cookiecutter-pytorch --checkout experiments
cd pack
pip install -r requirements.txt
make
```

To start Tensorboard, run:

```bash
tensorboard --logdir="./runs"
```

### License

MIT License

