# thegame-learning

## Install

- install OpenAI baseline
- install thegame client
- compile thegame server (sync version)

```shell
python3 -m pip install --upgrade git+https://github.com/chengscott/baselines.git
python3 -m pip install --upgrade git+https://github.com/chengscott/thegame.git#subdirectory=client/python
```

## Run

```shell
./run_thegame.py

python3 -m thegame.gui.audience :55666
```