# Tabular Q-Learning on FrozenLake

A small Python project that trains a tabular Q-learning agent on Gymnasium's `FrozenLake-v1` environment. It stores action values in a Python dictionary, samples actions during training, and evaluates the current policy with greedy actions.

## How it works

The agent learns a value (Q(s,a)) for each state and action using the Q-learning update:

```text
Q(s, a) <- (1 - alpha) * Q(s, a)
           + alpha * (reward + gamma * max_a' Q(s', a'))
```

The current settings in `frozenlake.py` are:

| Setting | Value |
| --- | ---: |
| Environment | `FrozenLake-v1` |
| Learning rate (`ALPHA`) | `0.2` |
| Discount factor (`GAMMA`) | `0.9` |
| Evaluation episodes per update | `20` |

Training selects actions uniformly at random. After each update, the agent is evaluated over 20 episodes using the best known action for each state. Training stops when the average evaluation reward is greater than `0.8`.

## Requirements

- Python 3.9 or later
- Gymnasium
- PyTorch and TensorBoard

Install the dependencies with:

```bash
python -m pip install gymnasium torch tensorboard
```

## Run

From the project directory:

```bash
python main.py
```

The training loop writes evaluation rewards to TensorBoard. To view them, start TensorBoard in another terminal:

```bash
tensorboard --logdir runs
```

Then open the local URL printed by TensorBoard. Run directories and event files are generated during training.

## Project files

```text
.
├── frozenlake.py   # Environment settings and tabular Q-learning agent
├── main.py         # Training and evaluation loop
├── README.md
└── LICENSE         # MIT License
```

## Current limitation

The type annotation for `self.values` in `frozenlake.py` is currently malformed (`tt.Dict[ValuesKey. float]`). Python evaluates that annotation when importing the module, so `python main.py` will fail before training starts. Correct it to `tt.Dict[ValuesKey, float]` before running the commands above. There is no dependency lock file yet, so installs use the latest compatible package versions available to pip.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
