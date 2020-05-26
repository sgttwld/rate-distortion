# Rate-distortion: _Blahut-Arimoto vs. gradient descent vs. mapping approach_

This repository contains implementations (using tensorflow v2) of 4 different algorithms to determine the optimal Shannon rate-distortion trade-off for a given set of samples from a continuous source,

* [RD_BA.py](https://github.com/sgttwld/rate-distortion/blob/master/RD_MA.py): Blahut-Arimoto algorithm.
* [RD_GD.py](https://github.com/sgttwld/rate-distortion/blob/master/RD_MA.py): Direct optimization using gradient descent.
* [RD_MA.py](https://github.com/sgttwld/rate-distortion/blob/master/RD_MA.py): Two different versions (_direct_ and _iterative_ optimization) of the mapping approach by Rose.

### View the [notebook](https://nbviewer.jupyter.org/github/sgttwld/rate-distortion/blob/master/rate-distortion_nb.ipynb) (using nbviewer) for an explanation of each algorithm. 

