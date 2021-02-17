# EWC

A Tensorflow implementation of **Elastic Weight Consolidation** (EWC, in short), introduced in paper titled [**"Overcoming catastrophic forgetting in neural networks"**](https://arxiv.org/pdf/1612.00796.pdf). 

---
## What is EWC?

EWC is an algorithm designed to safeguard against **catastrophic forgetting** in neural networks. Neural Networks also have their capacity for memorizing a large amount of information. In other words, they also possibly suffer from ‘forgetting problem’. Deep learning systems have become more capable over-time, however, traditional training approaches cannot handle incrementally learning new tasks or categories without forgetting previously learned training data. In simple terms, when you have trained a model on Task A and using the same weights for learning a new Task B, then your model forgets learned information about Task A. We call this problem catastrophic forgetting.

<p align="center"><img src="./img/forget.png"/></p></br>

Suppose we have two tasks, A and B, that we would like a neural network to sequentially learn. In the below picture, 𝜃*<sub>A</sub> refers to the configuration of 𝜃 that performs well at task A. But there are a number of configurations in close proximity that will also perform quite well on A; The grey ellipsoid represents the set of these configurations. The optimal solution for B would have a similar error space, represented above by the cream ellipsoid. If the network was subsequently set to learn task B without any interest in remembering task A, the network would shift its parameters in the direction of the blue arrow. However, we want to remember task A. If we naively try to make all parameters somewhat rigid, we follow the green arrow and we perform poorly on both tasks A and B. The superior approach is to make parameters more rigid or less rigid depending on their importance. By doing so, the network shifts its parameters in the direction of the red arrow, and in doing so, finds a configuration that performs well at both tasks A and B. We call this algorithm Elastic Weight Consolidation.

<p align="center"><img src="./img/ewc.png"/></p></br>

---
## Preparation

### Installing prerequisites

To install the prerequisite, simply type in the shell prompt the following:

```
$ pip install -r requirements.txt
```

You can use TensorFlow libraries with GPU support (See the guide at [TensorFlow tutorials](https://www.tensorflow.org/guide/gpu?hl=en)) to accelerate your code with CUDA back-end.

### Dataset



---
## Files

* `data.py`: Data provider (MNIST, QMNIST, MNIST-M). 
* `model.py`: Implementations of a categorical classifier (general vs EWC).
* `utils.py`: A bunch of utility functions for evaluation.

---
## Usage

* Load the MNIST dataset (1st task).

```
>>> import data
>>> mnist = data.MNISTdata().getTask()
>>> tasks = [mnist]
```
</br>

* Build a classifier and feed the first dataset to it.

```
>>> import model
>>> M = model.Model()
>>> M.train(tasks)
```

* .

```
>>> M_ewc1 = model.Model()
>>> M_ewc1.transfer_model(model=M, prev_tasks=tasks)
```

* .

```
>>> qmnist = data.QMNISTdata().getTask()
>>> tasks.append(qmnist)
>>> M.train(tasks)
>>> M_ewc1.train(tasks)
```

* .

```
>>> M_ewc2 = model.Model()
>>> M_ewc2.transfer_model(model=M_ewc1, prev_tasks=tasks)
```

* .

```
>>> mnistm = data.MNIST_Mdata().getTask()
>>> tasks.append(mnistm)
>>> M.train(tasks)
>>> M_ewc1.train(tasks)
>>> M_ewc2.train(tasks)
```

* .

```
>>> import utils
>>> utils.visualize_multi_acc([M, M_ewc1, M_ewc2])
```

![ex_screenshot](./img/results.png)
