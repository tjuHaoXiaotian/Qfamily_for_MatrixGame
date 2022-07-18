# Qfamily for MatrixGame

## Motivation
There have been a lot of research works (e.g., QTRAN and QPLEX) using matrix Games to analyze the performance of their algorithms. An example is shown in the following figure.

![A typical cooperative 2-player Matrix Game](./figure/matrix_game_example.png)

Although the task is simple and the Game only contains a single state, available implementations usually build the code on top of a popular repository: [PyMARL](https://github.com/oxwhirl/pymarl).
However, PyMARL is a relatively heavy codebase, which is naturally designed for complex multiagent tasks (e.g., the Starcraft Multiagent Challenge). 
Using PyMARL to test algorithms' performance on single state matrix Games is inefficient and not necessary. 
Therefore, we provide a very simple implementation of the typical value decomposition methods (e.g., QMIX, QTRAN) for solving single state Matrix Games.

## Algorithms
Currently, the supported algorithms include:


| Algorithm | Progress |
| :------- | :-------------- | 
| VDN | :white_check_mark: |
| QMIX | :white_check_mark: |
| QTRAN | :white_check_mark: |
| QPLEX | :white_check_mark: |

If you want to have a quick review of these algorithms, you could refer to [基于值分解的多智能体强化算法回顾 - 郝晓田的文章 - 知乎](
https://zhuanlan.zhihu.com/p/421909836). For more details, please refer to the original papers.

