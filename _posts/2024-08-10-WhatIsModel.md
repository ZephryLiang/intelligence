# glance at the model

# Preface
如果你是一个从0开始学习deep learning的初学者（新手村小白）那么这篇文章一定可以带你从最简单的角度切入the  world of deep learing
## 推荐课程
[数学课程](https://www.khanacademy.org/)[](https://www.khanacademy.org/)
# Preface
如果你是一个从0开始学习deep learning的初学者（新手村小白）那么这篇文章一定可以带你从最简单的角度切入the  world of deep learing
## 推荐课程
khan Academy  https://www.khanacademy.org/
## 终极目标？
训练出state-of-the-art的模型在以下几个领域
- [x] Computer Vison
- [x] Natural Language processing
- [x] Tabular data anaylze
- [x] Collaborative filter (eg某些推荐机制下我们需要分析多用户的数据进行collaborative工作)
作为初学者进行多领域的学习是可取的，模型的generalization 能力不也是这样么（后续有机会在介绍model generalization ability）

# Neural Net🗂
# brief history
stage0:
不知道是否您是否在高中阶段学习生物过程中接触到了神经元
![Natural and Artificial neurons](images/chapter1/realneuron.png)
xx对自然neuron抽象表示为 多个输入经过一个圆 处理得到输出。
stage1:
such a machine—a machine capable of perceiving, recognizing and identifying its surroundings without any human training or control
stage2:
基于stage1的研究，学者showed that a single layer of these devices was unable to learn some simple but critical mathematical functions (such as XOR). In the same book, they also showed that using multiple layers of the devices would allow these limitations to be addressed. 粗心的点在于，只有第一点被广泛知晓，导致没有在继续研究，停滞了20年左右。


# traditional program vensus 
作为程序员的我们日常工作就是编写一段program 通过我们拿到的数据经由这段program拿到结果完成KPI.
![a traditional program](images/chapter1/traditionalprogram.png)
![a traditional program](../images/chapter1/traditionalprogram.png)
## weight assignment
![weight](images/chapter1/weighassignment.png)
理解以下两点
- [ ] weight: just variables
- [ ] weight assignment : a particular choice of values for those variables
基于第一个阶段，这里我们将program换成Model（a special kind of program）
这张图片告诉我们weight 和input同时传递给Model处理，其实weights are in a sense another kind of input
## nachanism for maximize the performance
图片刻画了a procedure could be made entirely automatic and...a machine so programmed would “learn” from its experience.
这里我们看到模型的一开始响应的结果并不一定符合我们的要求，进行性能评估后 我们通过改变weight values，也就是进行weight assignment这个过程 ，实现模型整体输入的修改从而改进模型相应的结果，不断的改进最终得到我们想要的结果（性能）
![performance.png](images/chapter1/performance.png)
- [ ] performance:人为设定对特定任务的评判标准
### 说点远的
一旦model修炼成功,weight认为是model本身的一部分，就不再变动了。
后续考虑fine-tiune就在说啦。
## train loop


![loop](images/chapter1/train_loop.png)
## 终极目标？
训练出state-of-the-art的模型在以下几个领域
- [x] Computer Vison
- [x] Natural Language processing
- [x] Tabular data anaylze
- [x] Collaborative filter (eg某些推荐机制下我们需要分析多用户的数据进行collaborative工作)
作为初学者进行多领域的学习是可取的，模型的generalization 能力不也是这样么（后续有机会在介绍model generalization ability）