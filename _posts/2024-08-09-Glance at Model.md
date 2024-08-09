---
title: "glance at the model"
keywords:
  - Artcteture
  - Weigh assignment
  - Softmax
  - NeuralNET
  - SGD, Momentum, Adam, and other optimizers
  - Data augmentation
  - ResNet and DenseNet architectures
...
<!-- 上来先总结这篇文档目的，要不然读者很难花费attention-->
<!--  -->

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
![Natural and Artificial neurons](/images/realneuron.png)
xx对自然neuron抽象表示为 多个输入经过一个圆 处理得到输出。
stage1:
such a machine—a machine capable of perceiving, recognizing and identifying its surroundings without any human training or control
stage2:
基于stage1的研究，学者showed that a single layer of these devices was unable to learn some simple but critical mathematical functions (such as XOR). In the same book, they also showed that using multiple layers of the devices would allow these limitations to be addressed. 粗心的点在于，只有第一点被广泛知晓，导致没有在继续研究，停滞了20年左右。


# traditional program vensus 
作为程序员的我们日常工作就是编写一段program 通过我们拿到的数据经由这段program拿到结果完成KPI.
![a traditional program](/images/traditionalprogram.png)
## weight assignment
![weight](weighassignment.png)
理解以下两点
- [ ] weight: just variables
- [ ] weight assignment : a particular choice of values for those variables
基于第一个阶段，这里我们将program换成Model（a special kind of program）
这张图片告诉我们weight 和input同时传递给Model处理，其实weights are in a sense another kind of input
## nachanism for maximize the performance
图片刻画了a procedure could be made entirely automatic and...a machine so programmed would “learn” from its experience.
这里我们看到模型的一开始响应的结果并不一定符合我们的要求，进行性能评估后 我们通过改变weight values，也就是进行weight assignment这个过程 ，实现模型整体输入的修改从而改进模型相应的结果，不断的改进最终得到我们想要的结果（性能）
![performance.png](/images/performance.png)
- [ ] performance:人为设定对特定任务的评判标准
### 说点远的
一旦model修炼成功,weight认为是model本身的一部分，就不再变动了。
后续考虑fine-tiune就在说啦。
## train loop


![loop](/images/train_loop.png)
## A simple NN 🗄
muliti-input------>Recurrent NeuralNet----→ Output

When it comes to cross-linking files within your Zettelkasten, there are two general ways of doing so: Either by using an ID, or its filename (without extension). So if you have a file called “zettelkasten.md” you can link to it by writing `[[zettelkasten]]`. Zettlr will try to find a file with that filename and open it.

But what if you change the filename? Then, obviously the link will no longer work! To get around this limitation, you can make use of IDs. IDs are simply strings of digits that you can use to uniquely identify your files. Then you can use them to link to your files. Let’s create one now! Place the cursor behind the colon and press `Cmd/Ctrl+L`:

Now, this file has an ID which you can make use of! Try it out — go back to the tab with the “Welcome to Zettlr!”-guide, and type `[[` somewhere. From the popup autocomplete, choose this file and confirm your selection. Then, `Cmd/Ctrl`-click on that very link to switch back to this file. You’ll notice that Zettlr has started another search, but, more importantly: you can see the search results highlighted! This is useful both for Zettelkasten-crosslinking, but will of course also come in handy during global searches.

## Advanced NN🏷
combination of mulit-input and weight assignment --→ recurrent NeuralNet------>output
### the concpet of Weight assignment
Assumed that each input maybe have a different influence on the output by nn.Such as the first input has greatly effect the output,up to 80%,the second input has a lower effect on the output,just 20%.There,80% and 20% is weight assginment what i say.weight assigns on different input.

But creating links is not the only way to create relationships between notes. You can also use tags for this. Tags work exactly like hashtags on Twitter, so you can #create #hashtags #as #much #as #you #want! `Cmd/Ctrl`-clicking these will also start a search and will highlight all files that contain this tag.

There’s also a tag cloud that you can access by clicking the “tag” icon in the toolbar. It will list all your tags and indicate the number of files using it. You can filter and manage your tags from there. While Zettelkasten-links create “hard” connections between files, tags are some sort of “fuzzy” connection between related content and may suit you better.

## Final Thoughts 💭

We won’t go over methods for how to actually work with a Zettelkasten here, because there are a lot of tutorials out there that will get you started. Here’s a handy list of good tutorials:

- [A first introduction can be found in our docs](https://docs.zettlr.com/en/academic/zkn-method/)
- [On the concept of the Zettelkasten, read our blogpost](https://zettlr.com/post/what-is-a-zettelkasten)
- [The page zettelkasten.de (in English) contains many articles on Zettelkästen](https://zettelkasten.de/)
- [Reddit has a subreddit dedicated solely to the art of Zettelkasten](https://www.reddit.com/r/Zettelkasten)

These will prove excellent starting points for your journey to learn the arcane art of creating a Zettelkasten!

One last thing though: As the way Zettelkästen work is not very standardized, and there exist many right ways of doing it, Zettlr allows you to fully customize every single aspect of the Zettelkasten-methodology. To get started, have a look at [our documentation on how that works](https://docs.zettlr.com/en/reference/settings/#zettelkasten)!

**Ready for more?** Then head over to our guide on [[citing]] with Zettlr!
# what is mode


