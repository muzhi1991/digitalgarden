---
{"dg-publish":true,"permalink":"/003-Publish/History/流式超越批量-Streaming 102翻译/","title":"流式超越批量：Streaming 102翻译","tags":["技术","流式计算","大数据分析","Spark","翻译"],"created":"2024-12-11T18:09:35.717+08:00","updated":"2024-12-12T11:19:54.646+08:00"}
---

---

## 译者注

在流式系统方面，网络上流传了两篇精彩的文章，[Streaming 101](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-101) 与 [Streaming102](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102)。这两篇文章的作者是Google流式系统的负责人[Tyler Akidau](https://www.oreilly.com/people/09f01-tyler-akidau)，他是MillWheel与DataFlow的开发者，在流式系统方面十分权威。恰巧这两篇文章非常详细并且适合初学者，可以帮助我们理清流式系统的各种概念与面临的挑战，是十分难得的佳作。其中Streaming101已经有很多翻译的版本，质量参差不齐。官方的翻译版本参考[这里](https://www.oreilly.com.cn/ideas/?p=18)，然而不幸的是它依然晦涩难懂。这里并不打算翻译101这篇文章，读者可以参考原文和译文进行学习，它主要介绍了流式系统的一些基本概念和困境，同时**提出了一个核心观点：设计优良的流式系统完全可以代替批量系统**，并从原理上分析了如何实现这一目标。在流式102这篇文章中，作者以实际的流式系统（DataFlow）的设计为例，列举了多个场景，具体讲解了如何设计流式系统，解决这些问题。这里将翻译流式102这篇文章，我会参考原文采用意译的方式，力求更好的帮助读者理解。

**在阅读流式102之前，请务必通篇阅读流式101**，理解核心概念，这里的[思维导图](/files/流式101--超越批量系统.mindnode)仅供参考。Let's Go!

## 简介

欢迎回来！如果您错过了之前的[《The world beyond batch: Streaming 101》](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-101)，强烈推荐先阅读这篇文章，它是理解下文的基础。请注意：我们假设你已经熟悉了那些技术术语和概念。此外，本文有一些动画Gif，如果你用打印出来看就会丢失这些信息。

让我们进入正题。简要回顾一下，上一次我们主要讨论了3个主题：

* 技术术语：精确定义了一些已经已经被大家说烂了的词，比如『流式』（Streaming）
* 批量VS流式：比较了批量系统与流式系统的能力。并假定只要做好了**准确性(Correctness)**和**时序推理工具(Tools For Reasoning About Time)**这两件事，流式系统就可以超过批量系统。
* 数据处理的模式/套路：分别介绍了用流式系统(Streaming)和批量系统(Batch)来处理有界(Bounded)和无界(Unbounded)数据集的常用方法。

这篇文章中，我们会接着深入探讨数据处理模式，但是会更加关注细节和具体的应用场景。本文主要分为两个章节：

* 流式101的精简版：简要回顾Streaming 101中引入的概念，并引入一个Demo来突出一下重点。
* 流式102：流式101的姊妹篇，详细分析处理无界数据集的一些其他相关概念。同样会用一些Demo来解释概念。

当此文完结，我们就会掌握构建一个健壮的处理无序数据的系统所需的核心原理与概念。有一些时序推理工具可以让你真正地超越批量系统！😏

为了给你提供一些直观感受，我会使用DataFlow SDK的代码片段（例如：[Google DataFlow API]( https://cloud.google.com/dataflow/)）。同时，我还用了一些动画来可视化一些概念。至于为何使用DataFlow而不是Spark Streaming或者Storm这类系统，其一是我更熟悉DataFlow，其二是其他系统并没有这种表现能力来展示我所涉及的例子（潜台词：功能不全面）。好消息是其他系统也在朝着这个方向努力，一个更大的好消息是Google现在已经向Apache基金会提出[Propsal](http://googlecloudplatform.blogspot.com/2016/01/Dataflow-and-open-source-proposal-to-join-the-Apache-Incubator.html)来建立一个DataFlow的孵化器项目（与Artisans, Cloudera, Talend等公司合作）。它的目标旨在以由[DataFlow模型](http://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf)提供的强大的无序数据处理语意为基础来建立一个开源社区和生态。这会在2016年渐渐变得有趣，我跑题了。。。

抱歉，本文中没有上次我承诺的各个流式框架比较那部分。我错误估计了本文包含的内容以及时间，我不能再拖稿了。作为补偿，最后会附上我在2015年新加坡Hadoop大会上发表演讲[《The evolution of massive-scale data processing》](http://conferences.oreilly.com/strata/big-data-conference-sg-2015/public/schedule/detail/44947)（在2016的六月会更新版本），它有一些材料就是我想讲的。同时，奉上一个[精美的幻灯片](http://goo.gl/5k0xaL)，Enjoy。

现在，开始讨论流式！

## 回顾&&大纲

在流式101中，我首先澄清了一些技术术语：

* **bounded数据 VS unbounded数据**。有界数据源通常是有限的大小，经常指的是『批量』数据。无界数据源通常有无限的大小，通常指的是『流式』数据。我会尽量避免使用『批量』或者『流式』这些词来描述数据源，他们的含义会引起无解和导致不必要的限制。
* 然后，我进一步定义了**流式引擎与批量引擎的不同点**。批量引擎是仅仅为处理有界数据而设计的，而流式引擎则为无界数据而设计。我只会用『流式』和『批量』这些词来表示执行引擎。

紧接着我们又涉及了与无界数据处理相关的两个重要概念：

* 首先我们要严格**区分事件时间(Event Time，事件真实发生的时间)和处理时间(Processing Time，事件被我们观察到的时间)**。在流式101一文中我们认为这是重要的基础之一：如果您关心正确性和事件上下文，则必须使用事件时间，而不是处理时间。
* 我们又介绍了**窗口化(Windowing，沿着时间边界划分数据集)**的概念。这是一种解决无限数据源永远不会结束这一问题的常见方法。窗口化策略的简单的方法有**固定窗口**和**滑动窗口**，但还有一些更复杂的窗口类型，如**会话**（该窗口由数据本身的特征定义，例如，使用每个用户不活动的间隙来切分）也有广泛的应用。

除了101中介绍的两个重要概念，本文我们还将引入另外三个概念

* **Watermarks**：Watermark(水印)是用来表示与事件时间相关联的输入完整性的概念。对于事件时间为X的Watermark是指：已经观察到事件时间小于X的所有输入数据。因此，当观测对象是没有尽头的无界数据源时，Watermark来测量数据进度。
* **Triggers**：触发器是一直外部信号触发机制，用于表示什么样的信号会真正地触发窗口中的数据被计算。触发器为数据计算的触发提供灵活性。触发器还使我们可以在有需要时多次运算数据（例如数据在某时间更新了，要重算）。这为我们可以随着时间变化不断优化运算结果打开了方便之门：我们可以在数据到达时先提供一个推测(speculative)版本的结果，然后，随时间推移不断处理上游数据的变换（生成了修订版本）或者处理那种在Watermark之后才到达的数据（例如，手机的场景：某人在断网时记录各种动作及其事件时间，然后在重新联网后，上传这些事件进行处理。）
* **Accumulation**：累加/更新模式指定在同一窗口中观察到的多个运算结果之间的关系。这些结果之间可能完全不相关，例如与时间先后无关的结果，直接覆盖以前的运算结果即可。不同的累积模式具有不同的语义和成本，视各种场景而定。

最后，为了更好地理解所有这些概念之间的关系，我们将重新回顾全文，并通过回答下面四个问题来温故知新。我提出的这几个问题都至关重要：

* **What results are calculated?**：做什么计算？（计算逻辑是什么，如何表示？）这个问题的答案是：pipeline中的各种transform操作。这包括求和，建立直方图，训练机器学习模型等等。这也是经典批处理回答的问题。
* **Where in event time are results calculated?**：计算什么时间范围的数据？（这里的时间指的是Event-Time）这个问题的答案是：在pipeline中用EventTime来窗口化数据。这包括流式101中提到的窗口化方法（固定窗口，滑动窗口，会话窗口），以及似乎没有窗口概念的场景（例如，Streaming 101中描述的时间不可知处理;经典批处理也通常属于此类别。此外，还有其他更为复杂的窗口类型，如时间有限的拍卖（time-limited auctions）。请注意，如果在进入系统的时间标记为事件时间，上面这些清空也可以指的是Processing Time。
* **When in processing time are results materialized?** ：何时将计算结果输出？这个问题的答案是：使用watermark和trigger配合触发计算。这有很多的组合，但最常用的方法（最佳实践）是用Watermark表示某个窗口的输入完成，同时，配合允许Eearly Data（用于在窗口完成之前发出的推测结果）和Later Data（水印只是完整性的估计，当水印指示窗口的输入完成之后，更多的输入数据还是可能会到达）的触发器。
* **How do refinements of results relate?**：后续数据的处理结果如何影响之前的处理结果？这个问题通过Accumulation来解决：丢弃（结果之间是独立且不同的），累积（后来的结果建立在先前的结果上）或累积并撤回（其中累积值加上先前触发的值的撤回）

下文会详细讨论这些问题：我会用上一些配色以清晰地表示我们讨论的问题属于上面的哪一种 **What**/**Where**/**When**/**How**。

## 流式101的精简版

首先，我们来回顾一下Streaming 101中提出的一些概念，但这次我将用一些详细的例子使得这些概念更具体。

### What: transformations

经典批处理应用的转换操作已经回答了第一个问题：“计算逻辑是什么？”。尽管许多人可能已经熟悉经典的批处理，但我们将从那里开始，因为它是我们的基础，我将基于他添加一些其他概念。

这一节我们就看一个例子：在一个简单的数据集（如10个值）上按key聚合求SUM。你可以把它想象成某个手游里面通过把个人的分数求和来计算团队总分，或者其他计费类应用或者Monitor的应用等。

对于后面的每一个例子我都会用Dataflow的伪代码来定义具体的Pipeline。这将是伪代码，在某种意义上，我有时会稍稍改变规则，使示例更清晰，去除不必要细节（如使用的具体I / O源）或简化命名（Java中的当前触发器名称是否繁琐，为了清晰起见，我将使用更简单的名称）。除了这些小事情之外（大部分我在后文注释中明确列举），它基本上是真实可用的Dataflow SDK代码。稍后对那些对他们可以编译和运行自己的类似示例感兴趣的人，我还将提供一个链接到实际的代码。

如果你熟悉Spark Streaming或者Flink，那么比较容易理解DataFlow的代码，下面简单介绍DataFlow的两个基本原语：

* **`PCollections`**：表示可以执行并行（这里的P表示的含义）的转换（Transform）操作的数据集（可能非常大）
* **`PTransforms`**：应用到PCollection上，来执行的转换操作，生成新的PCollection。PTransforms可以是对元素一个一个操作，也可以是聚集（agg）操作，或者可以与其他的PCollection相互组合。

![20241212105125385](http://fodi.limuzhi.us.kg/images/IMG-ffe5152f18515401.webp)

如果有疑问或者想查看DataFlow的文档，看[这里](https://cloud.google.com/dataflow/model/programming-model)。

为了讲解例子的方便，我们这里使用了一个名叫**`input`**的`PCollection<KV<String, Integer>>`作为输入（input是由String/Integer作为键/值对组成的，String是球队名，Interger是每人的分数）。在实际的Pipeline中，input一般从I/O源读入原始数据，然后解析日志数据成Key/Value对，最终转换成`PCollection<KV<String, Integer>>`。我会在第一个示例中写出所有代码，但是在接下来的其他示例中隐去I/O相关的部分。

因此，一个例子：要求先从数据源中解析出team/score键值对，然后对每个team求和，算出球队总分。代码如下：

```java
// Listing 1. Summation pipeline
PCollection<String> raw = IO.read(...);
PCollection<KV<String, Integer>> input = raw.apply(ParDo.of(new ParseFn());
PCollection<KV<String, Integer>> scores = input
  .apply(Sum.integersPerKey());
```

对于下面的所有例子，我们都会先分析代码，然后用动画展示一个数据集的运行过程。具体点说，就是含有一个key和10条数据的pipeline的执行过程。在实际运行时，会有多台机器并行执行相同的操作。这里仅仅是为了简单清晰地讲解例子。

所有动画都会在两个维度上绘制输入和输出：事件时间（X轴）和处理时间（Y轴）。因此，从pipeline的视角来看，白色的粗线条从底部向上移动代表了真实时间的移动。输入用圆圈表示，圆圈内的数字表示该记录的值。他们一开始是灰色的，pipeline实际观测到他们后会改变颜色。

随着pipeline观测到这些值，会逐渐在状态（State）中累加他们，并最终实际输出结果。状态的变换和输出都由矩形表示，聚合值在矩形上面，被矩形覆盖的区域表示该部分的事件时间/处理时间已经累积计算到了结果中。在批处理引擎中执行Listing 1中的pipeline代码，运行过程如下（请注意，您需要点击/点击下面的图像才能启动动画，然后再循环直到再次点击/再点击）：

<iframe height="400" width="600" src=" https://fast.wistia.net/embed/iframe/24noytvllc?videoFoam=true&wvideo=24noytvllc"></iframe>

[图2：传统的批量处理过程](https://fast.wistia.net/embed/iframe/24noytvllc?videoFoam=true&wvideo=24noytvllc)

由于这是一个批量的pipeline，因此，只有接收到所有的input值，系统才会累加状态（由图中顶部的绿色虚线表示），最终产生了唯一的输出：51。而且，这个例子由于没有使用任何Window，我们计算了所有Event-Time内的值的和。所以，图中用于表示状态和输出的矩形覆盖了整个X轴。但是，如果我们要处理一个无界数据源，那么传统的批量处理就不行了；我们不能等待输入数据结束，因为它实际上永远不会结束。因此，我们需要引入一个在Streaming101中提及的概念：窗口。在回答第二个问题，“计算什么时间范围的数据？”之前，我们先简要回顾一下窗口。

### Where: windowing

如上次讨论的那样，窗口化是沿着时间边界分割数据源的过程。常见的窗口划分策略包括固定窗口，滑动窗口和会话窗口。

![20241212105125455](http://fodi.limuzhi.us.kg/images/IMG-8a6ccca5ab6c41ca.webp)

看一个实际的例子：把上面的求和的 pipeline 划分为 2 分钟的固定时间窗口。使用 DataflowSDK，添加一个 `Window.into` transform 操作即可：

```java
// Listing 2. Windowed summation code.
PCollection<KV<String, Integer>> scores = input
  .apply(Window.into(FixedWindows.of(Duration.standardMinutes(2)))) // 这一行
  .apply(Sum.integersPerKey());
```

回想一下，由于语义上批处理只是流式的一个子集，所有 Dataflow 提供了一种统一的批处理和流式模型。因此，我们首先在批量引擎上执行这个 pipeline，这种机制看起来更直观。当我们切换为流式引擎时，可以直接与这个进行比较。

<iframe height="400" width="600" src=" https://fast.wistia.net/embed/iframe/v12dlvvgfh?videoFoam=true&wvideo=v12dlvvgfh"></iframe>

[图4：在批量引擎上对窗口进行求和](https://fast.wistia.net/embed/iframe/v12dlvvgfh?videoFoam=true&wvideo=v12dlvvgfh)

如前所述，直到输入数据被完全读入，状态才会累加，之后产生输出。但是，在这种情况下，我们得到的不是一个输出值，而是四个输出值：每个输出对应着 2 分钟的事件时间窗口。

至此，我们回顾一下 Streaming 101 中介绍的两个主要概念：事件时间（Event-Time）与处理时间（Processing-Time）的关系以及窗口化（Windowing）。如果我们想深入学习，我们需要增加本节开头提到的新概念：水印（Watermark），触发器（Trigger）和累积（Accumulation）。因此开始 Streaming 102。

## 流式 102

我们刚刚看到了如何在批量引擎上运行窗口化的 pipeline。但是，我们更希望系统有更低的延迟并且可以原生支持处理无界数据源。切换到流式引擎是正确的方向！但问题是，在批量引擎上我们清楚地知道什么时间点数据完整了（比如：当有限数据源的数据全都读入的时候），但是对于无界的数据源，我们缺乏一种有效的方式来判断数据**完整性（Completeness）**。因此，引入了 Watermark

### When: watermarks

Watermark 可以给出"何时将计算结果输出？"这个问题的一半答案：Watermark 是在 Event-Time 域上的时间概念，用来刻画输入完整性。（这句话说明，watermark 首先是表示的是一个 Event-Time 时间）。换句话说，它们是系统以 Event-Time 为尺度来衡量事件流中 Record 处理进度/完整性的方式（不管是无界数据还是有界数据都适用，显然在无界的情况下更有用）。

回想一下 Streaming 101 中的这个图，这里稍作修改，其中描述了事件时间和处理时间之间的偏差（skew），在真实的分布式系统中，这个偏差会随时间不断变化。

![20241212105125671](http://fodi.limuzhi.us.kg/images/IMG-d2311396b61f7b27.webp)

上面这个红色曲线就是真实的 Watermark，随着 Processing-Time 的推移，他描述了 Event-Time 纬度的完整性的过程。你可以把 Watermark 看成是 F (P) -> E 的函数：输入是 Processing-Time，输出是 Event-Time。（确切的说，函数的输入是在 pipeline 中被观测到的 Watermark 这一点的所有上游的当前状态：输入源，缓冲数据，正在处理的数据等；但在概念上，将其视为从 Processing-Time 到 Event-Time 的映射更为简单。）在 Event-Time 上的这一点 E 表示：系统相信在 E 之前的所有数据都被观测到了。换句话说，系统『确信』不会再有 Event-Time<E 的数据出现了。根据这种『确信』是不是严格保证或者仅仅是猜想，我们把 Watermark 分为两种类型：完美 Watermark 与启发式 Watermark。

* **完美 Watermark**：如果我们对所有输入数据十分了解，就有可能构建一个完美的 Watermark。这种情况下，输入源不存在数据迟来的问题，所有数据只会提前或者准时到达。
* **启发式 Watermark**：对于大部分的分布式输入源，期望对输入数据完全掌控是不切实际。这种情况下一般使用启发式的 Watermark。启发式 Watermark 会使用一切可用的信息（包括分区，分区内排序，文件增长速度等）来尽量准确地推断输入的进度。在许多情况下，这种 Watermark 也可以预测了非常准确。即使如此，启发式 Watermark 的意味着它的预测有时可能是错误的，这将导致迟到的数据。我们将在下面的 Trigger 部分中了解如何处理迟来的数据。

Watermark 是一个有意思而且复杂的话题，超出我的讨论范围，期望未来有时间写个帖子讨论它。现在，为了更好地了解 Watermark 的作用和缺陷，我们使用 Watermark 的流式引擎，来看看 Listing 2 中的 pipeline 代码何时将计算结果输出。左边的例子使用完美的 Watermark，右边的使用了启发式 Watermark。

<iframe width="560" height="315" src=" https://www.youtube.com/embed/JKpe_wUyUWg?si=2zXBFHQFa8C73O1Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
[图6：在流式引擎上分别使用完美watermark（左）与启发式watermark（右）进行分割窗口求和的运算](https://www.youtube.com/watch?v=JKpe_wUyUWg)

这两种情况都是 watermark 到达窗口终点时结果被输出（注：看图中 watermark 的先与12:02 12:04 12:06 12:08的交点）。主要的不同在于：右侧启发式算法没有计算『9』这个值，这对 watermark 曲线的形态影响巨大。这个例子突出了 watermark 的两个缺点（不仅 watermark，任何一种完整性概念都有类似的问题），具体来说：

* **太慢了**：当 watermark 因为知道有数据还没到来被正当地推迟了（比如：网速下降导致输入变慢），如果我们最终输出结果仅仅依赖于 watermark 曲线的提升，那么这种延迟会直接影响了系统最终输出的时间。这在左图中体现的最明显：即使后面窗口的数据都已经到来了，第一个窗口迟到的数据『9』卡住了后续所有窗口的 watermark 线。第二个窗口[12：02，12：04) 的问题尤其突出，从窗口中的第一个值到达，直到输出结果，需要将近七分钟。在右图中的启发式 watermark 中，这个问题不是很严重（延迟 5 分钟分钟），但是**不意味着启发式算法没有这个问题**；这仅仅是因为这个例子中我选择启发式 watermark 比较特殊。这里我们有一个重要的结论：虽然 watermark 提供了完整性的概念，但是他对系统延迟影响很大。假设你有一个显示数据指标的面板，你一定不能忍受要等待一个小时或者一天才能看到当前窗口的数据出现—这是批量系统才会有的痛点嘛。我们更希望可以随着时间推移看到指标逐渐变化，最终产生完整地结果。
* **太快了**：启发式的watermark 算法不能正确地标识数据。某些 Event-Time 小于 watermark 的数据会迟到地到达，产生了 late data。这就是右图中发生的问题：在第一个窗口中，watermark 线在所有数据到来之前就提前穿过了窗口终点，最终导致了错误的计算结果『5』（正确应该是 14）。这是启发式算法导致的严重问题。『启发式』天然地隐含了这个意思—我有时会出错。所以，如果我们关心结果的正确性，仅仅使用启发式 watermark 是不够。

在 Streaming 101 中，我强调过：仅仅想使用完整性（Completeness）这个概念来健壮地处理无序无界的数据流还远远不够。Watermark 太慢或太快这两大缺点就是我这个观点的依据。**仅仅依靠完整性的系统是不能同时获得低延迟*和*正确性的，解决这些问题的关键是引入触发器（Trigger）。**

### When: The wonderful thing about triggers, is triggers are wonderful things!（触发器是个好东西）

"何时将计算结果输出？"这个问题的另一半答案就是触发器。触发器声明了一个窗口的计算结果什么 Processing-Time 时间被输出？（但是，触发器自身做出决定可能依据的是其他 Event-Time 时间域发生了什么，比如 watermark 线的进度）。窗口内的每次特定输出被称为窗口的窗格（pane）。

触发 Trigger 的信号包括下面这些：

* **Watermark 的进度（如：Event-Time 的值）**：其实这就是图 6 隐含的触发器，当 watermark 线到达窗口终点时触发输出。另外一个例子是：当触发器存在超过了一定时间后进行**触发器的垃圾回收**，后面我们会看到他的应用。
* **Processing-Time 的进度**：用来提供定期更新数据，因为 Processing-Time（不像 Event-Time）总是大致均匀地移动，而不会出现延迟。
* **到达元素的数量**：窗口中观察到一些有限数量的元素之后进行触发
* **特殊的标记**：在 Record 的一些记录或特征值（例如，EOF 元素或刷新事件）指示应该生成输出。

除了使用某个特定信号的简单触发器之外，还有**组合触发器**，允许创建更复杂的触发逻辑。组合触发器包括：

- **重复触发（Repetitions）**：特别适用于 Processing-Time 触发器以提供定时更新操作。
- **联合触发（Conjunctions）** ：(逻辑 AND)，只有**所有**子 Trigger 触发才会触发（例如，在 watermark 通过窗口终点并且我们观察到含有 EOF 符号的 Record）
- **各自触发（Disjunctions）**： (逻辑 OR)，只要有一个子触发器触发，就会触发 (例如，只要在 watermark 通过窗口终点或者我们观察到含有 EOF 符号的 Record 就触发).
- **按顺序触发（Sequences）**：以预定义的顺序触发子触发器。（后一个子触发器必须等待前一个触发器触发）

为了让 Trigger 这个概念更具体，让我们明确地表示出图 6 中（也是 Listing 2 的代码中）隐含的触发器：

```java
// Listing 3. Explicit default trigger.
PCollection<KV<String, Integer>> scores = input
  .apply(Window.into(FixedWindows.of(Duration.standardMinutes(2)))
               .triggering(AtWatermark())) // 这一行
  .apply(Sum.integersPerKey());
```

考虑到这一点，基于对触发器功能的基本了解，我们可以思考来解决 watermark 太慢或太快的问题。对于这两个问题，我们基本上希望对于一个窗口可以定期更新输出值（即除了水印线到达窗口终点之外的更新）。因此，我们需要一些**重复触发（Repetitions）**器。那么这个问题就变成了：重复触发的条件是什么？

对于**输出太慢的情况**（即希望提供早期的推测结果的情况），我们可以假设给定窗口的早期阶段，流入数据量是稳定且不完整的。因此按照 Processing-Time 周期性触发（例如 1 分钟一次）是明知。因为触发的次数与数据量无关，最坏的情况就是获得稳定不变的触发数据流。

对于**输出太快的情况**（即对于启发式 watermark 可以处理迟到的数据），我们可以假定 watermark 是基于比较准确的启发式算法（这个加上还是比较靠谱的）。在这种前提下，我们不会经常看到 late data 到达，但是如果出现了这种情况，我们必须快速处理更新输出的结果（例如，看到 late data，立刻更新）。因为我们假设这种情况很少见，所有他不会使得系统过载。

注意，这里只是举了一个例子，你可以根据自己的情况自由选择触发的条件（比如只在上面某一种情况下触发，或者都不触发）

最后，我们需要编排好各种触发器的时间：比 watermark 早触发，在 watermark 达到时触发还是比 watermark 晚触发？我们可以通过一个 `Sequence` 触发器和一个特殊的 `OrFinally` 触发器来实现，`OrFinally` 触发器可以安装一个子 Trigger，当子 Trigger 触发时终止父 Trigger。

```java
// Listing 4. Manually specified early and late firings.
PCollection<KV<String, Integer>> scores = input
  .apply(Window.into(FixedWindows.of(Duration.standardMinutes(2)))
               .triggering(Sequence( // 这两行
                 Repeat(AtPeriod(Duration.standardMinutes(1)).OrFinally(AtWatermark()), 
                 Repeat(AtCount(1))))
  .apply(Sum.integersPerKey());
```

但是，这样写起来有些麻烦。由于这种 ` repeated-early | on-time | repeated-late ` 的模式十分常见，我们在 Dataflow API 中提供了特定的简写方式（语意上等价的）。

```java
// Listing 5. Early and late firings via the early/late API.
PCollection<KV<String, Integer>> scores = input
  .apply(Window.into(FixedWindows.of(Duration.standardMinutes(2)))
               .triggering(
                 AtWatermark()
                   .withEarlyFirings(AtPeriod(Duration.standardMinutes(1))) // 这两行
                   .withLateFirings(AtCount(1))))
  .apply(Sum.integersPerKey());
```

在流式引擎上执行 Listing 4/5 的代码（使用完美 watermark 和启发式 watermark）产生的结果如下：
<iframe width="800" height="450" src=" https://www.youtube.com/embed/0YbPMLb4E30" title="Figure 07   streaming speculative late joint" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[图7.在安装了early/late触发器的流式引擎上进行窗口求和运算 ](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102?wvideo=li3chq4k3t)

这个版本比图 6 的版本有两处明显的提升：

* 对于第二个窗口[12: **02，12:04** ]中的“ **watermark 导致输出太慢** ”的情况：我们提供了在 early 阶段每分钟一次的定期更新。在完美 watermark 的案例中，差异最为明显，其中首次输出时间从近七分钟降至三分半；在启发式案例中也有明显改善。这两个版本现在随时间移动会提供稳定的不断改进的结果（窗格 pane 提供了 7，14，22 这些值的输出），同时我们做到了在输入数据完整和最终结果输出之间的低延迟。
* 对于第一个窗口[12:00，12:02]中的“ **启发式 watermark 太快输出** ”的情况：当『9』这个值晚点到达时，我们立即更新了输出结果，输出了正确的窗格值『14』。

一个意外的惊喜是：新加入的触发器机制让两种类型的 watermark 算法产生了统一的输出。对比图 6 的两个算法版本还截然不同，在这里的他们却十分相似。

这两个版本还有一个很大的区别：**他们窗口的生命周期不同**。完美 watermark 的案例中，我们知道 watermark 通过窗口终点后，系统不会再看到该窗口的任何数据了，因此我们可以关闭当前窗口的所有状态。在启发式 watermark 案例中，我们仍然需要让窗口等待一段时间来处理 late data。但是问题是：**到目前为止，我们没有办法知道需要保留窗口多长时间**。这里我们需要引入『允许迟到时间』这个概念（也叫 horizon）。

### **When: allowed lateness （垃圾回收，何时关闭 Window）**

在进入最后一个问题"后续数据的处理结果如何影响之前的处理结果？"之前，我们要讨论一下处理长期无序数据数据流的流式系统必备的一个功能：垃圾回收。图 7 的启发式算法 watermark 的例子中，每个窗口的状态在该示例的整个生命周期内都会保持。为了处理 late data，这么干是必要的。但是，在实际环境中，当处理无限数据源时，无限期地保持窗口状态（包括元数据）是不切实际的，我们最终会耗尽磁盘空间。

因此，任何实际的无序处理系统都需要提供一些限制窗口生命周期的方法。一个简单的办法是在系统内定义一个允许数据迟到的视界（horizon，理解成时间范围） — 例如：对*记录数据*可以影响处理流程的时间（相对于 watermark）进行限制；任何在这个时间点后到达的数据都会被简单地抛弃。一旦你划定了允许数据可能有多晚到达，你就准确地确定了窗口状态需要保持的最长时间，这段时间就是 watermark 线到达窗口终点线之后再继续等待的时间。此外还给予系统尽快丢弃超过 horizon 的数据的自由，这意味着不要在我们不关心的数据上浪费任何资源。

> 译者注：什么时间关闭窗口（垃圾回收）？
> Watermark 超过窗口结束的延迟时间（EventTime_diff）。（我们定义）即传入当前 ProcessTime，算出 EventTime 2，我们指定一个 EventTime_diff，对于 EventTime 1 的窗口，如果 EventTime 2-EventTime 1>=EventTime_diff, 就可以关闭之前的窗口了。
>
> 为什么用 watermark（Event-Time）来做呢？而不是 Processing-Time（比如在 watermark 到了之后等待 n 秒钟的 Processing-Time）？
> 参考文章给出的注释：因为可能有其他原因导致系统崩溃延迟等等，使得 processTime 就这么过去了，窗口过早地关闭！

由于『允许迟到时间』和『watermark』之间的相互作用有点微妙，所以值得再举一个例子。我们来看一下 Lising 5 /图 7 中的启发式 watermark 的 pipeline 的例子，我们添加一分钟的允许迟到时间（视界，horizon）（请注意，这个特定的 horizon 被选择是因为适合图标展示；在实际工程中，更大的 horizon 可能会更好一些）：

```java
// Listing 6. Early and late firings with allowed lateness.
PCollection<KV<String, Integer>> scores = input
  .apply(Window.into(FixedWindows.of(Duration.standardMinutes(2)))
               .triggering(
                 AtWatermark()
                   .withEarlyFirings(AtPeriod(Duration.standardMinutes(1)))
                   .withLateFirings(AtCount(1)))
               .withAllowedLateness(Duration.standardMinutes(1))) // 这一行
  .apply(Sum.integersPerKey());

```

上面这个 pipeline 的执行如下面的图 8 所示，我给图标添加了下面这些特性来强调『允许延迟时间』的影响：

* 粗白线的位置指示当前 Processing-Time，又给所有的窗口添加了有注释的虚白线来表示允许延迟的视界时间（注意延迟用 Event-Time 定义的）
* 一旦一个窗口内的时间通过 watermark 后再经过 horizon，窗口就会关闭，意味着窗口的状态就好被抛弃。我会留下一个虚线框表示当窗口关闭时，它覆盖的时间范围（两个时间域都有，右侧多出的虚线小尾巴与 watermark 相交）。
* 对于这个图，我额外地为第一个窗口添加了一个迟到值『6』。『6』是迟到的，但仍然在允许延迟时间（horizon）范围内，所以它被并入更新的值为 11 的结果。然而，迟到的『9』超过了延迟的地平线，所以它被简单地抛弃了。

<iframe width="800" height="450" src=" https://www.youtube.com/embed/33i-yP-5Vxw" title="Figure 08   streaming speculative late allowed lateness 1 min" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[图8. **运行在流式引擎上，具有早期和晚期启动并且允许1分钟延迟的窗口求和**](https://fast.wistia.net/embed/iframe/muwcqnrmxf?videoFoam=true&wvideo=muwcqnrmxf)

关于『允许延迟时间』（horizon）的最后两点说明：

* 如果使用完美的 watermark，就不需要处理 late data，这只允许延迟时间为 0 秒就行，这就是图 7 中所示的效果
* 某些特点的任务：如一些根据某些 key 计算 agg 的需求（如统计不同浏览器的访问量）。只要 key 的个数不是太多，没必要设置 horizon。系统只是保持 key 对应的值的数量的 window 是活动的就行。
  （译者注：推测这种情况下系统不是按照时间来分割 window 的。）

好了，让我们进入第四个也是最后一个问题。

### How: accumulation

随着时间的推移，触发器会在一个 Window 中产生多个 Pane。我们遇到了最后一个问题：『后续数据的处理结果如何影响之前的处理结果？』。迄今为止的例子中，每个窗格的新数据都建立在紧邻的前一个窗格的数据之上。但是，实际上有三种不同的更新模式。（注释：事实上有四种模式，第四种是丢弃并更正--discarding and retracting，这并不常用，这里不会讨论它）

* **丢弃（Discarding）**：每当有窗格输出，过去的状态就会被丢弃，这意味着后续的窗格与之前的无关。应用场景是：下游消费者会自己进行累计更新操作。例如：那些只希望接收到差值（delta）来求和的系统，我们只需要把接收到的新数值发送给他就行。

* **累计（Accumulating）**：如在[图7中](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG7)，每一个窗格输出时，过去状态被保留，和未来的输入一起累加形成新的当前状态。这意味着每个后续的窗格建立在前面的窗格之上。累加模式应用在这种场景：后续结果可以简单地覆盖以前的结果，例如把输出的结果存储在 BigTable 或 HBase 这类 key/value 存储中。

* **累计并更正（Accumulating & retracting）**: 与累计模式类似，但是当产生新的窗格时，它会再单独产生一个被更正/回撤的值。(新的累加值 Y, 更正值 X) 的组合相当于在说: "之前告诉你的结果 X 是错的，别管那个值了，用新的值 Y 代替它吧！"下面这两种情况更正值特别有用：

  * 当下游消费者需要把不同维度的**数据重新分组**时，新产生的值完全有可能会与旧值不同，因此新数据最终会在不同的组中。在这种情况下，新值不能覆盖旧值; 您需要从*旧*组中删除旧值，而在*新*组中加入新产生的值。
  * 使用**动态窗口**时（例如 Session 窗口，下面我们会研究它），由于可能发生窗口合并，新产生的窗口会替换多个旧窗口。我们无法从新窗口推知要替代哪些旧窗口，因此最好明确告知我们要被撤回的窗口有哪些。

  让我们并排对比这些情况就更加明确了，查看图 7 中第二个窗口的 3 个窗格，下表展示了三种累加模式中每个窗格的值将如何变换。

  | **放弃**        | **积累** | **累积与收缩** |        |
  | ------------- | ------ | --------- | ------ |
  | **第 1 页：[7]**   | 7      | 7         | 7      |
  | **窗格 2：[3,4]** | 7      | 14        | 14，-7  |
  | **窗格 3：[8]**   | 8      | 22        | 22，-14 |
  | **最后观察值**     | 8      | 22        | 22     |
  | **总和**        | 22     | 51        | 22     |

* **丢弃**：每个窗格仅包含当前到达的值。因此，最终观测值不是 sum。但是，如果要自己单独计算所有窗格，就会就会得到正确的答案 22。这就是为什么丢弃模式在下游消费者自己会执行某种聚集操作时很有用。

* **累计**：如在[图7中](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG7)，每个窗格与之前已经到达的值相结合，加上从先前的窗格中的所有值。因此，最终观测值就是 22。但是，如果你自己又单独地求了一次和，那么会给出不正确的 51。这就是为什么当你可以用新值简单地覆盖以前的值时，累积模式是最有用的：新的值已经包含了迄今为止所有的数据。（译者注：类似 reduce 函数）

* **累计和缩回**：每个窗格都包含一个新的累积模式值以及前一个窗格值的撤回值。因此最后最终观测值（不含撤回值）值以及最终所有窗格观测值（包括撤回值）的总和都能提供正确的答案 22。这就是为什么退缩是如此强大。

要使用丢弃模式，我们将对 Listing 5 进行以下更改：

```java
// Listing 7. Discarding mode version of early/late firings.
PCollection<KV<String, Integer>> scores = input
  .apply(Window.into(FixedWindows.of(Duration.standardMinutes(2)))
               .triggering(
                 AtWatermark()
                   .withEarlyFirings(AtPeriod(Duration.standardMinutes(1)))
                   .withLateFirings(AtCount(1)))
               .discardingFiredPanes()) // 这一行
  .apply(Sum.integersPerKey());
```

在具有启发式 watermark 的流式引擎上再次运行代码将产生如下所示的输出：


<iframe height="400" width="600" src="https://fast.wistia.net/embed/iframe/64r8oawoc2?videoFoam=true&wvideo=64r8oawoc2"></iframe>
[图9. 流式引擎上的含有early/later的触发器的丢弃模式版本](https://fast.wistia.net/embed/iframe/64r8oawoc2?videoFoam=true&wvideo=64r8oawoc2)

虽然输出的整体形状类似于图 7 的累计模式，但请注意，此丢弃版本中的任何一个窗格都不重叠。因此，每个输出与其他输出是独立的。

如果我们实际看看更正/回撤模式，代码修改类似（但是，请注意，Google Cloud Dataflow 的回撤模式仍然处于开发状态，所以这个 API 中的命名有点推测，不太可能与他相同）：

```java
// Listing 8. Accumulating & retracting mode version of early/late firings.
PCollection<KV<String, Integer>> scores = input
  .apply(Window.into(FixedWindows.of(Duration.standardMinutes(2)))
               .triggering(
                 AtWatermark()
                   .withEarlyFirings(AtPeriod(Duration.standardMinutes(1)))
                   .withLateFirings(AtCount(1)))
               .accumulatingAndRetractingFiredPanes()) // 这一行
  .apply(Sum.integersPerKey());
```

在流式引擎上运行，结果如下：

<iframe width="800" height="450" src=" https://www.youtube.com/embed/kU_hPrgEIis" title="Figure 09   streaming speculative late discarding" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[图9. 流式引擎上的含有early/later的触发器的更正模式版本](https://fast.wistia.net/embed/iframe/ksse5yiq3u?videoFoam=true&wvideo=ksse5yiq3u)

由于窗口内的窗格相互关联，不太容易看清这些回撤值。回撤值用红色标记了，他与蓝色的窗格重叠，所以看起来像是紫色。每个窗格的两个输出值用逗号分割使他们更容易区分。

让我们把图 7（只有启发式的版本），9，10 放到一起比较一下：

![20241212105125848](http://fodi.limuzhi.us.kg/images/IMG-860d7d4e9943e1e4.webp)

你可以想到，在存储和计算成本方面，丢弃模式，累计模式，累计&更正模式成本是不断增加的。为此，积累模式的选择也是对正确性，延迟和成本进行权衡。

## 插曲

至此，我们已经回答了所有四个问题：

- **What results are calculated?**：做什么计算？答案是：通过 transform 操作。
- **Where in event time are results calculated?**：计算什么时间范围的数据？答案是：通过窗口化。
- **When in processing time are results materialized?** ：何时将计算结果输出？这答案是：使用 watermark 和 trigger 配合触发计算。
- **How do refinements of results relate?**：后续数据的处理结果如何影响之前的处理结果？答案是：通过 Accumulation 模式来决定。

但是我们只看了一种类型的窗口化例子：Event-Time 的固定窗口。通过流式 101 我们知道，还有许多方式进行窗口化，今天我还想讨论其中的两种：**Processing-Time 的固定窗口**和**Event-Time 的 Session 窗口**。

### *When*/*Where*: Processing-time windows

Processing-Time 窗口十分重要有两个原因：

* 对于某些场景，例如用量监控（例如，Web 服务的流量 QPS），您希望在观察到数据流时就分析数据，Processing-Time 窗口绝对是合适的方式。
* 对于事件的实际发生时间很重要的场景（例如，分析用户行为趋势，计费，评分等），使用 Processing-Time 绝对是**错误**的选择，能够识别这些场景十分重要！！

因此，值得深入了解 Processing-Time 窗口和 Event-Time 窗口之间的不同，特别是考虑到当今大多数流式系统中 Processing-Time 窗口已经被普及。

当我们面对的模型是严格使用 Event-Time 作为基础时（例如本文的例子），有两种方式来获得 Processing-Time 窗口：

* **触发器：**忽略 Event-Time（即使用跨所有 Event-Time 的一个全局窗口，注：也可以理解成没有窗口），并配合使用触发器在 Processing-Time 轴中定时触发，来提供该窗口的快照。
* **使用入口时间**：当数据到达时，将进入系统的时间赋值给 Event-Time，然后使用正常的 Event-Time 窗口即可。这是 Spark Streaming 的实现方式。

请注意，这两种方法基本等价，但是在有多个 Stage 的 pipeline 的情况下，它们略有不同：在触发器版本，每个 Stage 都是独立地分割 process-time 窗口，因此，在第一个 Stage 的窗口 X 中出现的数据，可能在下一个 Stage 的窗口 X-1 或者窗口 X+1 中出现。而在使用入口时间作为 Processing-Time 的版本中，一旦将数据并入到窗口 X 中，那么在整个 pipeline 中都会在窗口 X 中，这是由于不同 stage 会通过 watermark（在 DataFlow 情况下，Spark Streaming 是通过 micro-batch 的边界来同步）或其他引擎级别的方式来同步处理进度。

**使用 Processing-Time 划分窗口的最大缺点就是：当输入数据被观测到是顺序发生改变时，窗口的内容就改变了。**为了用更具体的方式研究这一点，我们比较下面三种场景：

- **基于 Event-Time 的窗口**，就是我们之前讨论的窗口，不是 Processing-Time 的，作为比较的基准。
- **通过触发器实现的 Processing-Time 窗口**
- **通过入口时间实现的 Processing-Time 窗口**

我们会在这三种场景上分别使用两个不同的数据集（所以，一共会有 2*3 中情况）。这两个数据集除了观测到的顺序不同，其他均完全相同（比如相同的值发生相同的 Event-Time）。第一套数据集就是我们之前一直看到的那个顺序（白色标记）。第二套所有的值都在 Processing-Time 的轴上移动了（如图 12 所示，使用紫色标记）。你可以想象一下，只要发生一点意外（比如使用复杂的分布式系统就会打乱顺序），这种情况就会发生。
<iframe width="800" height="450" src=" https://www.youtube.com/embed/1QQlpwabje8" title="Figure 12   input toggle" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[图12 -  移动输入值在Processing-Time上的位置，同时保持相同的值和Event-Time](https://fast.wistia.net/embed/iframe/lf3v07a065?videoFoam=true&wvideo=lf3v07a065)

#### Event-time windowing（使用 Event-Time 窗口化）

为了建立一个比较基线，我们先看一下使用了 Event-Time 固定窗口搭配启发式 watermark 的引擎分别作用到这两个数据集上的结果。我们复用 Listing 5、图 7 中使用了 early/later 触发器的代码，得到的结果如下。左侧和我们以前看到的结果一样，右侧是使用了第二种顺序的数据集的结果。重点是：虽然输出过程的形状不一样，但是**最终四个窗口的输出结果全部相同**：14，22，3 和 12。

<iframe width="800" height="450" src=" https://www.youtube.com/embed/QiuMUsBzYV8" title="Figure 13   streaming speculative late toggle joint" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[图13.在Event-time固定窗口上使用两个不同Processing-Time顺序的数据集的结果](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102?wvideo=hnl6whv23j#F6)

#### Processing-time windowing via triggers（使用 Trigger 实现的 Processing-Time 窗口化）

好，现在让我们来比较一下使用了 Processing-Time 的两种方案。首先是 Trigger 方式。我们从三个角度来看这种方式的 Processing-Time 窗口是如何工作的：

* **窗口化方式**：由于我们使用 Event-Time 来模拟 Processing-Time，我们使用全局的 Event-Time 窗口（就是没有窗口）。
* **触发方式**：根据期望的窗口大小，在 prcessing-time 时间域进行周期性触发。
* **累加方式**：设置窗口的累加方式为『丢弃』，可以让 Processing-Time 窗口看起来互相独立。（译者注：这里全局的 Event-Time 窗口会周期性触发多次，但是从 Processing-Time 角度看一个窗口触发一次，且相互之间没有关系）

相应的代码如 Listing 9，注意：全局窗口是默认行为，，因此无需指定窗口化策略：

```java
// Listing 9. Processing-time windowing via repeated, discarding panes of a global event-time window.
PCollection<KV<String, Integer>> scores = input
  .apply(Window.triggering(
                  Repeatedly(AtPeriod(Duration.standardMinutes(2))))
               .discardingFiredPanes())
  .apply(Sum.integersPerKey());
```

当在流式引擎上运行上述代码分别处理上述两组不同观测顺序的数据集，结果如下面图 14 所示，一些值得注意的点有：

* 由于我们是通过 Event-Time 的窗格（Pane）来模拟 Processing-Time 窗口的，所以图中 Processing-Time 轴才是窗口，就是图中 Y 轴的宽度。
* 由于 Processing-Time 窗口对输入数据的顺序很敏感，所以，对于观测顺序不同的数据集，窗口的内容不同。左边的结果是 12，21，18，右侧的结果是 7，26，4。

<iframe width="800" height="450" src=" https://www.youtube.com/embed/Dgu8joaqSN4" title="Figure 14   proc time discarding joint" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
[图14，通过触发器实现的Processing-Time窗口，分别在两组不同观测顺序的数据集上运行](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102?wvideo=befc8n62yh#F6)

#### Processing-time windowing via ingress time（使用入口时间实现的 Processing-Time 窗口化）

最后，让我们看看使用输入数据的入口时间作为 Event-Time 来实现 Processing-Time 窗口的方案. 在代码方面，这里有四个方面值得一提：

* **修改 Time 值**：当一个数据到来时，他的 Event-Time 需要用入口时间来覆盖。注意，DataFlow 中还没有对应的统一标准 API，以后可能会有（下面的 I/O 源的代码是虚构的伪代码）。对于 [Google Cloud Pub/Sub](https://cloud.google.com/pubsub/)，您只在发布消息时将 `timestampLabel` 的字段设为空即可，其他数据源请参考文档。
* **窗口化方式**：使用标准的 Event-Time 固定时间窗口即可。
* **触发方式**：由于使用入口时间，所以我们可以使用完美的 watermark。因此使用默认触发器即可，它默认可以实现在 watermark 线通过窗口终点时触发一次。
* 累加模式：由于每个窗口只有一个输出（译者注：只触发一次，即 watermark 那一次），所以累加模式不重要，随意设置。

```java
// Listing 10. Explicit default trigger.
PCollection<String> raw = IO.read().withIngressTimeAsTimestamp();
PCollection<KV<String, Integer>> input = raw.apply(ParDo.of(new ParseFn());
PCollection<KV<String, Integer>> scores = input
  .apply(Window.into(FixedWindows.of(Duration.standardMinutes(2))))
  .apply(Sum.integersPerKey());
```

在流式引擎上的执行过程如下图 15 所示。当数据到达时，它们的 Event-Time 被更新以匹配它们的入口时间（即到达时的 Processing-Time），随着 Processing-Time 推移，完美的 watermarker 线向右水平移位。该图中注意点：

* 输入数据顺序改变，结果也会改变
* 与 Trigger 方式不同，窗口在 X 轴（Event-Time 时间域）中被描绘。但是它表示的含义已经不是 Event-Time 了。
* 窗口触发的时间也与 Trigger 相同，产生的结果也相同，都是：左边 12，21，18；右边 7，36，4。
* 由于使用了完美的 Watermark，而且 X 轴表示入口时间，所以，watermark 线与理想情况下的 watermark 线重合，都是向右上方，且斜率为 1。

<iframe width="800" height="450" src=" https://www.youtube.com/embed/Tgjj_FSjV5c" title="Figure 15   ingress time joint" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[图15，在相同输入的两种不同处理时间排序中，通过使用入口时间进行处理时间窗口划分](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102?wvideo=bahkp8byjn#F6)

虽然使用这两种方式实现 Processing-Time 窗口很有趣，但是：这里你要明白的最重要的一点是：Event-Time 窗口至少在运算结果上是输入数据顺序无关无关（在输入完成之前，实际的 Panel 会不断变化），而 Processing-Time 窗口却不是这样。**如果你关系事件发生的真实时间，必须使用 Event-Time 窗口，否则计算结果毫无意义！！**。

### **Where: session windows**（会话窗口）

我们就要讲完所有的例子啦。。。如果你读到这里，真的很有耐心。好消息是你的耐心没有白费，下面我会讲一个我最喜欢的特性：数据驱动的动态窗口—Session。请看好了。

Session 是一种比较特殊的窗口，它会用某段不活动的间隔为界来捕获一段时间内的事件。在数据分析领域，这种窗口特别有用：它能从用户视角提供用户某段时间内参与了哪些活动。在 Session 内看到关联的活动，根据会话的长度推断出用户的参与度。

从窗口化的视角，Sesiion 在下面这两方面特别有意思：

* 他们是**数据驱动 Window**的一个例子：窗口的大小和位置都是由输入数据本身决定的，而不像固定/滑动窗口那样有我们预先定义好的时间模式。
* 他们也是**不对齐 Window**的一个例子：这种窗口没有统一地应用到所有数据上，而只是应用到该数据的一个特定子集（如，每个用户）。把它与对齐 Window 进行对比：对齐 Window 会把类似固定/滑动窗口统一地应用到整个数据集上面。

在某些场景下：对于单个 Session 内的数据，我们可以在时间的前面加一个通用的标志符来标记它属于某个 Session（例如：视频播放器的携带服务质量信息心跳包 ping。对于某次观看，可以在所有的 ping 的时间前面加上 Session ID）。那么，这个例子的 Session 十分容易构建了，只要把数据按照 Key 聚合就是一个 Session。

但是，一些更通用的场景（例如：Session 自己没法知道它已经开始了）需要这样：Session 必须根据数据在时间内的位置来构建。在无序数据集的情况下，这十分棘手。

要提供通用的 Session 支持，一个核心的洞察是：**一个 Session Window 可以看成是相互重叠的小 Window 的组合，其中这里的小 Window（ proto-session window）比较特别，他们只含有一个 Record，窗口大小是 inactivity 的时间。因此，即使我们看到的数据是无序的，我们可以随着数据一个一个来到，简单地把这些有重叠的小 Window 合并起来组成一个最终的 Session Window。**

![20241212105125947](http://fodi.limuzhi.us.kg/images/IMG-d858f21011c62f17.webp)

我们来看一个代码示例，通过使用 Listing 8 的代码（包含了 early/late 触发器和 retract 的更新机制）来构建 Session：

```java
// Listing 11. Early and late firings with session windows and retractions
PCollection<KV<String, Integer>> scores = input
  .apply(Window.into(Sessions.withGapDuration(Duration.standardMinutes(1))) // 这一行
               .triggering(
                 AtWatermark()
                   .withEarlyFirings(AtPeriod(Duration.standardMinutes(1)))
                   .withLateFirings(AtCount(1)))
               .accumulatingAndRetractingFiredPanes())
  .apply(Sum.integersPerKey());
```

在流式引擎上允许结果如下图 17：

<iframe width="800" height="450" src=" https://www.youtube.com/embed/K3OTMCeS_iE" title="Figure 17   sessions" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[Figure 17 - 具有early/late触发器和retract更新模式的Sessions窗口，一个生动的例子](https://fast.wistia.net/embed/iframe/5eu4m6fgp1?videoFoam=true&wvideo=5eu4m6fgp1)

上图内容是否丰富，我们一起看一下：

* 当遇到第一个值 5 的时候，它被放入了一个小 Window（起始位置是该 Record 的 Event-Time，大小是 inactivity 时间，例如 1 分钟）。以后遇到的任何与这个重叠的 Window 都会合并到一起。
* 第二个到的是 7，因为它没有与第一个小窗口 5 重叠，所以类似，它被放入了第二个小 Window。
* 与此同时，watermark 线到达了第一个 Window 终点，所以，第一个窗口 5 作为预测值在12:06之前被产出。很快第二个 watermark 也到了，第二个窗口 7 也作为预测值在12:06之后被产出。
* 下面陆续看到 3，4，3。他们的小窗口相互重叠，最终被合并到了一起。在12:07时遇到了 early 触发器触发，一个窗口含有值 10 被输出。
* 很快 8 又来了，他与小窗口 7 和窗口 10 重叠，他们被合并到一起形成了一个含有值 25 的 Session。当 watermark 到达这个 Session 的终点时，值被输出：他具有值 25 和两个之前的值的 retract（7 和 10）
* 9 到达时也发生了相似的事情，与值 5 还有值 25 的 Session 合并形成了一个更大的值 39。在遇到 late 触发器触发时，39 和 retract 值（5 与 25）都被输出。

这很强大！更令人惊叹的是：在这么一个模型中描述他们是这么地简单。这个模型把流处理的按纬度分解成可区分可组合的部分的。最后，我们只要关系业务逻辑，而不用操心怎么把数据组合成有用的形式。

如果你不觉得这有多好，看看这篇文章[如何使用Spark Streaming手动建立Session](http://blog.cloudera.com/blog/2014/11/how-to-do-near-real-time-sessionization-with-spark-streaming-and-apache-hadoop/) 有多痛苦就懂了（注意：我没有职责他们，Spark其实已经做了不错了，现在已经有人努力去写文章说明如何在 Spark 之上建立一个特定的 Session）。当然，他们也没有合适的 Event-Time Session，也没有提供 early, late 的触发机制，还没有 retract 的更新模式。

## 终结篇，感觉好极了！

就这些了，讲完了所有的例子，撒花，撒花！你已经有了健壮的流式处理的基础知识，可以出发进入流式的世界。在结束之前，我快速回顾一个我们所涉及的内容，以免你错过了什么。首先，我们所涉及的主要概念：

* **Event-Time VS Processing-Time（事件时间和处理时间）**：最主要的区别是一个是事件发生的真实时间，另一个是数据被数据处理引擎观测到的时间。
* **Windowing（窗口化）**：把数据沿着时间边界分割是处理无界数据的最常用的策略。（无论是 Processing-Time 还是 Event-Time，在 DataFlow 中我们狭义地定义窗口化是用 Event-Time）
* **Watermarks（水印）**：一个时间时间相关的十分有用的概念，提供了在无序/无界数据集上推理数据完整性的一种方式。
* **Triggers（触发器）**：一种告知系统何时可以输出数据的声明机制。在某些场景中十分重要。
* **Accumulation（累计模式）**：当单个窗口需要不断更新多次数据并提供更精确的结果时，不同结果之间如何处理的方式。

其次，我们研究了下面四个问题（我承诺这是最后一次说它们）

* **What results are calculated?**：做什么计算？（计算逻辑是什么，如何表示？） == Transform
* **Where in event time are results calculated?**：计算什么时间范围的数据？ == Windowing
* **When in processing time are results materialized?** ：何时将计算结果输出？ == Watermarks + Triggers
* **How do refinements of results relate?**：后续数据的处理结果如何影响之前的处理结果？ == Accumulation

最终，深入这种流式处理模型的提供的灵活性（因为最终我们就是平衡准确性-Correctness，延迟-Latency 和成本-Cost, 这些因素关系）。回顾一下：我们只要修改一点点代码就能实现在相同数据集上各种不同的产出：

图 18. 相同输入数据，对应 9 种不同的产出

| ![20241212105126183](http://fodi.limuzhi.us.kg/images/IMG-695617e4e38222f6.webp) Classic batch: [Listing 1](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L1) / [Figure 2](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG2)                 | ![20241212105126276](http://fodi.limuzhi.us.kg/images/IMG-3c2c05b1dc1b1f24.webp) Fixed windows batch: [Listing 2](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L2) / [Figure 4](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG4)                                                                                  | ![20241212105126519](http://fodi.limuzhi.us.kg/images/IMG-483962b3e886f9b3.webp) Fixed windows streamingwatermark: [Listing 2](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L2) / [Figure 6](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG6) |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![20241212105126612](http://fodi.limuzhi.us.kg/images/IMG-a5512113c6e0c896.webp) Early/late discarding: [Listing 7](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L7) / [Figure 9](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG9)         | ![20241212105126813](http://fodi.limuzhi.us.kg/images/IMG-87de848e44cde18d.webp) Early/late accumulatingListings: [4](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L4) & [5](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L5) / [Figure 7](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG7) | ![20241212105126929](http://fodi.limuzhi.us.kg/images/IMG-f7944343c7a67d31.webp)  Early/late retracting: [Listing 8](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L8) / [Figure 10](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG10)         |
| ![20241212105127034](http://fodi.limuzhi.us.kg/images/IMG-2dad0493ca013959.webp)  Processing-time (triggers): [Listing 9](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L9) / [Figure 14](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG14) | ![20241212105127145](http://fodi.limuzhi.us.kg/images/IMG-151fde0313842de0.webp) Processing-time: (ingress time) [Listing 10](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L10) / [Figure 15](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG15)                                                                   | ![20241212105127231](http://fodi.limuzhi.us.kg/images/IMG-b866f4c8435face8.webp) Sessions: [Listing 11](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#L11) / [Figure 17](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102#FIG17)                     |

图 18：**相同输入数据，对应 9 种不同的产出.** Credit: Tyler Akidau.

感谢您的耐心和兴趣，我们下次再见！

## 后记

### 额外的资源

* DataFlow 的[文档](https://cloud.google.com/dataflow/docs/)，[以手机游戏为例在4种场景下的例子讲解](https://github.com/GoogleCloudPlatform/DataflowJavaSDK-examples/tree/master/src/main/java8/com/google/cloud/dataflow/examples/complete/game)，[例子的github代码](https://github.com/GoogleCloudPlatform/DataflowJavaSDK-examples/tree/master/src/main/java8/com/google/cloud/dataflow/examples/complete/game)
* Dataflow 的[视频演讲](https://www.youtube.com/watch?v=3UfZN59Nsk8)
* 学术文章：[VLDB的paper](http://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf)，虽然没有本文生动，但是提供了很多细节。而且可以从他的参考文件进一步研究开去。

### 文章中一些与现实不同的地方（略）

## 致谢（略）

## 注脚（略）

部分翻译到文章中。

## 作者简介

Tyler Akidau 是 Google 的软件工程师。目前，是 Google 内部流式数据处理系统（例如“MillWheel”）的技术主管，他花了五年时间研究大型流式数据处理系统。他热衷于将流数据处理视为更为一般的大规模计算模型。他最喜欢的交通工具是可以带着他两个小女儿的 cargo bike。



参考文章：

* [谷歌Dataflow编程模型和spark 2.0 structured streaming](http://blog.csdn.net/colorant/article/details/52163896)：介绍了 Dataflow 模型和 spark/beam 等项目，本文参考部分专业用语的翻译。



