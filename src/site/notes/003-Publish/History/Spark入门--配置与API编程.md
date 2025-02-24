---
{"dg-publish":true,"dg-path":"History/Spark入门--配置与API编程.md","permalink":"/History/Spark入门--配置与API编程/","title":"Spark入门--配置与API编程","tags":["技术","大数据分析","Spark"],"created":"2017-01-29 16:28:53","updated":"2017-01-29 16:28:53"}
---



参考资料：

* 书籍：《Spark快速大数据分析》
* 官方文档：
  * 编程
    * [快速入门](http://spark.apache.org/docs/latest/quick-start.html)&&[提交作业文档](http://spark.apache.org/docs/latest/submitting-applications.html)&&[基本术语](http://spark.apache.org/docs/latest/cluster-overview.html#glossary)：基本入门须知
    * [编程指南](http://spark.apache.org/docs/latest/programming-guide.html)：介绍了Spark的基本API（RDD等）必看资料
    * [Spark SQL编程文档](http://spark.apache.org/docs/latest/sql-programming-guide.html)：介绍了Spark的DataFrame与DataSet API，也是必读资料。
  * 配置
    * [spark配置文档](http://spark.apache.org/docs/latest/configuration.html)：spark的基本配置，如`spark-defaults.conf`文件
    * [在yarn上运行spark](http://spark.apache.org/docs/latest/running-on-yarn.html)：在yarn上运行Spark的一些配置，如外部shuffle
  * 原理
    * [Spark的集群模式简介](http://spark.apache.org/docs/latest/cluster-overview.html)：集群cluster模式下Spark的运行组成。
    * [job调度，资源分配](http://spark.apache.org/docs/latest/job-scheduling.html)：介绍了调度的策略（应用间与应用内，动态资源分配）
  * [spark调优指南](http://spark.apache.org/docs/latest/tuning.html)：从序列化，内存等角度讨论了Spark的性能优化。
  * [监控和一些工具](http://spark.apache.org/docs/latest/monitoring.html)：Spark监控相关的设置，API接口，一些测量Metric的方法。


知识图谱：

## 简介

Spark的官方介绍是：Apache Spark is a **fast** and **general-purpose** cluster computing system。这体现了Spark两个关键的特点：快速和通用。快速是指Spark与Hadoop相比具有可以在内存内快速处理数据的特点；通用是指Spark是一个大一统的软件栈，能使用一套API完成以前需要多个平台才能完成的任务，如：交互查询（Spark Shell），离线批量数据处理（RDD、Spark SQL），在线的流式处理（Spark Streaming），以及在大型数据集上运行迭代算法（ML任务）。

从工程角度看Spark具有下面一下特点：

* 支持多种语言编程，Scala（优先推荐，也是编写Spark的语言），Python，Java（不推荐，编写代码复杂）。
* 具有Local模式，Local[n]（类似伪分布式）模式，集群模式（StandAlone/Yarn/Mesos三种方式）。
* 可以读取所有支持HDFS接口的文件系统（本地文件，HDFS上的文件，Hive，Hbase，亚马逊S3）
* 底层文件系统推荐HDFS，同时推荐集群模式运行在Hadoop Yarn之上（Why？？）

Spark能完成的一些任务：

* 交互式分析与查询：用于快速的数据分析或者调试程序。
* Spark应用：以jar包的形式提交的Spark任务，可以执行离线（批量）或者在线任务（流式）。

## 基本概念

* SparkContext：我们程序与Spark打交道的入口，代表对集群的一个连接。无论如何使用Spark，第一步永远都是初始化SparkContext(理解为运行的上下文！)，之后可以使用sc进行一些列操作，如配置集群，读取文件等。

* Driver（驱动程序） && Executor（执行器）：驱动程序简单理解成Master，全局唯一，实际运行我们编写的程序（即**main函数入口**），**最终解析成一个个Task**发送给下面的工作节点执行具体的操作。执行器理解成Slave，在系统中存在多个，并且对于一个Application而言，Executor是存在与它完整的生命周期中的（常驻的）。因此，依据这个基本原理，要注意：**我们编写程序有一部分是运行在Driver中，一部分是发送到Executor中的执行**。<u>常见的错误</u>是在foreach的回调函数中修改外部变量。本质上是修改的每个Executor的副本。

* Client Mode 与 Cluster Mode：两者最大的区别是Driver程序的位置。如果Driver在本地则是Client Mode（典型地，交互式应用Spark Shell）。如果Driver在集群中的某个Node运行，则是Cluster模式。（对于Yarn模式的Client，Driver在Application Master中）

  ![](https://fodi.389266.xyz/images/IMG-2589dfa89e201c67.webp)

* Application && Job && Stage && Task：这几个概念十分容易混淆，从左到右由大变小。

  * Application：Spark上运行的用户程序，包括Driver和Executor。一个Application包含多个Job。
  * Job：作业，由**Spark的驱动程序中的Action操作触发**。它代表了一系列操作。
  * Stage：把一个Job按照某种规则拆分成n块，每一块是一个Stage。（规则简单理解成是否有网络数据传输Shuffle）
  * Task：实际执行的最小单元，任务的最小单元由Executor具体执行。

* 一些常识概念：

  * 失败重试：某个节点的Task失败会自动重试。
  * 长尾任务：某些节点很慢的情况下，会自动启动其他节点运行，哪个节点先完成则使用谁的结果。
  * 就近执行：尽量减少数据网络传输，尽量在本地、一个机架内完成数据处理。

## 开发环境配置

详细参考官方文档，这里主要介绍Yarn模式

* [spark-submit提交应用文档](http://spark.apache.org/docs/latest/submitting-applications.html)：应用提交命令的用法
* [Spark配置文档](http://spark.apache.org/docs/latest/configuration.html)：配置大全
* [Yarn配置文档](http://spark.apache.org/docs/latest/running-on-yarn.htm)：一些仅仅在Yarn下有效的配置
* [Moniror配置文档](http://spark.apache.org/docs/latest/monitoring.html)：配置History服务器

### 配置

配置上面提到的几种模式，这里重点提一下可以设置的地方。参考[配置文档](http://spark.apache.org/docs/latest/configuration.html)。一个技巧，可以在`spark-submit`时加上`—verbose`来打印参数的来源或者在Spark Web UI上查看使用的设置。

* conf目录下（低）
  * `spark-default.xml`：这两个文档中对参数有详细说明。[Spark配置文档](http://spark.apache.org/docs/latest/configuration.html)，[Yarn配置文档](http://spark.apache.org/docs/latest/running-on-yarn.htm)，[Moniror配置文档](http://spark.apache.org/docs/latest/monitoring.html#spark-configuration-options)
  * `spark-env.sh`： [Spark配置文档环境变量一节](http://spark.apache.org/docs/latest/configuration.html#environment-variables)，[Monirot的配置环境变量一节](http://spark.apache.org/docs/latest/monitoring.html#environment-variables)。配置`HADOOP_CONF_DIR`，`HIVE_CONF_DIR`这些常用的变量。
  * `log4j.properties`： 控制日志输出的等级。在conf下有模板。参考 [log4j](http://logging.apache.org/log4j/) 的官方文档。
* `spark-submit`的参数（中）
  * 命令行选项opition：如`--master`、`--name`等，基本都可以在`—conf`中找到对应的配置。**在上面的文档的表格`Meaning`中可以找到对应的opinion**。
  * `--conf<key>=<value>`方式：有很多内置的key，如`spark.app.name`、`spark.master`。**与`spark-default.xml`中的配置的key一致。**
* `SparkConf` 代码设定(高)：`val conf = new SparkConf() .setMaster("local[2]") .setAppName("CountingSheep") val sc = new SparkContext(conf)`
  * ​

### 启动

```shell
./bin/spark-submit \
  --master yarn \
  --deploy-mode client \  # 或者cluster
  --executor-memory 20G \
  --num-executors 50 \
  --queue spark-default # Yarn模式下 指定队列
  --conf <key>=<value>
  --jars PATH/utils.jar # 制定了第三方的jar包
  --class org.apache.spark.examples.SparkPi \ # 指定jar包中运行的Main Class
  /path/to/examples.jar \ #<application-jar>
  1000 #[application-arguments]
```

* spark-submit 运行Application的基本要素
  * `--master yarn` 指定启动模式，如
    - `local`（单核）、`local[2]`、`local[*]`(依据cpu设定)：注意使用单核local模式时比较特殊，一下逻辑可以正常运行，不代表在多核或者其他模式下可以运行，因此此模式仅供测试。
    - `yarn`：最**常用**的模式，在yarn上运行代码，配置hadoop文件的位置在`HADOOP_CONF_DIR`或者`YARN_CONF_DIR`中。改变量在`conf/spark-env.sh`中配置
    - 其他地址:`spark://spark_alone_mode_addr`,`mesos://mesos_mode_addr`
  * `--deploy-mode client`：运行模式。**使用spark-submit提交job默认是`client`**。对于一般短期例行应用，使用client模式配合调度系统（如Azkaban）的retry机制即可。备选`cluster`模式可用。
  * `--calss com.package.xxxClass` ：指定运行的main class 
  * `<application-jar>` && `[application-arguments]`：注意package时不要提交spark、hadoop自身，较少包体积。参数可以通过main的args读取。当然，也可以使用`—conf`来传参.
* 其他选项
  * 指定与提交依赖（常用）：经常有写第三方或者工具jar要调用，可以用这个方式。应当注意的是**必须确保executors和driver必须能访问到该jar文件**（如共享或者copy）
    *  `—-jars path/xxx.jar,path/xxx2.jar`： 多个jar则使用`,`分割。path支持
      * `local`：本地文件目录，注意，必须手动（或者由调度系统）**copy相同的jar到所有机器的相同目录下面**。当然也可以是NFS这些共享文件系统的目录。
      * `file://`：driver的绝对路径，从Driver内置的HTTP Server复制过去
      * `hdfs://`：从hdfs文件系统访问
      * `http://` `https://` `ftp://`：其他可以访问的url
    *  通过maven 依赖方式:
      * `—packages`
      * `—repositories`
    *  其他注意点：jar的复制可能会占用不少空间，yarn模式会自动清理这些jar。其他模式请手动处理。
  * **指定queue**（Yarn方式特有）： `--queue spark-default`
  * spark-submit的其他option参数设定
    * `--num-executors 20`
    * `--executor-memory 20G`
    * `--driver-memory 3G`
    * `--executor-cores 3`
    * ..参考文档中其他的配置。
  * `--conf <key>=<value>` 。
    * 添加系统配置。
    * 可以通过添加多个 `--conf`设置多个配置
    * 自定义参数
      * `--conf spark.test.arg.mypath=hdfs://xxx` 自定义参数。
      * 获取`String value = System.getProperty("spark.test.arg.mypath");`
* jvm控制变量
  * `-conf "spark.driver.extraJavaOptions =-XX:+PrintGCDetails"`:给Driver的JVM设置参数，注意不能设置heap内存这些，要通过spark.driver.memory来设置。**Client模式下不适用，因为Driver的JVM此时已经启动了，使用--driver-java-options设置**。
  * `-conf "spark.executor.extraJavaOptions =-Dlog4j.configuration=file_path"`：给Executor的JVM设置参数
  * `--driver-java-options "-Dfile.encoding=UTF-8 -Dsun.jnu.encoding=UTF-8"`：给Client模式的Driver设置虚拟机参数，常常用在设置spark-shell的字符编码.
* spark-shell：本质上还是spark-submit，配合了一个scala的shell。也可以传递依赖的jar包
  * 命令行字符问题处理`--driver-java-options "-Dfile.encoding=UTF-8 -Dsun.jnu.encoding=UTF-8"`
  * jar包上传

### 配置的介绍

* 应用配置：应用模式，名称，内存
* 运行环境：java环境，库
* shuffle相关
* sparkUI的设置
* 序列化和压缩
* 内存管理
* 执行器Executor行为
* 网络
* 调度schedule，资源分配
* 安全，加解密
* SparkStreaming

## 基本API

### RDD

RDD（Resilient Distributed Dataset）是**不可变**的分布式**对象集合**。是Spark最重要的概念，是对分布式数据的一个抽象表示。重点理解：

* 不可变：对RDD的每一次操作都会返回新的RDD，不会修改原有的集合。
* 对象集合：RDD中包含了**同一种类型**的对象，如String的RDD既RDD中的所有元素都是String。
* 惰性求值：只有执行Action时，才会真正运行一系列操作（包括读入数据）。
* RDD包含5个重要属性：分区表，一个用于计算每个split的函数（具体逻辑？），依赖的其他RDD的list，key-value RDD需要一个Partitioner（可选），计算每个split时优先使用的location（数据本地化，可选），[参考](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.rdd.RDD)。

#### 创建RDD

* 直接从本地文件，HDFS上的文件上创建 `val textFile =sc.textFile("README.md")`。具体API和支持的文件类型参考IO一节。
* 从内存集合数据创建 `val distData = sc.parallelize(Array(1, 2, 3, 4, 5))`。一般用于测试，例如：使用File打开一个文件读入到内存，然后转换成RDD。

#### <u>RDD的分类（重点）</u>

* 普通RDD：既包含了任意某种类型对象的RDD。如RDD[String]，具有统一的Transform和Action操作。
* 特殊类型的RDD：RDD的元素是特定的类型的RDD，除了通用方法外还有其他方法。
  * [Tuple2](http://www.scala-lang.org/api/2.11.7/index.html#scala.Tuple2)对象：即Key-Value对象的RDD，通过Scala隐式转换使它具有很多特有的方法。如ReduceByKey。
    * 如果key实现了Ordered，那么还有sortByKey操作
  * Double数值对象：RDD[Double]，数值类型在RDD中表示为Double。具有一下统计相关的特有方法，如计算均值。

#### 核心API

参考官方[API手册](http://spark.apache.org/docs/latest/api/scala/index.html#package)

> 技巧：由于**Scala的隐式转换**特性，查找API文档时需要注意，对于RDD的一下操作，除了[RDD](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.rdd.RDD)中通用操作，还有一下其他的类需要查看**XXRDDFunctions**相关的类，如[DoubleRDDFunctions](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.rdd.DoubleRDDFunctions)、[PairRDDFunctions](http://spark.apache.org/docs/latest/api/scala/org/apache/spark/rdd/PairRDDFunctions.html)。

* 通用 RDD
  * Transform
    * 针对每个元素的操作：
      * map：不改变元素的个数
      * filter：元素的个数小于等于old
      * flatMap：元素的个数大于等于old
    * 伪集合操作：
      * distinct
      * union：并集，结果包含重复数据
      * intersection：交集，结果自动distinct了。需要shuffle
      * subtract：差集，需要shuffle
      * certesian：笛卡儿积，即所有（a,b）对。
    * 其他
      * sample：生成采样的RDD（还是分布式的，区分takeSample）
  * Action
    * 数据整合
      * reduce：例如`rdd.reduce((x,y)=>x+y)`
      * fold：与reduce类似，但是提供初始值，例如`rdd.fold(0)((x,y)=>x+y)`
      * aggregate：与reduce类型，但是**可以返回不同的数据类型**。注意参数除了初始值外，需要提供两个函数分别用于集合内部value合并&&集合之间合并。demo参考《Spark快速大数据分析》的p35
    * count：计算RDD中元素的个数
    * countByValue：对**元素内容统计计数**
  * foreach：对每个函数调用某个方法。注意与map不同，**它没有副作用（返回RDD）**。典型的例子是用在把每个元素调用接口发送到其他地方。
  * collect/take(n)/top/takeSample/takeOrdered:返回驱动器的一系列操作，一般用于调试目的。


* Key-Value RDD 
  * pairRDD的生成
    * 某些文件读取的就是pairRDD
    * 通过map处理后生成
  * Transform
    * 单个pairRDD的方法
      * reduceByKey：与reduce区分，这里是Transform操作，生成了一个RDD，key是以前的key，value是这个key下reduce计算的结果。
      * **combineByKey**：reduceByKey的加强版。可以返回与输入不同类型的值。参数需要提供3个函数，1.为某个key的提供初始值的函数，2.该key集合内部value合并 3.该key集合之间合并。demo参考《Spark快速大数据分析》的p47
      * groupByKey：按key集合。key是以前的key，value是一个list，包含了这个key下的所有的value。
      * **mapValues**：保留Key，值修改Values，注意，这个方法很重要，它保留了原来RDD的分区信息。
      * flatMapValues
      * **mapPartitions**：一个分区调用一次该函数，参数是一个迭代器。
      * keys/values：返回所有key/value
    * 用于两个pairRDD的方法
      * join
      * rightOuterJoin/leftOuterJoin：右连接、左连接
      * **subtractByKey**：删除调key相同的元素
      * cogroup：按照相同的的key分组。注意结果中key是原来的key，**value是一个元组**，类似于先对两个RDD使用group，然后join。
    * 其他
      * sortByKey，对于key是Ordered的，可以对key排序。
* Double RDD
  * stats()：返回StatsCounter对象，包含了一系列统计值，如min max sum mean等
  * mean，sum
  * variance

#### 核心概念（重点）

* partition分区的**方式（只有Key-Value RDD可以设置Partitioner）**与**数目**
  * 默认的读入的数据的分区数目
    * 内存数据：与cpu核心数目相同。
    * 文件数据：与block块数相同
  * 默认读入数据的分区方式partitioner（Key-Value RDD）：None
  * 获取分区数目：`rdd.partitions.size`
  * 设置分区数目：
    * `rdd.repartition(100)`
    * `rdd.coalesce(100)`：**没有shuffle操作**，适用于将多分区变少的情况。如果shuffle = true，效果同上。
  * 获取分区方式partitioner（Key-Value RDD）：`rdd.partitioner`
  * 设置分区类型partitioner（Key-Value RDD）
    * 方法一：`rdd.partitionBy(new spark.HashPartitioner(2))`会返回混洗后的RDD
    * 方法二：一些操作具有可选的Partitioner作为参数，如`join()`、`groupByKey()`，操作返回的RDD具有次Partitioner
  * 设置新的分区方式Partitioner(rdd.partitionBy)，包含了shuffle的步骤，如果想保留效果**需要Cache这个RDD，否则没有意义。**
  * 自定义分区方式：继承`Partitioner`类，实现对应方法
  * 各种操作对分区的影响
    * 添加新的分区类型：
      * 哈希分区：
        * reduceByKey、CombineByKey、groupByKey
        * 二元操作：join、leftOuterJoin、rightOuterJoin、、cogroup(groupWith)
      * 范围分区：sortByKey
      * 添加指定类型：partitionBy
    * **破坏分区类型**：map
    * 保留父RDD的分区方式：filter、mapValues、flatMapValues
    * 对于二元操作：
      * **默认结果是Hash分区，分区数与操作的并行度相同。**
      * 如果一个父RDD设置过Partitioner，那么结果会采用它的分区方式
      * 如果两个RDD都设置了Partitioner，那么结果使用第一个的分区方式。
  * 分区的应用
    * 增加速度—大的数据集需要多次与不同的小数据集join的情况，**对大数据集设置Partitioner，再Join，使得shuffle中只传递小的数据集。**
    * 使用mapValues代替map，防止丢失Partitioner信息
* Cache问题
  * 默认情况下，Spark不会在内存中保留数据，第二次使用RDD时（如一个新的Aciton操作涉及到了以前的RDD），会重新计算这个RDD。对于计算RDD很费资源的情况，这是十分低效。
  * 使用cache()或者persist(MEMORY_ONLY)缓存到内存：默认情况下，RDD对象会使**用非序列化的格式直接缓存对象**。优点是下次使用时很快。缺点是比较浪费空间。**如果内存无法容纳，则会部分partition会不被缓存**。
  * persist的参数可选的有以序列化的方式缓存到内存/只缓存到disk/内存和disk混合使用，[参考这里](http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence)：对于十分耗时的RDD的运算，建议内存disk混合缓存，确保所有内存均被缓存。
  * 有些情况下可能缓存失效（如机器问题），可以缓存多份。在存储级别后面加上'_2'
  * unpersist可以取消缓存。不调用这个方法默认会以LRU的方式处理。

#### 注意点

1. 回调函数引用问题：RDD的操作函数中有大量的传入回调函数的情况（如map），由于回调函数会发送到Executor中去运行，所以需要注意回调函数的引用外层对象`this`的问题，会出现`NotSerializableException`的错误。参考[文档](http://spark.apache.org/docs/latest/programming-guide.html#passing-functions-to-spark)，总结一下，推荐使用下面两种形式：
   * 使用匿名回调函数作为参数
   * 使用object（全局对象）中的函数作为参数
   * 注意：**函数中如果引用类的成员变量，请先copy到层本地变量**，如：` val field_ = this.field`
2. map与mapValues对分区的影响
3. **使用rdd.coalesce的一个坑**：如果我们的代码类似rdd.map(xxx)....coalesce(1).write...这样的操作吧结果输出到一个文件中，活导致先合并在执行map操作的问题。这是由于spark的**stage合并**机制导致的，[参考文章](https://qnalist.com/questions/4981936/partitions-coalesce-and-parallelism).
4. 如同类型的RDD之间的隐式转化：在spark1.3.0之后版本，无需使用`import org.apache.spark.SparkContext._`，RDD会自动转换转换对应XXRDDFunctions。



### I/O方式

讨论一下Spark读入文件的方式。一般我们需要的处理的数据都是位于某个文件系统上的某个（些）文件。Spark可以访问下面这些数据。

* 文件格式
  * 文本文件
    * read:`sc.textFile("path")`
    * write:`sc.saveAsTextFile("path")`
  * json/逗号制表符分割的文本(半结构化)等特殊的文本格式
    * read：textFile读入数据，然后使用json/csv库在mapPartition中解析出对象
    * write：类似上面，用json/csv库解析成字符串，然后saveAsTextFile。
  * **SequenceFile结构化Key-Value文件**：要求key、value都是Writable对象
    * read：`sc.sequenceFile[Key,Value]("path",minPartitions)`。获得key-value RDD（这里自动把Writable对象转换成Scala对象）。等价于`sc.sequenceFile("path",classOf[Text],classOf[IntWritable]).map{case(x,y)=>(x.toString,y.get())}`
    * write：`data.saveAsSequenceFile("path")`。注意data是个key-value的RDD
  * 对象文件：不一定要是key-value。可以是简单的值RDD。**一般用于job之间的通讯？**
    * read：sc.objectFile()
    * write：sc.saveAsObjectFile
* 文件系统
  * 本地文件系统、nfs：`file:///home/name/xxx.txt`
  * HDFS：`hdfs://master:port/path`
  * Amazon S3：`s3n://bucket/….`
* 非文件系统存储系统/数据库
  * JDBC数据库 
  * HBase：
    * 使用`hadoopRDD`/`saveAsHadoopDataSet` (`newAPIHadoopRDD`/`saveAsnewAPIHadoopDataSet`)来读取，与**hadoopFile区别是没有path这个参数**
  * MongoDB\Elasticsearch
* SparkSQL读取结构化数据源（带有Schema，或者可以推断出来）：参考下一节
* 支持使用Hadoop `InputFormat`和`OutputFormat`接口访问的所有数据（如HDFS，S3，HBase）。这种方式比较原始，不推荐直接使用，推荐用更高层的API。
  * 注意点：
    * hadoop的`Inputformat`必须是键值对（key-value）的形式，**如果没有key，则使用假键null**。
    * 有新旧两套API
      * 旧的`sc.hadoopFile[Text,Text,xxInputFormat](inputfile)`
      * 新的`sc.newAPIHadoopFile(intputfile,classOf[xxInputFormat],classOf[Text],classOf[Text],conf)`
  * read：`sc.newAPIHadoopFile`、`sc.newAPIHadoopFile`
  * write：`data.saveAsHadoopFile()`、`data.saveAsNewAPIHadoopFile()`
  * 应用，
    * 读取pb数据 — 《Spark快速大数据分析》p75
    * 文件压缩 — 《Spark快速大数据分析》p74 p77常见的文件压缩格式（注意**是否可分割**）

> 一个Q&A：**哪些接口是基于原始接口（hadoopfile）构造出来的高层接口?**

### 共享变量

#### 累加器

* 理解：由于Spark的运行原理，变量只具有局部性。累加器提供了一个全局变量的功能。
* 使用
  * `val acc=SparkContext.accumulator(0)`初始化，参数的类型自动推断累加器的类型，这里是`Int`
  * 设置 `acc+=1`
  * 在driver中读取：`acc.value`
* 自带的累加器：`Int`,`Double`,`Long`,`Float`
* 自定义累加器：扩展AccumulatorParam
* 应用场景
  * 统计读入数据的错误行数。
* 注意点
  * **如果有节点失败，在Transfer操作里面的累加器可能重复增加**，但是实际应用中依据业务需求这个问题不严重。
  * cache也会对Transfer操作的累加的结果有影响
  * **可靠的作法是在Action操作中使用累加器。如foreach。**
  * ​

#### 广播变量

* 理解：由于Spark会分发任务执行，如果我们引用某个变量（只读），那么**每次执行操作时都会下发这个变量的值**。某些变量很大的情况下（如查询表），重复分发浪费带宽，可以用广播变量。优点是它之后分发一次，以后Executor节点可以直接只用。
* 使用：
  * `val bd=SparkContext.broadcast(variable)`
  * 使用时，`bd.value`
* 应用场景
  * 查找表
  * 机器学习总大的特征向量
* 注意
  * **一定要是只读常量**，对它的修改不会重新分发。
  * 一般几百KB到几MB的变量使用广播，不要滥用。
  * 如果广播的变量比较大，**对性能有一些影响，体现在变量序列化的时间上**，推荐使用Kryo方式序列号。



## Spark SQL

### 理解RDD DataFrame DataSet几种API的本质区别

* RDD--需要序列化（Java/ Kryo序列化），因此，即使只查询1列数据也要序列化整个对象（代表一行所有列）。而且所有数据都要存放在堆内存中，因此性能差。而且序列化的obj占用对象很大，每个对象都要存储数据+结构。（对比schema的方式）。优点是代码好写，支持编译时的检查。
* DataFrame--存储使用schema+数据的方式。数据还可以列式存储增强性能。而且可以在javaheap外操作。（与java对象无关）。缺点也很明显，代码不好写，只能运行时检测（和sql类似）
* DataSet，两者优点结合，但是需要一个Encoder。注意：虽然Encoder的作用也是把对象转换成bytes，但是**与Java/ Kryo序列化不同，和Encoder的优势是：**可以不反序列化成Object的情况下进行一些常规操作，如排序，过滤，hash。
* 推荐文章
  * [RDD、DataFrame和DataSet的区别](http://www.jianshu.com/p/c0181667daa0)：Dataset内部一些优化
  * 官网介绍：[Introducing Apache Spark Datasets](https://databricks.com/blog/2016/01/04/introducing-apache-spark-datasets.html)

### 核心API

* 入口

  * Spark1.6：sqlContext/HiveContext(带有Hive功能)
  * Spark2.0：sparkSession

* 创建DF/DataSet

  * 从RDD创建

    * **case class/Product类方式，隐式转换**。注意**必须使用**`import spark.implicits._` 再调用`rdd.toDF()`。需要注意的是，一般类型，或者**嵌套`Array`/`Seq`都是可以推断出来**的。

      > Case classes in Scala 2.10 can support only up to 22 fields. To work around this limit,you can use custom classes that implement the Product interface

    * 显式指定StructType的方式。参考API文档，比较复杂的情况下使用。RDD—>Row的RDD—>`spark.createDataFrame(rowRDD, schema)`

  * IO方式：读取的是DataFrame（即DataSet[Row]）参见下一节

  * 内存数据如Seq：`Seq(Person("Andy", 32)).toDS()`

  * DataFrame—>DataSet

    * Map方式：把**Row对象**转换成对应的对象
    * IO方式读取数据源时+as：`val peopleDS = spark.read.json(path).as[Person]`

  * 自定义Encoder：`implicit val mapEncoder = org.apache.spark.sql.Encoders.kryo[Map[String, Any]]`

* RDD类似API

  * map
  * filter
  * join

* 表相关的api

  * select
    * 使用sql语句：`sqlContext.sql("select count(name) as name_cnt from one_table")`
    * 使用select函数：`df.select("count(name) as name_cnt")`
    * 使用批量的select执行函数·:`val rrs = df.selectExpr(select_expr_list: _*)`。其中select_expr_list代表多个语句的list。如`val list=List("count(name) as name_cnt","xxudf(col_name) as xxflag")`
  * drop
  * groupBy&& agg聚合函数
  * withColumn
    * 添加常量列

* sql查询API

  * 新建表：`df.createOrReplaceTempView("table_name")`或者就API`df.registerTempTable("table_name")`
  * 执行查询 `sqlContext.sql("sql expression")`

* UDF：用户自定义函数，他的param是列Column，return是新Colomn的内容。写复杂逻辑代码用的很多！

  * 定义&&使用UDF的方式一：
    * `val my_udf=udf[String,String]( x => x+"end" )`:使用udf函数定义（该函数在function中）
    * `df.withColume(my_udf(col("col_1")))`
  * 定义&&使用UDF的方式二：
    * `spark.udf.register("my_udf",x=>x+"end")`
    * spark.sql("select my_udf('col_1') from table ")在sql中使用

* `org.apache.spark.sql.functions`的built-in函数：一些帮助方法，主要与生成新的Col有关，**这些函数大大方便了开发**。

  * 功能
    * 构建Column
      * `col('name')`：使用列名没返回Column对象
      * `lit(num/String)`:生成常量列
    * 构建UDF：`udf(f: FunctionN[...])`
    * agg函数
      * count/distinctCount
      * min/max
      * sum/mean
      * first/last
    * normal 函数（理解成内置的UDF）
      * String相关：split(col)/upper(col)/lower(col)
      * 数值相关：sin(col)/cos(col)/sqrt(col)/rand()等等
      * 表达式解析：expr(sql)。依据sql语句执行生成新的col `df.filter(expr("token = 'hello'"))`
    * 排序相关，与df.sort函数配合使用：
      * asc(col)
      * desc(col)
    * window函数:????
      * lag
      * lead
      * rank
  * 使用
    * 使用`import org.apache.spark.sql.functions._`导入
  * [参考文档](https://jaceklaskowski.gitbooks.io/mastering-apache-spark/content/spark-sql-functions.html)，[API文档](https://spark.apache.org/docs/1.6.2/api/java/org/apache/spark/sql/functions.html)

* DF的Map操作问题！！！

  * 旧版本DataFrame时，map后又变成了RDD

  * 新版由于DataFrame本质就是DataSet，map操作就是DataSet的map。它的返回也是一个DataSet，因此要求map的函数返回值必须是可以序列化的！！或者自定义encoder

    ```scala
    implicit val mapEncoder = org.apache.spark.sql.Encoders.kryo[Map[String, Any]]
    teenagersDF.map(teenager => teenager.getValuesMap[Any](List("name", "age")))
    ```

* RDD to DataSet/DF ：

  * 使用Product，`import spark.implicits._`

### I/O方式

* 最大不同，以Spark SQL的方式读取数据**不是全表扫描，而是按需读取需要字段**。（与SparkContext.hadoopFile不同）
* 文件格式：必须是**结构化**的数据源，
  * parquet（有schema），一直流行的**嵌套**式**列式**存储格式
  * hive表，
  * json（可推断），
  * jdbc（关系型有表结构）
* API
  * 读
    * `spark.read.format("json").load(path)`  其他格式："parquet"
    * `spark.read.json(path)`或者`spark.read.parquet(path)`
  * 写
    * `df.write.format("json").save(path)` 其他格式："parquet"
    * `df.write.json(path)`或者`df.write.parquet(path)`
  * Hive
    * 配置通过hive-site.xml，并构造HiveContext，可以直接`hiveContext.sql("sql")`
    * 读写通过sql语句实现
  * 旧版本（1.4以前）
    * `sqlContext.parquetFile`, `sqlContext.jsonFile`
* 性能
  * 内存式的列式存储
  * 谓词下堆：不需要读取整个数据集，根据值的特点选择性读取（如数值范围）
* 注意点：
  * json读取方式的性能问题，由于json的格式推断，导致**读取大量数据非常非常慢**！因为要全局推断它的schema。

### 常见需求最佳实践

1. 活用UDF：对于一些自定义要求很高的需求，我们最常见的思路是：使用UDF解决。通常配合下面的函数

   * **withColumn方法**：新增一列，可以使用udf对多个列进行计算，然后生成新列。`df.withColumn("new_col",my_udf(col("col1"),col(col2)))`
   * select方法：提取某些列，然后使用udf计算，**不保留原来的表**。 `df.select(my_udf(col("col1"),col("col2")) as "my_name", col("other_col"))`
   * **filter方法**：比较复杂的过滤。`df.filter(my_udf(col("col1"),col("col2")))`。这里的my_udf返回True/False
   * 注意：**groupBy之后暂时还不能用UDF，应当使用UDAF。**

2. 单表group操作：应当注意的是**表的GroupBy操作，返回的是GroupedData对象，不是DataFrame**。后面必须接上聚合操作才能返回聚合后的DataFrame：

   * 常见操作，只计算一个值。df.groupBy("col1","col2").count/max/min/sum/mean()
   * 复杂些的，计算多个值：df.groupBy("col1","col2").agg()
     * `df.groupBy("col1","col2").agg(max("age"),sum("expense"))`
     * `df.groupBy("col1","col2").agg(Map("age"-> "max", "expense" ->"sum"))`
   * 自定义agg操作：UDAF，新功能。

3. 多表join（filter in another table）：`df1.join(df2,condition_express,"type_string")`

   * join的三种方式。参考[Beyond traditional join with Apache Spark](http://kirillpavlov.com/blog/2016/04/23/beyond-traditional-join-with-apache-spark/)
     * 默认只包含公共的key:inner
     * left_outer/right_outer
     * fullouter
   * join之后的列名字
     * 先重命名withColumnRenamed，再join
     * 直接显示指定df1("col")==df2("col")
   * 空值问题==>DataFrame的na方法
   * 多列相等的join

   ```scala
   // 显示指定col	 
   val res = df1.join(df2,df1("uid") === df2
         ("uid") && df1("time") === df2
         ("time"),"left_outer").
         na.fill(Map(
         "click_cnt" ->0
       )).drop(df2("uid"))
     
   ```

4. 寻找不在另外一个表中的行(filter not in another table)

   * `df1.join( df2.select($"id".alias("id_")), $"id" === $"id_", "leftouter") .where($"id_".isNull).drop("id_")`
   * 如果另外一个集合比较小，可以考虑用filter配合UDF过滤set(可能还需要broadcast)

5. 空值处理：DataFrame的na方法

   * fill
   * drop
   * replace

6. 使用lit方法生成常量列，方便统计一些值。

7. 同时Group多组col，并统计行数（count操作）

   * 问题：使用groupby方法只能对一组col进行group操作，如'name'&&'age'，如果要同时对'name'&&'sex'进行group，这需要再对df进行一次group。如何对df进行一次操作就可以完成多组的group并输出结果呢？

## Spark Streaming

简单介绍一下，以后详细介绍

* mini-batch的模式
* 适合1s以及以上的实时性
* api与批量具有相似性

## 其他

- 调用其他语言程序—pipe 《Spark快速大数据分析》p97
- 安全问题

## 性能优化的基本技巧

* partitioner优化，使用mapValues代替map，保留分区信息。
  * RDD中使用partitionBy方法方法分区，然后在尽量不破坏分区的情况下（如：不使用map），可以获得性能收益。典型的情况是**需要多次join，且数据倾斜**。对数据量很大的RDD执行HashPartition操作，之后join会不会再发送大量数据移动，只会移动小的表到已经Partition好的RDD上。
  * DataFrame中的应用，参考这个[stackoverflow提问](http://stackoverflow.com/questions/30995699/how-to-define-partitioning-of-dataframe)。一般情况下，不需要使用这个方法，因为底层[Catalyst Optimizer](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html)会优化。
    * 1.6及其以上版本：`df.repartition($"key")`对某列使用内置的Partition进行**Hash分区**。这是**推荐的方法**。
    * 小于1.6：先对RDD进行partition，再构造df。会保留改RDD的partition方式。
* 使用广播变量，发送较大的table：应用在依据一个table来过滤数据。
* 使用基于分区的操作mapPartition代替map，减少数据库连接、网络连接数目（相对于一条一条处理数据）。此外还有mapPartitionsWithIndex,foreachPartitions
* 选择合适的序列化器：[参考文章](http://spark.apache.org/docs/latest/tuning.html#data-serialization)
* 内存优化：[参考文章](http://spark.apache.org/docs/latest/tuning.html#memory-tuning)
* 了解调度的方式：应用间调度，应用内调度。[参考文章](http://spark.apache.org/docs/latest/job-scheduling.html)


## 总结

至此，简单地介绍了Spark的配置和常用API，用来解决遇到的常见问题。其中**配置部分**我们很少关心，毕竟在业务开发中我们往往使用公司配置好了的Spark环境，但是调优和解决问题往往能从中入手，是理解Spark的重要一环，这个我会在了解Spark的内部原理之后再详细说明配置项的含义。**API编程**层面主要是RDD与DataSet/DataFrame的操作，熟悉常用的function工具与一下场景实践，同时介绍了**Shuffle,Partition和Cache的概念**。最后简单提了一下**Spark Streaming**和**性能优化**。后期，**工程方面**我会在这几个方面深入研究:

* Spark的**基本原理与源代码**，在深层次上理解Spark。
* 解决流式问题的**Spark Streaming**深入探究，包括且不限于流式的常见问题与Spark上的策略。
* 常见的问题：**性能调优**与Spark如何优雅的**工程化**，涉及内存调优，Scala函数式编程等。
* Spark社区新特性与发展方向。
