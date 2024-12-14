---
{"dg-publish":true,"dg-path":"History/fp-scala-part2.md","permalink":"/History/fp-scala-part2/","title":"Scala函数式编程-Part2","tags":["技术","Scala","函数式编程"],"created":"2018-04-02 20:23:42","updated":"2018-04-02 20:23:42"}
---



我们的目标是：掌握实现『函数式库』的技能

## 纯函数式的并行计算
目标：实现无副作用的并行库，例如实现写函数`val outputList=parMap(inputList)(f)`把f同时应用到list的每个元素
> 为什么不使用Thread？
> * Thread的run方法必然会产生副作用
> * Thread是使用的系统线程，占用资源太多
> 为什么不使用ExecutorService？
> * 线程池对线程进行抽象，本质上还是底层线程
> * Future.get的API也不好组合

下面从头开始，一步一步推导出并行计算的抽象 -- 数据类型+函数

### 第一步：首先从简单的例子入手
7.1
思考：如何实现求一组数的和，函数sum 输入：`ints:Seq[Int]` 输出：`Int`

串行方法：`ints.foldLeft(0)(_ + _)`
并行方法：分治法，先把ints分为两半分别递归求和，最后相加

``` scala
def sum(ints:IndexedSeq[Int]):Int = 
  if (ints.size <= 1)
    ints.headOption getOrElse 0
  else {
    val (l,r) = ints.splitAt(ints.length/2)
    sum(l) + sum(r)
  }
```

### 第二步：构造数据类型的抽象
7.11
思考过程如下
1. 我们需要sum(l) 与 sum(r)并行计算
2. sum他们需要包含一个并行计算的结果。。。因此我们需要一个数据类型Par[Int]表示**一个包含Int类型结果的并行计算**.
3. 抽象一下用Par[A]表示一个返回A类型结果的并行计算，**他像一个装有结果的容器**
4. 因此，比如有`存入/构造`和`拿出`两个动作。Par定义相应的方法如下。

```scala
def unit[A](a: => A): Par[A] //注意他的方法参数是lazy
def get[A](a: Par[A]): A
```

> Par[A]具体是什么？这里可以是一个类作为来存放结果的容器（这个更通俗），也可以是函数。都是行得通的，前者参考ParOrign，后者参考Par。但是我们会发现使用函数更加优雅（因为基于函数，后的转换都是函数间的转换），也更符合函数式编程的思想。。

基于此，我们使用上面的数据类型Par[A]来实现sum

```scala
// 变种一：
def sum(ints: IndexedSeq[Int]): Int = 
  if (ints.size <= 1)
    ints.headOption getOrElse 0
  else {
    val (l,r) = ints.splitAt(ints.length/2)
    Pair.get(Pair.unit(sum(l))) + Pair.get(Pair.unit(sum(r)))
  }
  
```

上面能实现并行计算吗？显然不行！！由于`+`运算符的执行顺序导致加号前面必然先运算，在计算后面的。稍加修改：


```scala
// 变种二：
def sum(ints: IndexedSeq[Int]): Int = 
  if (ints.size <= 1)
    ints.headOption getOrElse 0
  else {
    val (l,r) = ints.splitAt(ints.length/2)
    val sumL: Par[Int] = Pair.unit(sum(l)) // 1
    val sumR: Par[Int] = Pair.unit(sum(r)) // 2
    Pair.get(sumL) + Pair.get(sumR) // 3 
  }
  
```

现在，上面能实现并行计算吗？
如果我们**假设：在上式1，2调用unit时就开始后台计算**。那么，在运行到3的时候就并行了！在3处，会先等待sumL计算结果完成，再尝试获取sumR的结果。
现在的问题是：对比变种一和变种二，发现sum方法不是纯函数，因为在3处不符合**引用透明**这个定义！
问题的本质是：`Pair.get`方法当使用Par[A]作为参数时是有副作用的！
因此，是否可以不在这里调用`Pair.get`，**延迟调用`Pair.get`方法**？
怎么实现呢，结论是：为了延迟调用，sum函数返回一个Par[Int]，并且上面3处的代码需要把两个Par[Int]（sumL和sumR）组合成1个Par[Int]。我们需要一个**组合并行计算的函数**：`(Par[Int], Par[Int])=>Par[Int]`，抽象后就是`(Par[A], Par[A])=>Par[A]`

### 第三步：实现组合并行计算的函数
结合之前章节的内容，我们很容易想到，形如`(Par[A], Par[A])=>Par[A]`的函数是`map2`，这里我们就用这个名字吧。sum变为如下的样子

``` scala
def sum(ints: IndexedSeq[Int]): Par[Int] =  // 1 sum返回值变了
  if (ints.size <= 1)
    Par.unit(ints.headOption getOrElse 0) // 由于sum返回签名变了，这里需要unit
  else {
    val (l,r) = ints.splitAt(ints.length/2)
    val sumL: Par[Int] = sum(l) // 2 由于sum的返回签名变了，这里不需要unit了
    val sumR: Par[Int] = sum(r) // 3 同上
    map2(sumL, sumR) (_ + _) // 4
    // map2(sum(l), sum(r)) (_ + _) // 5 思考：这里和上面等价吗？
  }
  
```

重新思考之前的两个问题，
1. 现在2，3，4是并行计算吗？5处呢？是否依赖于之前的假设（即unit时就触发后台计算）
2. 现在是纯函数了吗？是否引用透明，即上面5处和2、3、4等价吗？

上面这两个问题的答案似乎都和map2的实现有关，我们**先假设map2就是一个正常的立即求值的参数的函数**。

问题1：需要我们人肉把ints带入上面的sum递归函数
* 如果2，3处不触发后台计算（本质是递归到unit不触发，map2也不触发），由于Par只是对运算的描述，所以4处也没有触发计算（返回的是Par），因此2，3是否并行，完全**取决于对map2返回值的调用所触发计算的实现**.
* 如果2，3处触发后台计算（本质是递归到unit触发，map2等结果返回也会立刻触发新计算），也不影响并行性

问题2：已经解答，完全等价

我们可以定义map2如下：
```scala
  def map2[A,B,C](ap:Par[A],bp:Par[B])(f:(A,B)=>C):Par[C]={
    // 这里没有用fork，所以f的运算实在调用的线程
    es=>{
      val x=ap(es) // 注意！！这里可能隐藏一个非常隐蔽的bug，如果UnitFuture(f(ap(es).get(),bp(es).get()))，会使得ap，bp的运算失去并行性！！！！！！！！！！！
      val y=bp(es)
      UnitFuture(f(x.get(),y.get())) // 注意这里一个明显的问题，会一直等待a,b的完成，而且我们无法从外部控制
    }
  }

```

思考：map2的参数可以用lazy吗？与上面strict的区别是什么
答：以上面的sum函数为例，使用严格求值时，等map2返回时，就已经完成了参数sum的递归工作，**已经形成了一个完整的调用链**（这个形成的过程遵守参数递归的顺序），相当于返回了一个**组合后的新函数**。这个函数由**已经生成了的**n个函数嵌套组成。
使用lazy求值时，等map2返回，只要我们没对他的返回值（注意：他返回值是一个函数）进行真正的调用，作为参数的sum就不会运行，此时还没有生成完整的调用链，**仅仅相当于生成了一个调用链的头**。随着我们真正调用返回的值（函数），此时调用链一步异步生成，边生成边调用，调用到最底层时，也生成到最底层，等最底层返回，之前的链不断返回与消失。
本质上：两者只是函数生成的开销发生在什么阶段的问题，严格求值的发生在map2调用时，lazy求值发生在对结果的调用时（真正求值时）。
实践：书中可以发现他使用的是**严格求值**的方式

### 第四步：再思考--如何控制并行
可能我们并不希望所有任务都在新的线程中运行，`Par.map2(Par.unit(1),Par.unit(2))(_ + _)`，在之前的语意下，这里unit(1)和unit(2)都会开辟新的线程（至少是逻辑线程），甚至map2也会开辟新的线程（显然上面的代码函数`f`的运行我们没有这么干）。我们希望能**自由控制并发的时机**。这里我们重新设计API：
* 默认情况下所有操作都不会在新的线程
* 如果需要在新线程运行某个Par，需要显式调用fork（也叫做分流）--- `fork(par)`

此时上面的语意就是在当前线程运行1+2。如果需要分流，`Par.map2(Par.fork(Par.unit(1)),Par.fork(Par.unit(2)))(_ + _)`，更进一步，你甚至可以对map2的返回值也分流：`Par.fork(Par.map2(Par.fork(Par.unit(1)),Par.fork(Par.unit(2)))(_ + _))`

至此，我们需要修改之前的`unit`（参数为lazy），为新的版本(参数为strict，因为他无需开辟线程执行，可以立刻执行)，并增加`fork`函数。之前版本的unit可以变为现在的`lazyUnit`

``` scala
def unit[A](a:A):Par[A]= es=> UnitFuture(a)

 def fork[A](a: => Par[A]): Par[A] = // 注意参数是lazy的
    es => es.submit(new Callable[A] {
      def call = {
        val x=a(es).get
        x
      }
    })

def lazyUnit[A](a: => A): Par[A] = fork(unit(a))

```

思考：可以看到上面实现的fork，**并没有正在开启运算**，只是返回了一个函数，真正的调用被延迟了。为什么这么设计呢？理论上这里直接使用全局线程池submit，然后返回一个UnitFuture也可以，这里为了让线程池的选择更灵活。（参考树p84）

最后，我们这里还需要一个真正运行计算的函数run

``` scala
  def run[A](es:ExecutorService)(x:Par[A]):Future[A] ={ // 返回future时已经开始了计算。对future.get是获得计算结果
    x(es)
  }
  
```

一个完整的API设计思路就完成了！下面我们会在此之上完善一些API，并**再次进行抽象**，获得一些法则。
