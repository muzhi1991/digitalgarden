---
{"dg-publish":true,"permalink":"/003-Publish/History/fp-scala-part1/","title":"Scala函数式编程-Part1","tags":["技术","Scala","函数式编程"],"created":"2024-12-11T18:09:35.707+08:00","updated":"2024-12-11T18:09:35.708+08:00"}
---


## Overview
第一部分主要使用Scala来函数式编程，介绍了核心概念。
我们的目标是追求**使用『纯函数』编程**。

* 学会写小的**函数式程序**。（能写一个程序中的小模块）-- 第一章
* 函数式编程中的**基本概念**以及**Scala中的实现**：高阶函数，多态函数 -- 第二章
* 使用函数式编程实现**函数式数据结构** -- 第三章
* 使用函数式编程进行**错误处理** -- 第四章
* 提升函数式编程运行效率：**非严格求值**的概念 -- 第五章
* 使用函数式编程实现**状态处理**（命令式编程） -- 第六章

[我的练习题](https://github.com/muzhi1991/learn-fp-in-scala)

> 背景知识：Scala是在JVM上运行的一门语言，是带有**自动类型推断**的**静态类型**语言。使用它即可以使用与Java类似的命令式编程（甚至兼容Java库），也可实现纯的函数式编程。

## 什么是函数式编程
核心理解：通过对比命令式编程与函数式编程，来体会函数式编程的优点。
场景：实现一个使用咖啡厅付款的程序，实现buyCoffee的功能，输入信用卡，我们对他扣款，最后获得一杯咖啡

```scala
// 命令式实现
class Cafe {
  def buyCoffee(cc: CreditCard): Coffee = {
    val cup = new Coffee()
    cc.charge(cup.price) // 这里发生了状态变化
    cup
  }
}
// 或者高级一点：使用一个单独的类来支付
class Cafe {
  def buyCoffee(cc: CreditCard, p: Payments): Coffee = {
    val cup = new Coffee()
    p.charge(cc,cup.price)  // 通过p来发起支付，这里也发生了状态变化
    cup
  }
}
```

问题是
* 很难测试：没法判断购买逻辑的准确性，尤其涉及支付，外部payment的状态改变会影响我们buyCoffee运行结果。我们可能需要一个Mock的Payment来做。
* 很难复用：如果要实现一个函数`buyCoffees(n)`购买多杯咖啡呢？可以直接调用N次buyCoffee，问题是信用卡付款需要每一笔都支付交易费，我们理想情况是：合并这个12个账单，一次支付。这就没法复用buyCoffee方法了，要重新实现。

函数式的方法如下：

```scala

// 函数式实现
class Cafe {
  // 没有副作用
  def buyCoffee(cc: CreditCard): (Coffee, Charge) = {
    val cup = new Coffee()
    (cup,Charge(cc,cup.price)) // 没有实际扣款，我们只是把状态变化这个行为封装后输出了。
  }
}
```

把支付这个『副作用』推迟。本质上，支付是引起了外部某个状态的变化，我们这种**状态的变化**通过返回值传递出去（把状态变化抽象了）。
* 易于测试：这个函数随便调用，只要输入相同，输出就相同
* 易于复用：同样实现上面的需求，代码如下：

```scala
// charge定义
case class Charge(cc: CreditCard, amount: Double){
  // 合并付款的函数
  def combine(other: Charge): Charge = 
    if (cc == other.cc)
      Charge(cc, amount + other.amount)
    else
      throw new Exception("can't combine chareges to different cards")
}

class Cafe {
  // 买多杯咖啡
  def buyCoffees(cc: CreditCard,n: Int): (List[Coffee], Charge) = {
    val purchases: List[(Coffee, Charge)] = List.fill(n)(buyCoffee(cc))
    val (coffees, charges) = purchases.unzip
    // 合并了付款，依然没有副作用
    (coffees,charges.reduce((c1, c2) => c1.combine(c2))) 
  }
  // 更复杂的需求：比如买了N被咖啡，用了多个不同信用卡，合并相同信用卡的付款。
  def coalesce(charges: List[Charge]): List[Charge]= {
    charges.groupBy(_.cc).values.map(_.reduce(_ combine _)).toList
  }
}
```

上面我们所有操作的目标就是吧副作用去除，这样写出的函数可以叫做『纯函数』
由此，我们很容易得到纯函数的几个特点
* 引用透明：任何引用透明的表达式（或者函数）都可以被他的结果取代
* 替代模型：利用『引用透明』来推导程序求值的过程就是替代模型
* 纯粹度：表示某个计算是不是纯粹的局部影响。（先忽略这个概念）

> 纯函数的定义：如果一个**函数的参数**是引用透明且**函数调用**也是引用透明，那么他就纯函数

纯函数的优点：
* 模块化、可组合、可复用
* 易于测试

## 函数式编程
核心理解：
* 使用函数式的方法实现Loop
* 尾递归：什么情况下是尾递归，尾递归的好处，scala的实现（注解）
* 高阶函数的概念，应用场景，
* 匿名函数的作用：常作为高阶函数的参数。
* 泛型函数的作用，以及一个问题&&解决：必须使用高阶函数。
* 泛型函数+高阶函数的典型应用：部分应用函数、柯里化、Compose


### scala的程序的基本组成
* 模块：我们可以任务scala中的object就是一个模块，广义的说，对象（class的实例）可以看作模块
* 函数：无副作用的方法，就是我们需要写的函数式程序。就是def定义。
* 过程：有副作用的方法，如main，典型的**返回值Unit的方法暗示了有副作用**。

> 一般当我们说模块时，就是指的`object xxx {}`定义的单例对象，我们`import xxx`表示导入这个module。在shell中load包含object的文件，系统会提示`defined module xxx`

* scala的代码必须在object或者class中
* main只能在object中

> 此外，我们也会`import XXX` 其中XXX是类`class XXX{}`。这表示引入类，他不是模块！。在shell中load包含class的文件，系统会提示`defined class XXX`

### 模块 && 对象 && 命名空间
理解这三个概念是一个东西，
* 一个对象（object）就是一个模块（module）
* 一个模块/对象给成员（函数，过程，值）提供了命名空间

### 高阶函数
核心实现：函数像变量一样，可以再赋值给其他变量，存储在数据结构，当做参数传递
用乘积的例子，分别用[普通递归](https://github.com/muzhi1991/learn-fp-in-scala/blob/84199b807cf1a1363efbc3b814a15623c1cdf9b9/src/chapter2/HOF.scala#L12-L18)和[尾递归](https://github.com/muzhi1991/learn-fp-in-scala/blob/84199b807cf1a1363efbc3b814a15623c1cdf9b9/src/chapter2/HOF.scala#L21-L38)实现。其中尾递归是一种Loop的方式。

* 使用高阶函数实现Loop
* Loop与尾递归的关系
* 尾递归`@annotation.tailrec`的使用
* 匿名函数作为高阶函数参数

> 一个理解『函数也是值』的角度是：Scala中定义匿名函数（也叫**函数字面量**）：`(x:Int) => x==9`本质上是定义了一个`Function2[Int, Int, Boolean]`的对象。
>```scala
>val lessThan = new Function2[Int, Int, Boolean] {
>  def apply(a: Int, b: Int) = a < b
>}
>```
> **高阶函数的参数**的命名约定：f g h来命名函数
> 匿名函数作为高阶函数的参数时，匿名函数的参数的类型如果可以自动推到出，即可省略(a,b)=>a<b

#### 练习
* 实现斐波那契数列（使用尾递归范式），[参考](https://github.com/muzhi1991/learn-fp-in-scala/blob/84199b807cf1a1363efbc3b814a15623c1cdf9b9/src/chapter2/HOF.scala#L40-L71)

### 多态函数（泛型函数）
* 单态 vs 多态
* [多态的类型](https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%9E%8B_(%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91%E5%AD%A6))
  * 静态多态
    * **参数化多态**
    * 非参数化多态/特设多态（Ad-hoc polymorphism）
  * 动态多态
    * 子类型多态
* **使用泛型不可避免地使用高阶函数**：参考下面的参数p，因为A没有类型信息，需要高阶高阶函数来判断状态。

```scala
// 一个多态的例子
def findFirst[A](as: Array[A], p: A=>Boolean):Int={
  @annotation.tailrec
  def loop(n: Int): Int={
    if (n >= as.length) -1
    else if (p(as(n))) n
    else loop(n+1)
  }
  loop(0)
}
```

> 习惯上，用大写字母[A,B,C]表示泛型参数

### 通过类型实现多态
下面是一下常见的函数变换：
* [部分应用](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter2/PolymorphicByType.scala#L8-L15)：`(A, B) => C` 变成 `B => C`
* [柯里化、反柯里化](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter2/PolymorphicByType.scala#L17-L36)：`(A, B) => C` 变成 `A => (B => C)`。反柯里化则相反
* [Compose](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter2/PolymorphicByType.scala#L39-L48)：`f:B=>C` `g:A=>B` 变成 `A=>C`。功能是：改变了函数的输入或者输出
  * 改变了函数f的输入（最常用的视角），在Scala中 `f.compose(g)`就是这个功能
  * 改变了函数g的输出
  * 多态高阶函数：在高阶函数中加上了多态
  
> 由于=>是右结合的操作符，A => B => C 与 A => (B => C) 等价。含义是输入是一个值，输出是一个函数



### 其他技巧
* `scalac xxx.scala` 编译scala生成class文件
* 在shell中，使用`:load xx.scala`加载中定义的模块
* 使用`javap xxx.class` 反编译,scala生成的java代码
  
> 函数与过程：我们一般用『过程』表示有副作用的方法，用函数表示没有副作用的方法

## 函数式数据结构
核心理解：
* 函数式List的数据结构定义：
  * Cons && Nil 这种惯用法
  * trait和class的继承设计
  * 可以对比后边的Option，Stream，State的设计
* 函数式数据结构的优点：
  * 最大优点：不可变性
  * 效率问题：什么是数据共享--》减少复制
* List的实现：
  * 关键技术：**模式匹配**--函数式编程中核心概念
  * 最佳实践：改进高阶函数的**类型推导**
  * 简单函数的实现：有了模式匹配和上面的最佳实践，实现普通的List[Int]的sum没有问题
  * 关键函数（泛型+高阶函数）：
    * 递归概念的泛化：foldRight
      * 输入List输出一个『值』
      * 可以使用该函数代替递归
    * foldLeft的实现
    * 难点理解：使用foldRight实现foldLeft
      * 结合了函数作为初始值
      * 理解调用链
    * foldRight对比Stream实现的foldRight
* 代数数据类型ADT的理解
  * 计算代数数据类型的个数
  * 从模式匹配的角度理解
* Tree：如何设计一个函数数据结构中的函数，从具体到抽象。

> 本书第一部分中，除了List的定义的函数都是在外部的伴生对象里以独立函数的形式提供，其他的结构（Option，Stream，State）都尽量把相关方法放在了接口中。（List没有特殊的原因，只是示例，一般都应该在接口里定义）

### 定义函数式数据结构：List
实现一个List:
* sealed Trait List[+A]
* Nil定义
* Cons定义：一个经典递归结构，可以实现链表，树等常见函数式数据结构 
* 型变--协变
* 伴生对象--apply--函数可变参数

涉及的技术细节如下

#### scala定义class/trait
* 空trait、class：可以直接 `trait xxx` `class xxx` 定义没有任何方法的空trait和class，一般作为基类。
* class和object都可以extend trait或者baseClass
* 具有多个trait时，`class test extend trait1 with trait2 with trait3`
* trait支持在里面实现函数，定义val

#### Nothing的使用
参考[这篇](https://www.cnblogs.com/moonandstar08/p/5759137.html)，对比了Scala常见的『空』
* Nothing是一个Trait,Nothing是没有实例的
* Nothing是所有类的子类
* 所以**它可以赋值给任何类型**，配合协变，可以赋值给人以容器 --> `List[String] = List[Nothing]()`
* Nothing比较适合用来定义基类容器--->Nil就是List[Nothing]，那么Nil就可以当做是一个空的String List，空的Int List，甚至使Any List。

#### `case object` vs `case class`
  * 他们一个是object（对象），另一个是类
  * 使用case（语法糖）使得两者自动有了一下方法
    * 主要目的是支持模式匹配
    * 共同的方法：toString、hashCode、copy、equals方法 
    * case class多了 伴生object，且有apply方法和unapply实现参数case匹配
  * 形式上，**class可以有构造参数**，object由于是一个对象没有参数一说。
  * 总结，虽然功能上`case object`基本可以被`case class`代替。但是实践的时候，**当我们需要使用case类时（需要支持模式匹配功能/序列化时往往需要case类），如果case类有参数时我们选择`case class`，当case类没有参数时选择 `case object`**。
    * 原因：在没有参数时，使用`case object`的原因是，对于没参数的class，由于函数式编程要求immutable。**所有该类是一个常量，只要一个单例共享数据**就行！！
  * 参考文章：https://www.quora.com/Whats-the-difference-between-case-class-and-case-object-in-Scala

#### 用Python实现链表 
https://python.freelycode.com/contribution/detail/1021
同样是cons结构

### List的各种函数的实现

#### 基本知识：模式匹配
定义：**可以侵入到表达式的数据结构内部，对这个结构进行检验和提取子表达式。**
    
#### 数据共享
List中减少赋值，对比tail与init（如何实现高效的init，需要改变数据结构，参考vector，使用了trie）  

#### 类型推导的改进
* 为了使用匿名函数函数作为参数时的自动类型推导，我们需要柯里化（Scala的缺点导致的）
* 类型推导时，z是子类可能需要显示声明父类以至此后面的推导：z:List[Int]（参考foldRight）

#### 基本函数的实现
实现基本函数sum，product

#### 泛化高阶函数的实现与应用
重要实现：
* [**foldRight**和foldLeft](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter3/List.scala#L137-L171)（重要） && 相关练习应用，如：[append](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter3/List.scala#L319-L321)
  * 难点：
      * [使用foldRight实现foldLeft](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter3/List.scala#L214-L296)（参考习题，涉及使用函数对象作为初始值z）
思考（题3.7）：foldRight实现product时如何短路？如何优化product在乘以0时的效率：提前结束计算（或者用Stream的lazy来做）
* [filter的实现](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter3/List.scala#L393-L414)
* [map的实现](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter3/List.scala#L368-L391)（重要）--思考：foldRight与List结构的关系
* [flatmap 的实现](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter3/List.scala#L416-L430)（重要）:foldRight+append。对比Stream（Option，State不太一样）

其他实现
[zipWith](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter3/List.scala#L467-L473)

组合应用:
* 链式调用List[A]：map map => List[C]
* 嵌套一层的List[List[A]：flatmap(map) ==> List[B]
* 嵌套两层的List[List[List[A]]：flatmap (flatmap (map)) ==> List[B]

> flatmap后面的map在这里用链式调用也行。之所以嵌套调用，是为了和后面Option等应用兼容

#### 效率问题
（题3.24）
[hasSubsequence](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter3/List.scala#L475-L502)：使用if语句显式提前终止递归。写法不太『优美』，Stream有更好的写法，效率和提前终止类似

### 标准库中的List
* 使用`::`代替`Cons`
* `::`使用`:`结尾的操作符是有关联，机 `1::Nil`表示调用Nil的`::`方法，该方法内部调用`::(1,Nil）`
* `case Cons(h,t)`变为`case h::t`   

### 树的数据结构
* ADT的定义：代数数据类型，参考[文章](https://www.cnblogs.com/moonandstar08/p/5759137.html)，理解sum（对应类的继承，option），product类型（对应tuple，record）
* 书中[Tree的实现](https://github.com/muzhi1991/learn-fp-in-scala/blob/master/src/chapter3/Tree.scala)
  * 只有叶子结点Leaf存储数据
  * 分支节点Branch存在左右子树
* 相关函数：
  * 具体函数：size(所有节点数量），maximum，depth，map
  * **抽象函数**：fold的实现（对比List的FoldRight抽象）-- 总结具体函数，有具体到抽象

### 遗留问题
1. 3.7 foldRight实现product时如何短路（第五章解答:使用Stream的foldRight）
2. 3.24 如何高效实现hasSubsequence（第五章：使用Stream的tails方法）

## 不用异常来错误处理
核心理解：
* 知道为什么要用Option这个函数式的方式处理异常，而不是Java的抛出异常
* Option函数数据结构的定义：**只包含一个元素的List**
* Option函数定义 
  * 常见函数 map/flatmap
  * 特有函数 getOrElse\orElse
* 如何获得Option
  * lift
  * Try
* Option转化为Java异常
* 常见pattern：
  * 1个输入，1个值输出链式处理：`func().map.getOrElse` ，其中func()返回一个Option
  * 2个输入，1个值输出：`map2(op1,op2)(f)`，只要一个None，结果就是None
  * n个输入，1个值输出：只要一个None，结果就是None
    *  **for推导** 或者 flatmap[*n] + map 
  * n个输入，n个输出（一个List输出）
    * sequence：List[Option[A]===>Option[List[A]]
    * traverse：List[Option[A]===>Option[List[B]]
* Either：保留了具体异常的Option（只能存一个异常）
  * 数据结构定义：理解互斥并集
  * 获得Either的方法与Option类似类似
  * 具有和Option相同的函数和用法
* 可以存储多个异常的高级版本：Partial

### 抛出异常的问题
参考示例4.1的例子，很容易发现下面的问题
抛出异常的缺点：
* **不是引用透明的！！**
* 不是类型安全的，由于没有在函数定义时声明，我们不知道是否内部会有异常。

### 除了抛出异常，还可以怎么做?
我们还可以使用类似于C语言的方法：**返回一个特定的普通值（或者默认值）来表示异常**。
缺点也很明显
* 易错，这个很容易被我们忽略，忘记处理。
* 容易产生模板代码，尤其是有很多个异常要判断时，要重复n次
* 难以描述多态类型。对于多态函数，很难找一个有意义的返回值表示错误。

使用Option方案，也是基于返回普通值的方式，但是解决了上面的问题：
* 易错：使用统一的Option类代表返回值，获得真实值必须get，不会忘记处理
* 模板代码：使用各种Option函数组合，map flatmap有效解决了模板代码
* 难以描述多态类：统一使用None代表错误
* 额外的优点是：可以**推迟并且集中处理异常**

> 科普： 部分函数 VS 完全函数。抛出异常的函数就是部分函数，使用Option就是把部分函数转化为了完全函数。
> 部分函数的定义：对于一些输入没有结果的函数。例如：抛出异常的函数，case语句定义的函数。

### Option数据类型定义 && 常用函数
* Option：只包含一个元素的List
  * 数据结构定义
  * 常见函数 map/flatmap
  * 特有函数 getOrElse\orElse

### Option使用模式分析
Option的使用场景是什么？
使用在**部分函数的返回值**中，典型地
* 可能抛出异常的函数：处理除0，参数判断等
* 可能返回空值的函数：查找数据等

一旦我们把某个方法改造为返回Option的函数如`def func():Option[A]`
我们可以使用`func().map(f).getOrElse` 这种模式处理并获得结果。
如果func()是一个数据库查询函数，
* map函数可以方便地对返回数据二次处理：如提取某些字段
* getOrElse，会返回具体的值/统一设置默认值。如果有需求，也可以模式匹配判断是否是None来错误处理

#### 转化
* 把方法改造为Option
  * lift的概念
    * 可以[lift函数](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter4/Option.scala#L64-L79)：把函数改造为，输入输出都是Option的函数。提升已经存在的任何函数到Option上下文，
    * 可以提升PartialFunction（部分函数）为Option输出
  * Try方法：把基于异常的方法改造成Option
* Option转化为抛出异常：`outputOpt.getOrElse(throw new Exception("fail"))`

#### 错误处理
我们多某个Option对象，可以像处理List那样处理

1. 1个输入，1个值输出链式处理：`func().map.getOrElse` ，其中func()返回一个Option
2. 2个输入，1个值输出：`map2(op1,op2)(f)`，只要一个None，结果就是None
3. n个输入，1个值输出：只要一个None，结果就是None
  *  **for推导** 或者使用 flatmap[*n] + map （两者等价）
4. n个输入，n个输出（一个List输出）
  * sequence：List[Option[A]===>Option[List[A]]，实现方法是list.foldright与map2组合
  * traverse：List[Option[A]===>Option[List[B]]

我们可以把一个带着Option的调用过程（即把上面1/2/3合并）描述为：
从n个Option取出里面的值，来调用函数f(a1,a2...an)。任何一个a为None则结果为None
这都可以用**for推导** 来表示！！！

注意
* map2 map3 .. mapn：map2的实现：flatmap+map
* for推导的实现：flatmap[*n] + map

> 一个常识：实现了map&&flatMap的对象可以支持for推导

### Either
* 数据结构定义 Left（错误） Rihgt（正确）
* Either有相同版本的函数
* Either中只能存储一个Exception，对于上面的链式调用中出现多个Exception，无法存储所有的Exception，此时需要一个新的数据结果，类似的Left存储了Seq[Exception]。参考[Partial](https://github.com/muzhi1991/learn-fp-in-scala/blob/master/src/chapter4/Partial.scala)

## 严格求值与惰性求值
核心目标：实现一个更好的List--Stream（更高效）
核心理解：
* List的存在问题有哪些
* 严格求值与惰性求值的概念&&Scala中的实现
  * Scala中默认都是严格求职
  * 惰性求值多次执行的问题--lazy关键字
* 惰性列表Stream的数据结构定义
  * 传名参数作为类的参数来延迟求值
  * 智能构造器
  * Helper函数的定义：toList，take，方便debug
* Stream常见函数
  * 理解为什么foldRight为什么能延迟执行：对比List中的foldRight
  * 其他函数（对标List）：map flatMap append（注意实现时参数应该是惰性的）
* 理解串联函数时的惰性求值--一等循环（类比一等函数）
  * 优点：效率优化（提前终止）、内存优化（没有中间对象）
* Stream的应用
  * 使用Stream实现无限流
    * 栈安全问题（惰性列表也有栈安全问题！！）
  * 共递归 && unfold
    * 共递归的含义，以及一些专业术语：守护递归、共结束
    * **对比foldRight与unfold的异同**
  * tails方法
    * hasSubsequence的实现优化
    * 泛化方法：scanRight方法

### List的一个问题
观察下面的List的链式使用方法
```scala
List(1,2,3,4).map(_ + 10).filter(_ % 2 == 0).map(_ * 3)
```
有一个明显的效率问题，我们会：
* 遍历List，生成+10的新List2
* 再次遍历List2，执行filter，生成List3
* 最后，再次遍历List3，生成最终结果

然而，一个显而易见的优化方式是，合并这3个步骤成为单个函数。（虽然filter不是很好合并，这里从概念是上感受）我们希望这个步骤能自动完成

### 严格与非严格求职
* 非严格求值是**函数的一个属性**，含义是：这个函数可以选择不对它的一个或多个**参数**求值。
  * 非严格求值只适用于**函数参数**，不能应用再类的参数，为此，我们需要对类的参数用函数表示，变相达到惰性的目的。
  * 任何函数都是默认严格求值的
* 实现非严格求值的方式 
  * 使用高阶函数作为未求值的参数--`def if2[A](cond:Boolean, onTrue: () => A, onFalse: () => A):A`
  * 使用Scala内置语法--`def if2[A](cond:Boolean, onTrue: => A, onFalse: => A):A`
* 惰性问题：每个参数引用的地方都会被求值一次，不会缓存参数求值的结果。
  * 解决：使用`lazy val` 来缓存：lazy表示：Scala会延迟对这个变量的求值 && 缓存这个结果

### 惰性列表Stream

#### 数据结构定义
```scala
sealed trait Stream[+A] 
case object Empty extends Stream[Nothing]
case class Cons[+A](h: () => A, t: () => Stream[A]) extends Stream[A]

object Stream {
 def cons[A](h: => A, t: => Stream[A]): Stream[A] = {
    lazy val h1 = h
    lazy val t1 = t
    Cons(() => h1, () => t1)
  }

  def empty[A]: Stream[A] = Empty

  def apply[A](as: A*): Stream[A] = {
    if (as.isEmpty) empty
    else cons(as.head, apply(as.tail: _*))
  }
}
```
对比List几个重要的不同点：
* Cons的类定义：使用了函数对象作为参数，达到延迟调用的目的（由于=>语法不支持作为类的参数）
* 添加了辅助方法构造Empty和Cons。主要关注cons方法，目的是要是
  * 为了封装**传名参数**给Cons，
  * 同时使用了lazy来缓存变量，确保h/t只运行一次。

#### Helper函数
helper函数是从Stream中获取一部分数据的方法：
* toList：把Stream转化成List，是全量转化
* take(n):Stream[A]：截取一般的Stream，返回的还是Stream，需要再toList获取实际值
* drop(n):Stream[A]：类似take
* takeWhile(p: A=>Boolean):Stream[A]：加入了条件判断
[参考代码](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter5/Stream.scala#L16-L44)

### 描述与求值的分离：Stream关键函数的实现
#### 理解Stream的FoldRight函数
对比Stream的foldRight和List的foldRight，[参考代码](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter5/Stream.scala#L46-L73)

* 求值顺序不同
  核心：由于f的第二个参数B变成了『延迟调用』，导致求值顺序不同，即，
  * 在List版本中会在执行f前先调用参数B部分的代码块
  * 在Stream版本中在先执行f，再『按需』执行参数部分
* 结果相同
  核心：两个方法在结果上是完全等价的，典型的例子是如果用foldRight重新构造一个新的相同的List/Stream。产生的结果相同

应用：可以使用Stream版本的**FoldRight实现『中断』效果**。提高类似exists、forAll函数的效率。一个极端的例子如下（没必要用这个方法实现）：

```scala
  def headOption_foldRight(): Option[A] = {
    foldRight(None: Option[A])((x, _) => Some(x))
  }
```

#### Stream的链式调用&&优点--一等循环
对应的，我们可以**基于foldRight**实现[Stream版本的map，flatMap，append等函数](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter5/Stream.scala#L105-L132)。

这些惰性版本的函数有**一个巨大的优势：一等循环（first-class loops）**。
```scala
Stream(1,2,3,4).map(_ + 10).filter(_ %2 == 0).toList
```
一等循环解决了本章开头的提出的问题：如何自动多个函数。当面执行上述代码时，在`map`调用后立即遍历`Stream`，在`filter`调用后也不会执行，而是在最后`toList`时实际发生操作。你会发现**循环延迟执行**。最终，只在一次循环中完成了多步计算（相当于自动合并了）。

一个有趣的应用：
```scala
  // 虽然filter的含义是过滤整个链表，但是由于是惰性求值，我们只用了head，所以过滤出head之后就不会往后遍历了
  def find(p: A => Boolean): Option[A] = {
    filter(p).headOption()
  }
```

### Stream应用：无限流
因为Stream的惰性（增量）的特点，可以用它来实现一个无限长度的Stream
```scala
    def constant[A](a: A): Stream[A] = {
      lazy val repeat: Stream[A] = Stream.cons(a, repeat)
      repeat
    }
    val ones: Stream[Int] = constant(1)
```

另外一个有趣的应用是里有无限流生成一个[无限斐波那契数列](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter5/Stream.scala#L301-L307)

### Stream应用：共递归 && unfold

unfold是这么一个函数：
* Input:
  * 一个状态转化函数：状态=>(值，新状态)
  * 一个初始状态
* Output:
  * 生成一个Stream

```scala
  def unfold[A, S](z: S)(f: S => Option[(A, S)]): Stream[A] = {
    f(z) match {
      case None => empty[A] // 如果新状态为None，这结束
      case Some((a, s)) => Stream.cons(a, unfold(s)(f))
    }
  }
```

unfold就是共递归函数。
对比『递归』与『共递归』：
* 递归：每一步都指向一个更小的范围，最终结束 --> foldRight
* 共递归：每一步产生数据以及状态，不断生产新的值。（生产数据的能力由新的状态控制）--> unfold

共递归也叫守护递归，生成能力（f控制的的结束）也叫共结束

unfold的应用
* 使用unfold实现无限流
* 当初始状态z为Stream时，可以对该Stream进行遍历-->[实现map、take、takeWhile等函数](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter5/Stream.scala#L144-L163)，以及[zipWith](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter5/Stream.scala#L165-L171)

**对比foldRight和unfold（参考书笔记p65）**


### Stream应用：tails函数实现hasSubsequence

* tails函数：生成Stream的所有后缀。[使用unfold实现](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter5/Stream.scala#L203-L209)
* 用tails实现hasSubsequence
  * 实现[简洁明了](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter5/Stream.scala#L211-L213)。
  * 理解为什么tails实现hasSubsequence是高效的（相比List的tails高效，对比List的hasSubsequence实现效率相同）：惰性
* tails的泛化scanRight，对每一个后缀（tails）执行foldRight，直接用tails+foldRight实现比较低效（要迭代len(tails)次），建议[使用foldRight直接实现](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter5/Stream.scala#L214-L232)

## 纯函数式状态
这章很难理解，慢慢体会。建议状态转化函数（Rand/State）类比Option。
核心：使用函数式编程实现带状态的编程

理解：
* 以随机数生成器为例：随机数生成器的特点
  * 副作用的方式的随机数生成器
  * 纯函数式的随机数生成器
    * 效率问题
* 如何转化待状态的方法==>纯函数式
  * 通用的转化方法
  * 带状态API的例子--其他随机数生成器：生成随机的double，随机的pair等
* 抽象状态：更好的API
  * 随机数的例子：Rand--优点：避免显示传递RNG（对比上面的实现）
    * 传递状态：
      * unit：不修改状态，传递输出值
      * map：不修改状态，只修改输出值
    * 组合状态行为：map2
      * sequence的实现 List[Rand[A]]=>Rand[List[A]]
    * 嵌套状态行为：flatmap---从另一个角度理解：由于返回的是函数，如果是相同的函数，则是递归函数
      * flatmap实现map,map2
  * 通用实现：State
    * 常见参数 map map2 flatmap sequence   
* 纯函数命令编程：无副作用的维护状态
  * flatmap[*n]+map组合==>for推导
  * 使用函数式方法修改任意状态：**modify方法**的定义
    * get
    * set
    

### 副作用版本的随机数生成器
Scala函数库 `scala.until.Random`就是一个带副作用的随机函数
```scala
val rng= new scala.util.Random // 可以用带种子的版本
rng.nextInt 
rng.nextInt // 返回不同的值
```
缺点：
我们很难预测得到的值，就很难测试&&复现bug。因为即使相同Random的种子，我们还必须保证调用`nextInt`的次数相同（伪随机）。本质上，是`Random`内部封装了一个状态，每次我们调用nextInt时，他内部会存储这个状态（副作用）
我补充：
有状态更困难的是在并发的时候，对共享状态的变化不可控！容易导致难以复现bug。

### 纯函数版本的随机数生成器
核心：**显示** 传递状态，把状态和值一起返回。
```scala
trait RNG {
  def nextInt: (Int, RNG)
}
```
这种做法导致我们不行每次使用新的RNG对象来生成随机数。

```scala
val rng = SimpleRNG(22)
val (n1,rng1) = rng.nextInt
val (n2,rng2) = rng1.nextInt // 注意这里用的上面的新的RNG对象rng1
```
其中[SimpleRNG实现](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L8-L20)使用了『线性同余生成器』
优点是：对于某个确定的RNG实例，每次调用都会返回相同的值。

上面我们成功改造了nextInt为『纯函数式』的实现。其实这个改造方法是通用的。
```scala
// 带副作用的版本，s就是共享的状态
class Foo {
  private var s:FooState = ...
  def bar: Bar // 假设，他是一个会改变s的函数
  def baz: Int // 假设，他是一个会改变s的函数
}
// 纯函数的版本
trait Foo {
  def bar: (Bar, Foo) // Foo包含了新状态
  def baz: (Int, Foo) // Foo包含了新状态
}
```

> 一个效率问题：我们返回的状态是一个新的值（[SimpleRNG实现](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L8-L20)），相当于拷贝了一个值。相比与『副作用』的版本，直接修改内存中的状态，比较低效。有两种办法：
>
> * 使用高效的函数式数据结构？
> * **在第四章会以一种『引用透明』的方式修改内存**。

### 用纯函数实现带有状态的API

核心：**实现输入状态，输出（新）状态的函数式API**。

我继续用随机数生成器作为例子，上面我们实现了RNG中的nextInt方法，我建议分下来看：

* 把RNG看出一个纯的状态存储器（什么SimpleRNG存了seed 22）
* 把RNG里面的nextInt方法拿出去，**nextInt改成独立的状态转换函数**：RNG=>(value,RNG)。因为上面nextInt的输入状态其实就是包裹nextInt的RNG对象。

目的：我们接着实现和状态相关的其他状态转换API：**与改造后的nextInt类似，所有API函数都需要接受一个状态，输出一个新状态**，这样我们就可以链式地传递状态了。

> 另一个角度看：都改造成放在RNG类里面的方法也是可以的

我们这里还是**基于之前的nextInt**（在RNG类里面的）那个，来实现其他随机数生成器：

* 生成随机的（Int，Int）的数字Pair
* 0-max的正整数随机数。

这些函数都是：**输入是一个RNG（实现了nextInt的）。用它生成新的值和状态，然后对生成的值进行处理，返回处理后值&&刚才生成的状态。**
即：RNG=>(value,RNG)

```scala 
  def nonNegativeInt(rng: RNG): (Int, RNG) = {
    val (n1, rng1) = rng.nextInt
    val n = if (n1 < 0) -(n1 + 1) else n1 // 仅仅转换了生成是数字，依然有上面的状态
    (n, rng1)
  }  
  def randomPair(rng: RNG): ((Int, Int), RNG) = {
    val (n1, rng1) = rng.nextInt
    val (n2, rng2) = rng1.nextInt // 注意状态的串联
    ((n1, n2), rng2)
  }
```

类似的随机数生成器有很多，[参考github](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L40-L91)

可以发现这些所有的实现有**通用的模式**：
`RNG=>(value,RNG)`，这里的`RNG`可以理解成状态，它就是一个值，与nextInt无关，nextInt看成一个无关的函数就行，输入是seed输出一个数）

> 注：书中的随机数的例子不好理解，RNG本质就是个状态，夹杂了nextInt方法实在令人费解，不如最后的一到习题中的状态定义。

### 更好方法实现带状态的API
核心：上面实现的`nonNegativeInt`， `randomPair`方法比较『笨』，实现这两个方法都用到了nextInt，都是显示传递了rng状态。如果我们已经实现了一个状态函数，其他的函数可以由它衍生而来。具体来说，直接由生成的随机数Int的函数，得到nonNegativeInt或者randomPair。而不是每次都要调用nextInt，然后传递状态。因此我们需要：
* 抽象出上面的函数
* 定义一些辅助函数，帮助传递状态

我们把`RNG=>(value,RNG)`这个**函数**，看做一个整体定义为：
```scala
type Rand[+A] = RNG => (A, RNG)
```

一定注意：**他是个函数**！！！

首先，把生成随机数int的函数改造为符合`RNG=>(value,RNG)`的**状态转换函数**：

```scala
val int: Rand[Int] = rng => rng.nextInt // 注意理解int与nextInt函数的不同
```

定义unit函数，返回常量值与状态

```scala
def unit[A](a: A): Rand[A] = rng => (a, rng)
```

下面我们基于这个`int`函数可以生成`nonNegativeInt` `randomDouble`等等。可以使用map函数可以帮助我们实现**状态/函数之间的转化--[map](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L102-L107)**（输入是函数，输出也是函数）：

* [nonNegativeInt](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L114)：map(int)(f)（函数本身的变换）Rand[Int]==> Rand[Int]
* [double](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L116)：map(int)(f)（函数本身的变换）Rand[Int]==> Rand[Double]

我们还可以把多个**状态/函数组合起来--[map2](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L118-L125)**：

* [both](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L127-L129)：用map2来组合两个状态，生成状态Pair
* [randomPair](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L131-L133)：map2(int,int)(f)（函数的组合）Rand[Int]==> Rand[(Int,Int)]
* **[sequence](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L137-L158)**：3种实现方法，见下面的State，关注执行顺序，为什么他们都是是可行的。-- 本质：函数链


除此之外，还有**状态/函数的嵌套--[flatmap](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L164-L178)**：

* [nonNegativeLessThan](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L180-L188)：生成不小于某个数的正整数，应用了flatmap的可以表示嵌套函数的思想
* 使用flatMap实现
  * [map](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L190-L194)：里有unit
  * [map2](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/RNG.scala#L196-L203)：使用flatmap实现map2.map3...mapn

### 通用状态行为数据类型

我们泛化Rand，定义了更通用的类型：

```scala
// 本质上是一个函数
type State[S, +A] = S => (A, S)
```

**为了后续支持for推导**，我们使用另一种**等价的表示方法**

```scala
/**
 * 用class内的函数对象(run变量)来代表上面的函数
 */
case class State[S, +A](run: S => (A, S)) {
    def map = ...
    def flatMap = ...
}

// Rand的定义
type Rand[+A] = State[RNG, A]
```

相似的，我们可以对State实现在上面Rand里面实现的所有函数。因为使用了`run`来代表函数，因此有些差别。

* [map](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/State.scala#L24-L32)
* [map2](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/State.scala#L34-L42)
* [flatMap](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/State.scala#L44-L50)
* sequence
  * [foldRight](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/State.scala#L105-L112) 
  * [loop](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/State.scala#L118-L129) 
  * [reverse+foldLeft](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/State.scala#L138-L139)

### 纯函数式命令编程

#### for推导

当上面的State对象实现了map、flatMap时，我们可以使用for的语法糖

```scala
val ns: Rand[List[Int]] =
	int.flatMap(x =>
                int.flatMap( y=>
                            ints(x).map(xs =>  // ints为生成x个随机整数
                                       xs.map(_ % y))
                           )
               )
// 等价
val ns: Rand[List[Int]] = for {
    x <- int
    y <- int
    xs <- int(x)
} yield xs.map(_ % y)

```

#### 状态修改

modify的基本的想法是，

* get方法：看成一个函数，把State[S,A]中的A部分设置为状态S `State(s=>(s,s))`
* 对get进行flatMap，获得存储的状态S（flatMap只能获取到A部分，这就是get的作用），此时可以对状态进行修改。
* set(S)方法：看成一个函数，修改State[S,A]中的S，返回`State(_ => ((), s))`

```scala
def get[S]: State[S, S] = State(s => (s, s))

def set[S](s: S): State[S, Unit] = State(_ => ((), s))

def modify[S](f: S => S): State[S, Unit] = {
  get.flatMap(s1 => {
    val x = set(f(s1))
    x
  })
}
```


### 有限状态机

基于上面定义的函数我们可以实现一个有限自动机状态机

* 输入：List[Intput] 
* 输出：新的State（函数）
总体上：每个Input对一个的状态转换函数State，按n个输入的Input的顺序进行组合（sequence），最后输出一个新的状态转换函数。我们使用这个函数时（即传入参数&&调用），能直接产出最终的状态结果。

一个例子（书中练习）投币的机器(p61 6.11)
练习[参考](https://github.com/muzhi1991/learn-fp-in-scala/blob/28ed9e5cbae3f588ae6ec07fce4f3a01183710ab/src/chapter6/Machine.scala)
* simulateMachine：书中的答案--十分精巧
* simulate：我自己的实现



* 定义函数式数据结构
  * 实现一个List:
    * sealed Trait List[+A]
    * Nil定义
    * Cons定义：一个经典递归结构，可以实现链表，树等常见函数式数据结构 
    * 型变--协变
    * 伴生对象--apply--函数可变参数
  * 各种函数的实现
    * 模式匹配--定义：**可以侵入到表达式的数据结构内部，对这个结构进行检验和提取子表达式。**
      * 实现基本函数sum，product
    * 数据共享，List中减少赋值，对比tail与init（如何实现高效的init，需要改变数据结构，参考vector，使用了trie）
    * 类型推到的改进：
      * 为了使用匿名函数函数作为参数时的自动类型推导，我们需要柯里化（Scala的缺点导致的）
      * 类型推导时，z是子类可能需要显示声明父类以至此后面的推导：z:List[Int]（参考foldRight）
    * 基本函数sum，product的**泛化**为高阶函数-->**foldRight**和foldLeft && 相关练习应用
    * 难点：
      * 使用foldRight实现foldLeft（参考习题，涉及使用函数对象作为初始值z）
  * 标准库中的List：
    * 使用`::`代替`Cons`
    * `::`使用`:`结尾的操作符是有关联，机 `1::Nil`表示调用Nil的`::`方法，该方法内部调用`::(1,Nil）`
    * `case Cons(h,t)`变为`case h::t`
  * 有一些方法使用函数实现不高效：hasSubsequence函数需要**递归循环**？
  * 树的数据结构
    * ADT的定义：代数数据类型，参考[文章](https://www.cnblogs.com/moonandstar08/p/5759137.html)，理解sum（对应类的继承，option），product类型（对应tuple，record）
    * 书中Tree
      * 只有叶子结点Leaf存储数据
      * 分支节点Branch存在左右子树
    * 相关函数：
      * 具体函数：size(所有节点数量），maximum，depth，map
      * **抽象函数**：fold的实现（对比List的FoldRight抽象）


其他：

### 总结：重要的抽象函数
* 单值数据结构(Option,State)：
  * map：同数据结构的转化
  * flatmap：嵌套数据结构的转化
  * 单值数据结构的List：
    * sequence
    * traverse
* 多值数据结构(List,Stream)
  * map：同数据结构的转化
  * flatmap：嵌套数据结构的转化
  * foldRight：一组值到一个值的转化
  
如果把函数式编程看成一个数据的pipeline
* map：再pipe内一个一个阶段的处理数据
* flatmap：接入合并新的pipe数据
* compse：接口的适配？

关注分离：
* 一等函数（函数作为变量）：分离了『运算逻辑』与『运算的执行』
* Option：分离『实际发生的错误』和『错误处理』
* Stream：分离函数的『描述』与『求值』




