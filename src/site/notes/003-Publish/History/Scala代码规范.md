---
{"dg-publish":true,"dg-path":"History/Scala代码规范.md","permalink":"/History/Scala代码规范/","title":"Scala代码规范","tags":["技术","Scala","Spark"],"created":"2024-12-11T18:09:35.726+08:00","updated":"2024-12-11T18:09:35.726+08:00"}
---



在工作中，我们使用Scala作为在Spark应用的开发语言，由于公司的代码托管平台暂不支持Scala代码的规范检测，团队内部拟定了一个的代码规范，希望能在统一代码风格的同时，足够简单且可操作。

## 基本原则

* 遵循[Baidu Java代码规范](http://styleguide.baidu.com/style/java/index.html)
* 使用Eclipse/IntelliJ idea的format文件格式化代码，[地址](http://gitlab.baidu.com/dingxuefeng/java-style/tree/master)
* 使用空格代替tab/文件编码UTF-8
* 使用驼峰命名法则



## 通用规范

### 源文件规范

#### 文件名

源文件名必须和它包含的顶层类名保持一致，包括大小写，并以`.scala`作为后缀名。

#### 文件编码

所有源文件编码必须是 **UTF-8**

#### 特殊字符

##### 空格

除了换行符之外，ASCII空格（0x20）是唯一合法的空格字符。这意味着

- 所有在源代码中（包括字符、字符串以及注释中）出现的其他空格字符需要转义,例如 `Tab` 用`\t`表示。

- 缩进必须使用 **4个空格** 而不是 `Tab`

  > 新增 **4个** 空格

#####  特殊转义字符

对于有特殊转义表示的字符 `\b, \t, \n, \f, \r, \", \', \\`，禁止使用其它等价转义方式。例如`\012`或者`\u00a`表示。

##### 非ASCII字符

对于非ASCII字符，可以使用实际字符（如 `∞`）或者它的Unicode转义（如`\u221e`），取决于哪种写法的可读性更好。
使用注释有助于增强可读性
示例：

| 示例                                       | 说明                                       |
| ---------------------------------------- | ---------------------------------------- |
| val unitAbbrev = “\u03bcs”; // “μs”      | 虽然合法但是很没有必要的写法                           |
| val unitAbbrev = “\u03bcs”; // Greek letter mu, “s” | 虽然合法但是很难理解的写法                            |
| val unitAbbrev = “\u03bcs”;              | 让人完全读不懂的写法                               |
| return ‘\ufeff’ + content; // byte order mark | 很好的写法，用Unicode转义来表示非打印字符，并且有合适的注释帮助阅读者理解 |
| val unitAbbrev = “μs”;                   | 最佳写法，无需注释就可以理解                           |

#### 组织结构

源文件必须按顺序由以下部分组成：

- 许可证（License）或版权声明（Copyright）
- package语句
- import语句
- 唯一的顶层类
- 每两部分之间用一个空行分隔

### 代码书写规范

#### 列宽

每行不超过**150个字符**

#### 缩进

4个**空格**，严禁使用Tab

#### 括号的使用

基本原则：K&R风格。

- 左花括号（{）前不能换行，在其后换行。
- 在右花括号（}）前要有换行。
- 如果右花括号是一句语句、一个方法、构造函数或非匿名类的结尾，其后需要换行。

```scala
new MyClass() { // 左花括号前不能换行，在其后换行
    @Override
    def method():Unit= {
        if (condition()) {
            try {
                do {
                    something()
                } while (!stop()) // do-while中间的右花括号后
            } catch  { // try-catch中间的右花括号后无需换行
                case e:Exception =>recover()
            } // try-catch结束，右花括号后需要换行
        } else { // if-else中间的右花括号后无需换行
            doSomethingElese()
        } // if-else结束，右花括号后需要换行
    }
```

其他：scala中的简单**表达式**可以省略括号

```scala
// 推荐
def square(x: Int) = x * x
val y = if (x < 0) -x else x 
if(cond1){
  // one line statement
}else{
  // one line statement
}

// 避免
if (x < 0) 
	-x
else
	x
```

#### 空行的使用

在以下情况下增加空行：

- 在类的不同的成员间增加空行，包括：成员变量、构造函数、方法、内部类、静态初始化块、实例初始化块等两个成员变量声明之间可以不加空行。空行通常用于对成员变量进行逻辑分组。
- 方法体内，按需增加空行，以便从逻辑上对语句进行分组
- **禁止使用连续的空行**。

#### 注释风格

使用java风格的注释，不用使用scala风格

```scala
/** 单行注释 */
// 单行注释

/**
 * java风格的多行注释
 * 推荐使用
 */

/** scala风格的注释
  * 不推荐使用
  */
```

## 命名规范

基本原则：**驼峰命名**，命名有业务含义。这里列举了Spark编程中常见是变量/数据结构，在工程中尽量统一使用下列的命名规则。
* 输入输出目录的命名
  * input命名：`xxInput`
  * output命名：`txtOutput/parquetOutput` / `output`
* Spark中的RDD/DataFrame/DataSet数据结构的命名，使用具有业务含义的命名，并带上后缀，禁止无意义的名字，如a,b。
  * DataFrame：`xxDF`
  * RDD: `xxRDD`
  * DataSet: `xxDS`
* UDF命名规范
  * `xxUDF`
* 函数参数命名
  * Column作为参数的命名`def min(col: Column)`
  * 用String作为列名的参数`def min(colName: String)`
* 特别地，存储落地文件的列名可以使用自定义的规则，如Person表的`min_age`列



## Scala相关规范

### 变量

* 一行只声明一个变量
* **scala中尽量使用val代替var**
* 一个不变的val 必须被初始化，也就是说在声明的时候就必须定义
* 一个可变的变量用关键字var 来声明，声明一个var 时将其初始化

```scala
// 推荐
val name = "abc"
// 不推荐
var name = "abc";
```

### 方法

#### 定义

* 显示指定调用的参数类型与返回类型

#### 方法调用避免中缀表示法

```scala
// 推荐
list.map(func)
string.contains("foo")
// 不推荐
list map (func)
string contains "foo"
```

#### Return的使用

原则：在方法中避免使用Return语句，而是直接使用值返回，否则容易出现错误

```scala
// 使用
def receive(rpc: WebSocketRPC): Option[Response] = {
  tableFut.onComplete { table =>
    if (table.isFailure) {
       None // Do not do that!
    } else { ... }
  }
}
// 避免，下面会导致onComplete返回的是NonLocalReturnControl
// return的意思是：退出包含 return 的最近的一个方法（不是函数）,这里是receive
def receive(rpc: WebSocketRPC): Option[Response] = {
  tableFut.onComplete { table =>
    if (table.isFailure) {
      return None // Do not do that!
    } else { ... }
  }
}
```

下面几种情况可以方法Method可有使用return

* 简化控制逻辑，使用guard

  ```scala
  def doSomething(obj: Any): Any = {
    if (obj eq null) {
      return null
    }
    // do something ...
  }
  ```

* 退出循环逻辑

  ```scala
  while (true) {
    if (cond) {
      return
    }
  }
  ```

  ​

### 其他

####  使用`@Override`

Scala中，虽然有的时候override是可选的，但是坚持使用override是安全的。

#### 应尽量避免让trait去extend一个class。

```scala
class StarfleetComponent 
// 不推荐
trait StarfleetWarpCore extends StarfleetComponent 
// 推荐
class Starship extends StarfleetComponent with StarfleetWarpCore 
class RomulanStuff // won't compile 
class Warbird extends RomulanStuff with StarfleetWarpCore
```

#### 链式数据结构的选择

选择使用Seq时

* 若需要索引下标功能，优先考虑选择Vector
* 若需要Mutable的集合，则选择ArrayBuffer：
* 若要选择Linear集合，优先选择List
* 若需要Mutable的集合，则选择ListBuffer。


* 若需要快速、通用、不变、带顺序的集合，应优先考虑使用Vector。
* 如果需要选择通用的可变集合，应优先考虑使用ArrayBuffer。

一个原则是：**当你对选择集合类型犹疑不定时，就应选择使用Vector。**
需要注意的是：当我们创建了一个IndexSeq时，Scala实际上会创建Vector对象：

```scala
scala> val x = IndexedSeq(1,2,3) 
x: IndexedSeq[Int]  = Vector(1, 2, 3)
```

Vector很好地平衡了快速的随机选择和快速的随机更新（函数式）操作。Vector是Scala集合库中最灵活的高效集合。

当面对一个大的集合，且新元素总是要添加到集合末尾时，就可以选择ArrayBuffer。如果使用的可变集合特性更近似于List这样的线性集合，则考虑使用ListBuffer。

如果需要将大量数据添加到集合中，建议选择使用List的prepend操作，将这些数据添加到List头部，最后做一次reverse操作。

####  Lazy的使用

当一个类的某个字段在获取值时需要耗费资源，并且，该字段的值并非一开始就需要使用，则应将该字段声明为lazy。

####  异步

##### Future的使用

在使用Future进行并发处理时，应使用回调的方式，而非阻塞。

```scala
// 避免
val f = Future {
 //executing long time
}
val result = Await.result(f, 5 second)

//推荐
val f = Future {
 //executing long time
}
f.onComplete {
 case Success(result) => //handle result
 case Failure(e) => e.printStackTrace
}
```

#### 使用par对集合的操作并行化

若有多个操作需要并行进行同步操作，可以选择使用par集合

```scala
// 同时访问
val urls = List("http://scala-lang.org",
  "http://agiledon.github.com")
def fromURL(url:  String)  = scala.io.Source.fromURL(url)
  .getLines().mkString("\n")
val t = System.currentTimeMillis()
urls.par.map(fromURL(_))
println("time: " + (System.currentTimeMillis - t) + "ms")
```

##### 线程联合操作

若有多个操作需要并行进行异步操作，并在全部完成后进行某些操作，则采用for comprehension对future进行join方式的执行。

```scala
val result1 = Cloud.runAlgorithm(10)
val result2 = Cloud.runAlgorithm(20)
val result3 = Cloud.runAlgorithm(30)
val result = for {
  r1 <- result1 r2 <- result2 r3 <- result3 } yield (r1 + r2 + r3) result onSuccess { case result =>  println(s"total = $result")
}
```

## Spark相关规范
由于Spark API的特点，我们特殊规定了几种规则，适用于Spark应用的开发。

### 链式调用规范

如果链式调用过长，尽量一个方法调用占用一行，增加可读性。注意：spark-shell下直接粘贴代码会出错，输入`:p `或者`:paste`**进入粘贴模式**，`ctrl-D`退出并**开始执行**。

```scala
val extractDF = spark.read.parquet("someS3Path")
  .select(
    "name",
    "Date of Birth"
  )
  .transform(someCustomTransformation)
  .withColumnRenamed("Date of Birth", "date_of_birth")
  .filter(
    col("date_of_birth") > "1999-01-02"
  )
```
### Spark SQL书写规范

Spark SQL中属性的SQL语句往往很长，且可读性很差，我们规范: 使用`"""`表示多行字符串来书写语句。

```scala
  val firstDF = sqlContext.sql(
    """ select
          ip,
          sum(ip_line_cnt) as ip_line_cnt,
          sum(ip_click_cnt) as ip_click_cnt,
          sum(ip_zhudong_cnt) as ip_zhudong_cnt,
          sum(ip_line_cookieage_less1d) as ip_line_cookieage_less1d,
          sum(ip_ua_abnormal_cnt) as ip_ua_abnormal_cnt
        from table_day_temp_ip
        group by ip
    """)
```

## 工具篇

### scala-style工具

为了使得团队代码风格一致，我们可以使用一些[静态代码检测工具](http://www.scalastyle.org)。该工具对代码Scala代码进行基本的格式检查，并输出检测结果。可以配置为**maven插件**，作为例行的代码检测。在官方的默认配置中有一些配置可能并不使用，例如println配置。

推荐：[Spark官方的配置文件](https://github.com/apache/spark/blob/master/scalastyle-config.xml)

## read more

[spark 官方scala编程风格指南](https://github.com/databricks/scala-style-guide/blob/master/README-ZH.md)
[spark-style-guide](https://github.com/MrPowers/spark-style-guide)
[内网Scala编程规范](http://wiki.baidu.com/pages/viewpage.action?pageId=156874244)
