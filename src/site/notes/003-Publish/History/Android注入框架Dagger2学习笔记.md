---
{"dg-publish":true,"permalink":"/003-Publish/History/Android注入框架Dagger2学习笔记/","title":"Android注入框架Dagger2学习笔记","tags":["技术","Android","主框架"],"created":"2024-12-11T18:09:35.723+08:00","updated":"2024-12-11T18:09:35.723+08:00"}
---



## 简介
Dagger2是一个Google推荐的注入框架，关于什么是注入，为什么需要注入，可以参考下面的疑问以及解答文章。如果你在项目中使用MVP架构开发，强烈建议配合dagger2一起使用。
Dagger2大量使用注解，使开发者可以优雅的构造对象，同时，这些注解仅在编译期间生成代码，没有用到运行时的反射，对性能的影响可以忽略。使用者可以查看编译期间生成的代码，方便debug。
本文是学习Dagger2的笔记，需要先学习下面的文章，了解Dagger2，然后在学习过程中有一些疑问，形成一些思考与测试，形成此文。但是我还没有机会在项目中实际使用，若以后有实践经验会不断补充。

主要学习的文章：

*  [Android常用开源工具（1）-Dagger2入门](http://blog.csdn.net/duo2005duo/article/details/50618171)：dagger2的基本用法，很好的文章。
*  [Android常用开源工具（2）-Dagger2进阶](http://blog.csdn.net/duo2005duo/article/details/50696166)：dagger2的一些高级用法，如限定符。
*   [详解Dagger2](http://www.jcodecraeer.com/a/anzhuokaifa/androidkaifa/2015/0519/2892.html)：这篇文章翻译的不好，参考英文原文。

### 疑问以及解答
* [什么是依赖注入](<http://www.codekk.com/open-source-project-analysis/detail/Android/%E6%89%94%E7%89%A9%E7%BA%BF/%E5%85%AC%E5%85%B1%E6%8A%80%E6%9C%AF%E7%82%B9%E4%B9%8B%E4%BE%9D%E8%B5%96%E6%B3%A8%E5%85%A5>)

* [Dagger 源码解析](http://www.codekk.com/blogs/detail/54cfab086c4761e5001b2537)：dagger第一代的讲解，参考了解两者的差异。

* [Android中是否有必要使用依赖注入的争议](<http://dk-exp.com/2015/01/28/use-ioc-in-android/>)

## 一个非常容易的错误
inject的参数。。。不能是父类，必须是你注入的那个内

## 一个重要的概念：向子图暴露
即在component中写如下方法向把它依赖的component来暴露构造方法，且只能暴露一层。

```java
//定义ComponentB,modules可以不写，也可以与其他Component依赖的modules相同
@Component(modules={××××××××})
interface ComponentB{
    // 必须暴露出来，给A注入的对象用
    Apple apple()
}
//定义ComponentA，dependencies是A依赖的，区分 extend ComponentB
@Component(dependencies={ComponentB.class},modules={××××××××})//使用dependencies
interface ComponentA{
    // 注意注入的必须是具体的子类MyApp/XXActivity，不能是父类，如Application/Activity
    void inject(MyApp app)
}
```

## 错误汇总
1. 错误: com.android.example.devsummit.archdemo.di.module.Apple is bound multiple times:

	原因：一个类定义了多个provider：
	* component关联的（n个或者1个）module中有重复的provider
	* component A有dependencies的component B，A关联的Module A中有provider，component B关联的Module B中也有provider，**且在component B中暴露出来了**（不暴露不影响）

	下面的情况**没有**影响：
	
	* component 关联的module有provider，且类自己的构造函数有Inject标志，不影响，且provider优先。
	* component A有dependencies的component B， A关联的Module A中有provider，component B关联的Module B中也有provider，但是component B中没有暴露，不影响
 

2. 错误: Types may only contain one @Inject constructor.

	原因：一个类的构造函数只能有一个有@inject

3. 错误: com.android.example.devsummit.archdemo.di.module.Apple cannot be provided without an @Inject constructor or from an @Provides-annotated method.

	原因：
	* 没有provider或者inject
	* 如果component含有dependencies，可能没有向外面暴露

##区分易混淆的概念

1. component与module
2. component的dependencies与component接口的继承（extend）

## 关于Scope的粗浅理解
### 概念
scope是范围，标志一个注入器/对象的使用范围，那么可以理解：

* 父类（dependencies）有范围，子类必须有，且要小于它
* 父类子类范围不能一样？
* component中某一个对象构造有范围（Module中的一个方法有范围），那么该component就有范围。

### 作用
作用是允许对象被记录在正确的组件（component）。即module的构造方法用在正确的注入器（component）中，防止用错地方。**因为一个component关联的module中如果有其他的scope的provider会报错，因为component没有这个scope。**

### 自定义scope
```java
@Scope
@Documented
@Retention(RUNTIME)
public @interface Singleton{}
```

### 使用scope时规则
结合scope的概念理解

* component关联的Model中的**任何一个被构造的对象**有scope，则该整个component要加上这个scope。在暴露或者注入时（不暴露且不注入时，既不使用它构造对象时，不报错），会有如下错误
 > Error:(21, 1) 错误: com.android.example.devsummit.archdemo.di.component.MyTestComponent (unscoped) may not reference scoped bindings:
@Provides @com.android.example.devsummit.archdemo.di.scope.ActivityScope com.android.example.devsummit.archdemo.vo.User com.android.example.devsummit.archdemo.di.module.TestModule.user()

* component的dependencies与component自身的scope不能相同，即组件之间的scope不同
* Singleton的组件不能依赖其他scope的组件，只能其他scope的组件依赖Singleton的组件。
* 没有scope的不能依赖有scope的组件，理解一下。。。

  > Error:(21, 1) 错误: com.android.example.devsummit.archdemo.di.component.MyTestComponent (unscoped) cannot depend on scoped components:
@com.android.example.devsummit.archdemo.di.scope.ActivityScope com.android.example.devsummit.archdemo.di.component.MyTestComponentX
* 一个component不能同时有多个scope(Subcomponent除外)

	> Error:Execution failed for task ':app:compileDebugJavaWithJavac'.
	> java.lang.IllegalArgumentException: com.android.example.devsummit.archdemo.di.component.MyTestComponent was annotated with more than one @Scope annotation

## 其他概念
### @Subcomponent
Subcomponent与dependencies区别：
dependencies不会继承范围，Subcomponent会。子component同时具备两种不同生命周期的scope。

## 实践
### Module
dagger使用的目的在于更多的使用面向接口的编程。在Module中构造的对象返回的应当尽量是：**构造具体的类的实现，但是返回接口**，如②中所示，参考[文章](http://www.jianshu.com/p/c2feb21064bb)，常用写法如下：

```java
@Module
public class AppModule {
    private final MyApplication application;

    public AppModule(MyApplication application) {
        this.application = application;
    }

	// ①这种情况是返回module内部保持的变量
    @Provides
    @Singleton
    Context provideApplicationContext() { 
       return application;
    }

	// ②这种含有参数的写法会自动构造参数，前提是provideThreadExecutor，有provider或者@inject，ThreadExecutor是接口，JobExecutor是具体的类
	// 注意一种错误:参数和返回值不能是相同的类！！
    @Provides
    @Singleton
    ThreadExecutor provideThreadExecutor(JobExecutor jobExecutor) {
        return jobExecutor;
    }

	// ③这种写法会自动构造参数，前提是RetrofitManager，有provider或者@inject
    @Provides
    @Singleton
    ApiService providesApiService(RetrofitManager retrofitManager) {
        return retrofitManager.getService();
    }

	// ④这种写法是比较常见的，内部new一个对象返回
    @Provides
    @Singleton
    DBManager provideDBManager() {
        return new DBManager(application);
    }
}
```	
