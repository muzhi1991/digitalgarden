---
{"dg-publish":true,"dg-path":"History/Android构架系列之一-工程结构与Gradle配置.md","permalink":"/History/Android构架系列之一-工程结构与Gradle配置/","title":"Android构架系列之一--工程结构与Gradle配置","tags":["技术","Android","主框架"],"created":"2016-03-27 13:51:46","updated":"2016-03-27 13:51:46"}
---


## 从最基本结构说起
新建一个helloworld工程结构：
![](https://fodi.389266.xyz/images/IMG-80d2095f5d24dd0f.webp)
工程目录下只有根目录和一个app模块，没有其他模块。

* 根目录`build.gradle`：关于项目的一些总体配置(比如编译系统使用的仓库和依赖，其他模块都使用的仓库和依赖)；
* 根目录`settings.gralde`：项目的模块构成，最简单。
* app模块`build.gradle`：app模块的具体配置，该模块自身的性质，该模块依赖的库（jcenter的开源外部库，内部模块之间）

### 根目录build.gradle
```groovy
// Top-level build file where you can add configuration options common to all sub-projects/modules.

buildscript {
    repositories {
        jcenter()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:1.5.0'

        // NOTE: Do not place your application dependencies here; they belong
        // in the individual module build.gradle files
    }
}

allprojects {
    repositories {
        jcenter()
    }
}

task clean(type: Delete) {
    delete rootProject.buildDir
}

```
首先，区分一个概念：**编译系统自身**与**被编译的内容**,[见stackoverflow](http://stackoverflow.com/questions/30158971/whats-the-difference-between-buildscript-and-allprojects-in-build-gradle)。

* `buildscript`：配置的是**buildsystem自身（即Gradle）**使用的仓库和依赖，在这里，buildsystem依赖的库/插件是android gradle1.5，从jcenter下载。gradle就是用来编译的。

	> 区分一个东西：gradle 与 Android tools的gradle插件
> gradle自身是一个编译系统，它的版本配置在AS的设置中。
> Android tools的gradle插件，是google开发的在一个gradle插件，在gralde中使用，可以让gradle支持编译Android应用程序。也就是说
> 
> * 如果gradle引用其他插件可以编译其他应用程序
> * 可以自定义插件，修改gradle编译android应用的行为，实现一些自定义功能。相应的**会引入plugin（必有）、一些自定义的命令、配置区块！**

	此外buildsystem还可能依赖一些其他常用的库/插件，**编译注解**常用的apt：
	
	```groovy
	 dependencies {
    classpath 'com.android.tools.build:gradle:1.5.0'
    classpath 'com.neenbedankt.gradle.plugins:android-apt:1.4'
  }
	```

* `allprojects`：不是必须的，作用是遍历所有的模块并且应用同一个配置。在这里配置的是所有模块都使用jcenter仓库。如果不在这里配置，可以在每个模块的`build.gradle`中单独配置（思考：how to？）。除此之外，`allprojects`中还可以配置：
	* `dependencies`，所有模块公用的dependencies
	* 一些变量，所有模块都用的变量,
	
	``` groovy
	allprojects {
	//为什么是ext，请参考其他文章
  ext {
    androidApplicationId = 'com.fernanependocejas.android10.sample.presentation'
    androidVersionCode = 1
    androidVersionName = "1.0"
    testInstrumentationRunner = "android.support.test.runner.AndroidJUnitRunner"
    testApplicationId = 'com.fernandocejas.android10.sample.presentation.test'
  }
}
	```
	
* 自定义`task`：上面的clean是覆盖了系统的clean任务，同样在这里我们可以写一下自定义的gradle 任务

	```groovy
	task wrapper(type: Wrapper) {
  description 'Creates the gradle wrapper.'
  gradleVersion '2.10'
}
// 例如定义一下task运行某个模块的单元测试
task runDomainUnitTests(dependsOn: [':domain:test']) {
  description 'Run unit tests for the domain layer.'
}
task runDataUnitTests(dependsOn: [':data:cleanTestDebugUnitTest', ':data:testDebugUnitTest']) {
  description 'Run unit tests for the data layer.'
}
task runUnitTests(dependsOn: ['runDomainUnitTests', 'runDataUnitTests']) {
  description 'Run unit tests for both domain and data layers.'
}
```

* 其他内容：是否需要 `apply plugin: 'com.android.application'`？这个例子中不可以，application插件一般放在app模块。
	> 有些特别的例子（比如eclipse转换成的as工程），根目录就是app模块，此时的app的build.gradle与根目录的build.gradle是一个文件，所有可以这么写。[android gradle的官方demo](http://tools.android.com/tech-docs/new-build-system/user-guide#TOC-Basic-Project-Setup)中，最简单的gradle工程也是用了一个`build.gradle`文件。

### 根目录的settings.gradle
```groovy
include ':app'
```
这个文件最简单，含义是这个项目包括的**所有**模块的名字（位置），这里的含义是根目录下面的app模块，如果是子目录，需要写出目录

```groovy
include ':app'
// 子目录third_party下的android_support模块
include 'third_party:android_support'
// 也可以这样写
// include ':app', 'third_party:android_support'
```
注意：一个特例：当根目录就是app模块是不需要在`setting.gradle`中写出。

### app模块build.gradle
```groovy
apply plugin: 'com.android.application'

android {
    compileSdkVersion 23
    buildToolsVersion "23.0.2"

    defaultConfig {
        applicationId "com.example.baidu.helloworld"
        minSdkVersion 14
        targetSdkVersion 23
        versionCode 1
        versionName "1.0"
    }
    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        }
    }
}



dependencies {
    compile fileTree(dir: 'libs', include: ['*.jar'])
    testCompile 'junit:junit:4.12'
    compile 'com.android.support:appcompat-v7:23.2.1'
}

```

这个是android app配置的主要内容，从顶层开始看：`apply plugin`,`android`,`dependencies`三大区块内容（除此此外还可以有`repositories`，**`configurations`**）。

* `apply plugin: xxx`：插件就是之前引入的android gradle 1.5库导入的常用的有：
	* `'com.android.application'` 指示当前编译的是android app主模块，一个应用中有且仅有一个。
	* `'com.android.library'` 指示当前编译的是android 库模块，这种library可以包含android包和资源并且**可以导出aar包**，这是AS特有的功能，包含了资源文件。
	* `'java'` 指示当前编译的是纯的java库模块，注意这个plugin不是android gradle插件引入的，是gradle原生。如果某个库被定义成此类型，**此模块不能引用其他Android模块！**这种library能导出jar包，和eclipse的功能类似。**疑问：可以只引用，不编译android.jar吗？**
	
* `dependencies`区块：该模块依赖的库。通常有下面这些写法：
	* `compile fileTree(dir: 'libs', include: ['*.jar'])` ：编译位于libs目录下的一下jar包。
	* `compile xxx` ：编译jcenter的某个库。最常用。
	* `compile project(':xxx')` ：编译其他模块，**建立模块之间的依赖，十分重要**，也可以在AS的项目配置中查看。
		
	除此之外，还可能包括：
	
	* `provided xxx` ：**还不了解，只引用，不编译？？？**
	* `testCompile xxx` ：**测试情况下编译?**
	
	当请入第三方gradle插件时，比如apt插件会引入：
	
	* `apt xxx` ：引入第三方注解依赖，**思考并理解与apt插件的关系**：apt是编译工具，xxx是被编译的内容，比如dagger的注解。
	
* `android`区块：这里android gradle插件引入的区块，可以定义所有android相关的配置，比如应用编译的签名，applicationId，版本，lint开关，编译选项(java版本)，设置源码目录，生成多个包等等，**大部分的android编译的配置**。[请阅读文档！](ttp://tools.android.com/tech-docs/new-build-system/user-guide)
* 其他区块：还有一些区块是其他gradle插件引入的，比如微信的资源混淆插件就回加入一个自己的区块，可以配置资源混淆相关的选项。

### 其他
* `apply from: "$rootDir/config/config.gradle"`：这种样式可能有多个，表示引入配置文件。`build.gradle`可能很长，按照功能把build分为多个子build是不错的实践。

	> 在根目录新建文件夹`config`，在文件夹config中，写多个`xxx.gradle`文件，比如`quality.gradle`，它来为我的项目集成并配置所有的质量控制工具。然后通过`apply from`导入到对应的build.gradle文件。
	
## 一些待理解的问题
* `provided` 的具体含义的理解
* `configurations` 区块
* 测试相关的内容

## 工程实践
参考[clean构架的demo](https://github.com/android10/Android-CleanArchitecture)，总结几点

* `build.gradle`按功能分成多个`xxx.gradle`文件，即必须分块！
* 将所有依赖的内容放入一个单独的文件中，如`config.gradle`。方便以后配置管理。（`ext`的使用！）
* 细节：
	* `compile project()` 模块间依赖的建立。 

## 深入学习
* [深入理解Android之Gradle](http://blog.csdn.net/innost/article/details/48228651)
* [android gradle 插件官方文档](http://tools.android.com/tech-docs/new-build-system/user-guide)
* [gradle官方文档](https://docs.gradle.org/current/userguide/userguide_single.html)

