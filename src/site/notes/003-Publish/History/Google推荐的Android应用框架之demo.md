---
{"dg-publish":true,"permalink":"/003-Publish/History/Google推荐的Android应用框架之demo/","title":"Google推荐的Android应用框架之demo","tags":["技术","Android","主框架"],"created":"2024-12-11T18:09:35.716+08:00","updated":"2024-12-11T18:09:35.716+08:00"}
---


近日，在Google开发者的微博上推荐了一个视频，是2015年开发者峰会上两位Google的Coder的演讲，关于Android应用框架，中文相关介绍文章[看这里](http://blog.zhaiyifan.cn/2016/01/29/android-app-architecture-2015/)。
演讲中有个demo，类似于一个QQ的聊天界面的程序，用这个例子讲解了如何优雅地处理在各种网络异常的情况下的UI界面，既如何设计一个offline的应用。
网络请求与本地缓存相结合以获取最好的用户体验应当是几乎所有Android应用的基本需求。从我的开发经验看，我们往往会忽略之一点。大部分情况下，都是直接从网络拿回数据显示在界面上，如果网络不好，就不断loading。。。在无网情况下，app是不可用的。

* 拉逻辑--获取列表，此过程是我们最常用，我们大部分是获取数据然后展示给用户，仅仅需要本地有一份数据以备在无网时使用。这个逻辑在应用中基本是通用的。
* 推逻辑--提交post，demo中对某些请求进行排队，在后台不断重试，这种逻辑对于某些应用应该是不适用的，毕竟大部分应用是不需要这种『强同步』的（既server与local完全一致，如聊天list或记账，微信发朋友圈）。在一般应用中以下两种情况更加常见：

	* 结果失败与否并不重要。如点赞这些功能，当然也可以这么干。
	* 操作是一些重要的不可重复操作，必须锁死屏幕。如提交订单，付款等。

选择哪一种交互是设计阶段必须考虑的，这直接决定了编码方式。当然，一个设计良好的框架必须要提供这些能力:

* 封装网络请求与本地数据支持拉逻辑的操作，开发者可以选择是否先使用本地数据再等待网络更新。还是只使用网络数据。
* 支持简单请求，锁死等待，还是异步的。
* 支持后台可重复发送的请求--强同步--既Act locally, Sync globally。

扯远了，需要一篇文章写一些app中可能用到各种交互与请求逻辑。

下面翻译下Google这个demo的README文件。了解一下设计思想。

## 简介
这是个简单的社交软件，列表可以展示其他人的消息（我们叫他feed流），用户还可以发送文本信息。

本文的目的是讲述如何设计一个应用，可以离线使用，并在网络可用时同步信息到服务器，尽量不影响用户。

* 如何工作
* 组件
* 数据流
	* 发送信息
	* 同步feed流
	* 保持UI更新
* 安装&运行&测试
	* 安装
	* 运行
	* 测试
* 其他
	* 避免重复发送信息 
* 版权许可

## 如何工作
关于『离线设计』这个主题，许多解决方案都是依赖于特定的需求，如果想要运用到其他场景，需要一些修改。因此，这个demo的同步逻辑可能并不适合你的需求。你应当把它当做一个例子学习，然后找出适合你的应用的解决方案。不幸的是，没有设计离线应用的万金油。
这里，我们会解释用户的交互流是如何工作的，会给你的应用提供一些不错的想法。
这个例子没有严格依照任何的构架模式，它使用混合方法满足自己的需求。我们假设这个应用会不断成长为一个大的app（因此下面比较复杂）。尽管增加了复杂度，但是我没尽量是这个demo更加真实有用。

## 组件
* Value Objects:
	* 这些对象通常保持着数据库中存储的数据，这些对象也知道如何校验这些数据的合法性。如果Server发送了一下无效数据（比如，API改变了），我们可以在它们影响Model之前就忽略它们。
* Models:
	* 扶着数据持久化，保持到本地，并提供方法获取数据。
* Controllers:
	* 保存着应用程序主要逻辑，它决定如何做某事（如发送消息），处理gcm消息等。
* Activities:
	* 控制用户界面，他们知道哪里需要加载数据，什么时候需要刷新自己。
* Events:
	* 本demo使用一个全局的EventBus，它唯一的作用是通知UI更新。
	* 当UI需要从应用逻辑那里获取数据，直接调用相关方法就可以。这种设计符合Android这个UI生命周期短的设计。这种数据流也避免了UI和后台线程的循环引用。
* Jobs:
	* 他们可能是定义良好的网络操作（这比较典型，但未必）， 可以把你的应用逻辑拆分成多个job，这更方便测试和扩展。例如发送用户的文本消息到服务器就是一个job，同步用户的feed流到服务器也是个job。

## 数据流

### 发送消息
当用户点击按钮，首先会发生这么4步：1.验证消息合法性。2.保持消息到本地磁盘存储（这个例子中这是一个job）。3.更新PostModel添加新的信息。4.发送event通知有消息更新。
（可选的）4.a.如果UI可见，在收到event后更新内容。
注意：上面这些步骤没有涉及网络，但是我已经更新了用户界面&&保持必要的数据到磁盘。最后，我们需要与Server同步。

同步的步骤如下
![Sync To Server](https://camo.githubusercontent.com/590b0315c04a7e5c0b429fbd521f99062aacde1d/68747470733a2f2f7777772e657665726e6f74652e636f6d2f73686172642f7331392f73682f31313036313730662d346637332d343531392d616537642d3466346166313364343138322f366562656436376534613734613461342f7265732f62656166626235362d633564632d346563622d393934662d3330653036616232313961342f736b697463682e706e67)

* 优先级作业队列，负责处理持久化job，失败时的回退操作等，所以图中没有这些细节。
* 一个真实环境的app应该与JobScheduler结合，确保消息在应用关闭后也可以发送。
* 当job失败时，你最好保持一些信息到磁盘，那样下次用户打开app时，你可以通知他们发生了什么。这个demo中，当FeedActivity不可见时会显示系统通知。

### 同步Feed流
同步操作有下面三个组件负责：

* FeedModel：
	* 为每个feed流保存最后一条FeedItem（既发送的消息）的时间戳。这个时间戳有两个作用
		* 当我们刷新feed流时，只接收最新的item。
		* 创建保存在本地的消息。本地创建的消息会显示在feed查询中是很重要的。 客户端的时间戳可能与server不同，只有当消息与Server同步时，我们分配feed最新的时间戳给这个新的消息。？？？
	* 给UI层提供从数据库获取feed流的方法。
* FeedController：
	* 负责创建`FetchFeedJob`
	* 监听消息更新失败，使用系统的通知栏通知用户。除非其他的UI组件（如FeedActivity）已经处理了错误。
	* 这个示例的app很基础，在真实环境中，它可能还要负责当有GCM推送到达时刷新feed流。以及包括一些防止用户频繁刷新的逻辑。
* FetFeedJob：
	* 发出API调用，获取某个feed流的最新消息列表。还负责更新model和发出event。

### 保持UI更新
后台与UI的交互需要良好的定义。

* 等界面需要做某件事情时，UI直接调用某个方法（如发送消息）。
* 后台任务结束时发出event通知UI更新（如更新失败）。

UI组件根据自身的生命周期负责注册/反注册EventBus，因为后台永远不会持有UI的引用，我们不想有内存泄漏的风险。

* EventBus的使用可能引起一些特殊的情况：由于UI错过了一些event时，导致不同步。本Demo使用下面的规则避免这种清空。
	* 当组件的生命周期开始时，先注册event，然后在再从model加载数据。
	* 当正在加载数据时，如果有event到来，会在数据加载完成后再次触发一次同步。
	* 所有的event都含有一个时间戳，它表示与该event相关的最老的item的时间。当UI访问model层时会使用这个时间戳，那么，如果item插入数据库时顺序不同，我们依然可以获取到这些item，因为我们使用了最老的时间戳。？？？
	* 当组件的生命周期结束时（例如`Activity#onStop`）,停止监听event。如果，我们再返回应用，他会做一个完整的同步，确保不会丢失在这期间发生的event。

**这不是唯一的办法**：示例中使用的是一个全局的EventBus，你也可以用Rx实现相似的功能，或者手动设置listener，以及其他相似的技术。总之，你需要根据自己的应用来评估。

## 安装&&运行&&测试
### 安装
demo包含一个简单的server，你需要安装Ruby On Rails来运行它。建议你通过[Ruby Version Manager](https://rvm.io/)来安装。安装完之后，使用下面命令启动server：

```xml
> cd server;
> bundle install;
> rake db:migrate RAILS_ENV=development;
```
这会安装依赖，创建数据库。

### 运行
#### 服务器

```xml
> cd server;
> rails s
```

#### 客户端
demo中app使用模拟器环境默认的host地址。（`http://10.0.2.2:3000`）。如果你在模拟器里运行，已经可以正常工作。你可以在设置菜单中修改，或者在`DemoConfig`类中直接修改。

### 测试

* 服务器：服务端没有。。我又不关心。
* 客户端：你可以这样运行测试 `> cd client; > ./gradlew clean app:connectedCheck app:test`

## 其他
### 避免重复发送消息
编写移动应用意味着你要处理不可靠的网络。通过使用持久化的job（存储在本地），在网络可用时，app工作了很不错，可是并不完美。在不可靠的网络条件下，我们的应用可能出现这种情况：服务端那数据已经更新存储了，但是客户端没有收到成功的返回。那么客户端依然认为消息没有上传，会不断重试。如果server的返回有问题，这就会变得更糟糕。
通常情况下，重试就以为这重复上传消息。有很多策略可以解决这个问题。demo中我们使用一个唯一的二维元组`(userId, clientId)`来避免重复，它的原理如下：

* 客户端创建消息时，生成一个唯一的`clientId`（`UUID.randomUUID().toString()`,注：随着时间会变化）。这个UUID和用户ID的组合在客户端和服务端都是唯一的。
* 当服务端收到消息时，检查这个元组是否存在，如果以及存在，不会新增一条存储记录，而是仅仅返回这个条目已经存在。
* 当客户端获取feed时，如果发现一个消息的` (userId, client Id)`与某个存在的消息相同，会覆盖这存在的消息。这可能发生在这种情况下发生：客户端没有收到服务端的返回，只是本地保存了这个消息，，但是这个消息出现在其他某个请求的返回里面。？？需要这样吗

你可以在`server/app/controllers/posts_controller.rb`中使用`error_before_saving_post`和`error_after_saving_post`来触发这种特殊的情况。

## 版权许可
>Copyright (C) 2015 The Android Open Source Project

>Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

>You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0

>Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

>See the License for the specific language governing permissions and limitations under the License.
