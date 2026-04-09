---
{"dg-publish":true,"dg-path":"History/Docker技术入门与实践.md","permalink":"/History/Docker技术入门与实践/","title":"常见Web架构与Docker部署实践","tags":["技术","全栈","运维"],"created":"2016-09-20 16:00:12","updated":"2016-09-20 16:00:12","dg-note-properties":{"title":"常见Web架构与Docker部署实践","aliases":[],"tags":["技术","全栈","运维"],"date created":"2016-09-20 16:00:12","date modified":"2016-09-20 16:00:12","status":"Done"}}
---





运维部署的第一篇文章，自上而下（从架构向下）的介绍一些常见概念。最后用docker部署一个简单的架构，可以应付小企业的常规需求。

## 基本知识

### Web架构的演变

这里说的Web架构是指系统架构，不涉及代码层面。

* 第零阶段：常见的开发状态，单点服务器、数据库。甚至服务器与数据库在同一个机器上，没有任何负载均衡。
* 第一阶段：业务简单，流量小，没有专业运维。利用Nginx或HAProxy进行单点的负载均衡，在七层之上利用HTTP协议就可以。
* 第二阶段：业务复杂，流量变大，这时单点的Nginx已经不能满足，这时使用开源LVS/商用负责均衡F5（Array），Nginx此时就作为LVS的节点来使用。**形成LVS/F5—> Nginx/HAProxy—>AppServer**。可能为了满足不同网络环境下**静态资源**响应速度还需要购买CDN服务。
* 第三阶段：规模很大，需要**降低成本**，在上述构架上优化，负责均衡使用自定义开发的LVS代替商业方案。CDN自行部署使用Nginx或者 Squid/Varnish 方案部署缓存服务器。LVS — Nginx/Haproxy — Squid/Varnish — AppServer。

一个典型结构：

* CDN服务缓存静态资源。
* 第一层：LVS管理**多个Nginx**，LVS只负责**TCP/IP层的转发**，实现负载均衡，与业务无关。
* 第二层：每个Nginx，下面都和所有的应用服务器相连接，实现业务相关的负载均衡（Nginx是http应用层的负载均衡，如根据访问目录动静分离）。同时，Nginx还可以提供静态文件的Web服务和部分的缓存功能。
* 第三层：N个web服务器
* 第四层：web服务器可能需要访问数据库MySQL集群，缓存redis集群。

> 注意:
>
> 1. lvs,ngnix都有单点问题，需要配合使用keepalive
> 2. 更一般情况，如果第一层不用lvs，单独用ngnix也可以
> 3. lvs能力强一些，并发几百万，nginx弱一些，几万到几十

这里缺个图😀

### 网络基础概念

* VIP：虚拟IP地址，LVS中的一个概念，即对外暴露的IP地址。
* 虚拟主机：虚拟主机是使用特殊的软硬件技术，把一台真实的物理服务器主机分割成多个逻辑存储单元。每个逻辑单元都没有物理实体，但是每一个逻辑单元都能像真实的物理主机一样在网络上工作，具有单独的IP地址（或共享的IP地址）、独立的域名以及完整的Internet服务器（支持WWW、FTP、E-mail等）功能。虚拟主机的关键技术在于，即使在同一台硬件、同一个操作系统上，运行着为多个用户打开的不同的服务器程式，也互不干扰。而各个用户拥有自己的一部分系统资源（IP地址、文档存储空间、内存、CPU等）。各个虚拟主机之间完全独立，在外界看来，每一台虚拟主机和一台单独的主机的表现完全相同。所以这种被虚拟化的逻辑主机被形象地称为“虚拟主机”。**这种技术适用于那种小型网站，毕竟大部分网站的所消耗的资源远远小于一台机器。**参考Nginx的配置

### LVS基础知识

LVS涉及较少，暂时用不到，可以参考[这篇文章](http://liaoph.com/lvs/)，其中有些错误。

- 基本概念：LVS是一个实现可伸缩网络服务的Linux Virtual Server框架。虚拟的服务器集群系统，提供负载均衡的功能，工作在四层上（TCP/IP层）。
- VIP

### Nginx基础知识

#### Nginx提供的功能

- 配置虚拟主机
- web http服务器，超强的静态文件性能
- 负载均衡
- fastcgi
- 缓存

#### 配置说明

配置文件：/etc/nginx/conf.d/目录下添加xxx.conf文件

配置文件分为块和语句

- 块
  - 全局配置：配置一些通用设置，worker_processes数目。
  - server块：一个虚拟主机，如果要一个nginx有多个虚拟主机，可以配置过个server块。
    - location块：在server内，表示一条网页匹配规则
  - upstream：一组上游服务器，即在nginx后面的web服务器。可以在里面配置轮询的策略。
- 语句，一般在块中
  - listen 监听端口
  - proxy_pass 代理指向的位置
  - proxy_set_header 设置http头
    - 应用：后端ip获取client真是ip，吧client的ip放入header中。
  - root/alias 区分两者

虚拟主机的[配置](http://1567045.blog.51cto.com/1557045/809972)有三种方式：

- 基于ip：一个网卡多个真实IP（ip别名的概念），非常费IP
- 基于域名：一个IP多个域名，根据域名转发，**用的最多**。
- 基于端口：一个IP依据不同端口区分主机，需要手动输入端口，体验不好。

#### 基础教程

- [简明教程](http://xxgblog.com/2015/05/17/nginx-start/)
- [各种配置项说明](https://segmentfault.com/a/1190000002797601)

#### 高级教程

- [nginx进程与架构](http://tengine.taobao.org/book/chapter_02.html)

#### 配置实例

负载均衡的例子：

```nginx
upstream backend { 
    #ip_hash; # 可以配置规则
    # 注意这里的web1，是在host中定义的，不用写http://！！
    server  web1:3000 max_fails=2 fail_timeout=30s;  
    server  web2:3000 max_fails=2 fail_timeout=30s;  
}

server {

  listen 80;
  # server_name example.org;
  access_log /var/log/nginx/nodejs_project.log;
  charset utf-8;

  # public folder static file
  # 这种方式不行，因为很多请求是public/name.js?v=xxx
  # 会把name.js?v=xxx当做文件名，应该有其他办法处理
  # location /public {
  # alias /src/nodeclub/public/;
  # root /src/nodeclub;
  # expires 1d;
  # }

  location ~ .*\.(gif|jpg|jpeg|bmp|png|ico|txt|js|css)$ {   
   root /src/nodeclub;   
   expires 7d;
  }

  location / {
    # 一组服务器，注意写法
    proxy_pass http://backend;
    # 为了后面的web server能获取真实的clientIP
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  }
}
```

### 缓存基础知识

#### 缓存的分类

* 数据库缓存：数据库层的一些临时表或者索引，减少数据库联合查询等复杂操作，用空间换时间的思想。目的：提高数据库性能。
* 应用缓存（server端后台分布式缓存）：对Web应用层做的缓存，一般使用Redis作为存储，对业务数据进行缓存（如缓存Top10列表等），下面重点说这个。
* 前端缓存（cdn等）：客户端浏览器的缓存，CDN静态文件的缓存等外部缓存，参考[这篇文章](http://bbs.qcloud.com/thread-3775-1-1.html)。
* ​

#### 应用层缓存的几个阶段

目的：应用逻辑首先从缓存获取数据，减少db流量，加快速度，提高qps。因此，对于缓存一个重要的指标是命中率。相对而言，对于**缓存的暂时失败是可以容忍的**，只是减少命中率，不会引起业务的失败。

* 第一阶段：单机，单实例的Redis。所有key在一个实例中。
* 第二阶段：分布式缓存，多个缓存实例 — 缓存按key做一致性hash分布。
* 第三阶段：提高每个实例的可用性— master/slave模式备份
* 第四阶段：提高每个实例的QPS — 多级缓存，实现负载均衡。

#### 分布式缓存介绍

* 多个缓存实例，既**把不同缓存内容按照Key分散存储在不同的Redis实例中**。可以提高redis的性能和增大可用内存。同时，当某个实例挂掉，不影响其他缓存。

  * key的分布方式：
    * 取模：添加新实例会动荡
    * **一致性hash**（推荐）：添加新实例，影响小
  * 不足：
    * 某个实例挂了，它的含有的数据的所有请求都穿透到DB，DB负载可能突然升高。
    * 单个实例（某一个）承受QPS达到极限怎么办？对于某些热门数据有这种情况出现。

* master/slave模式备份：对单个实例设立master与slave两个实例之间缓存数据复制，从master读写，没有去slave，再没有去db，然后同步到m/s中，某个挂了影响很小。

  * 引入的问题：
    * 一致性
    * 需要2倍机器—解决方案：两台机器互为Master-Slave复制

  ![master-slave模式](https://fodi.389266.xyz/images/IMG-77703e55891dfd1d.webp)

* 多级缓存：在master/slave模式上再加一层L1Cache。形成多级缓存，L1Cache有多个实例，实现负载均衡。

  * 过程：前端请求首先会随机请求到一组L1缓存，如果这个L1缓存命中则返回，否则再请求到主缓存，如果命中，返回，同时将key-value回种到这个L1缓存中。如果主缓存中没有中，则穿透到DB，返回并同时回种到主缓存及刚才那个L1缓存。L1缓存可以有多组，很好的分担了带宽的压力，并且可以线性扩展
  * 引入的问题：一致性

  ![带有L1的多级缓存](https://fodi.389266.xyz/images/IMG-e8366fc8149e07cc.webp)

* 数据一致性CAP问题：上面的所有情况都会面临数据一致性的问题，针对这个问题有一个理论，既CAP下面三条不可能同时满足：

  * Consistency（一致性）也称作原子性或事务性，表示所有操作在同一时间不可分割。如写入缓存和DB。 对于一致性，在很多场景下**只要达到最终一致性**就可以了。
  * Avaliability（可用性）**必须满足**。
  * Partition tolerance（分区容忍性） 当有一台或几台实例挂掉后致使整个系统不可用，也就是分区容忍性必**须得到满足**。

  因此，针对互联网应用，策略是首先写入**主缓存**（既写入master中，同时使L1全部失效）和**处理队列**，队列写成功，后续的处理（比如同步salve，**写DB**）就可以异步执行。第一步成功后，数据可以被用户读取到，就可以认为写已经完成了。至于后面的存储到DB及容错问题，就交给后面的异步处理程序搞定了。这种不一致只允许在很短的一段时间内存在，最终需要保障缓存和DB达到最终的一致。

其他：http://www.imooc.com/article/3930

### 数据库基础知识

## Docker

### docker基础

* 功能：提供隔离的环境，轻量虚拟机，方便运维部署
* 原理：参考这一些列文章。
  * lxc（Linux Containers）技术
    * [namespace](http://coolshell.cn/articles/17010.html)（环境隔离）：独立的进程空间，文件系统，网卡，权限等。
    * [cgroup](http://coolshell.cn/articles/17049.html)（资源隔离）：限制单个容器cpu，内存等的使用量。
  * 分层文件系统（选择一个即可）：含有只读层，并可以在上面添加可以层。联合文件系统可以对每一层文件系统设置三种权限，只读（readonly）、读写（readwrite）和写出（whiteout-able）。
    * [aufs](http://coolshell.cn/articles/17061.html)
    * [DeviceMapper](http://coolshell.cn/articles/17200.html)
* 应用：
  - 私有paas云
  - 开发测试环境
  - 部署web应用


* 重要概念
  * 仓库
  * 镜像：
    * 特点
      * 分层的，是一个多层结构，一层层文件系统，叫做 Union FS，联合文件系统
      * docker 镜像中每一层文件系统**都是只读的**。
      * **挂载之后**的修改会在新的一个writable层上，既每一层 layer 所保存的修改是增量式的。
      * layer 在镜像间是**共享的**，不同镜像间，对于摘要一样的 layer 只会保存一份
    * 创建方法
      * 修改已有镜像：运行某个镜像(启动容器)，commit提交
      * **使用Dockerfile**，基于某个基础镜像创建
    * 结构：参考[这篇文章](http://blog.csdn.net/jcjc918/article/details/46500031)
  * 容器：运行的镜像
  * 数据卷volume：容器外部的可供一个或多个容器使用的特殊目录，**一般用来放数据库，日志，这些持久化的存储**。
    * 数据卷可以在容器之间共享和重用
    * **对数据卷的修改会立马生效**
    * **对数据卷的更新，不会影响镜像**
    * 数据卷默认会一直存在，即使容器被删除
  * docker中的init进程：与linux的init进程不同，[参考文章](https://yq.aliyun.com/articles/5545)
    * 鼓励：一个容器一个进程(one process per container)。非常适合以单进程为主的微服务架构的应用。
    * 一个容器也可以运行多个进程
    * 每个Container都是**Docker Daemon的子进程**，每个Container进程缺省都具有不同的PID名空间。
    * 当创建一个Docker容器的时候，就会新建一个PID名空间。容器启动进程在该名空间内PID为1。**容器的生命周期和其PID1进程一致**
    * `docker exec`命令可以进入指定的容器内部执行命令。由它启动的进程属于容器的namespace和相应的cgroup。但是**这些进程的父进程是Docker Daemon**而非容器的PID1进程。
    * PID1进程对于操作系统而言具有特殊意义。操作系统的PID1进程是init进程，以守护进程方式运行，是所有其他进程的祖先，具有完整的进程生命周期管理能力。在Docker容器中，PID1进程是启动进程，它也会负责容器内部进程管理的工作。而这也将导致进程管理在Docker容器内部和完整操作系统上的不同。
    * 利用**Supervisor等工具作为PID1进程**是在容器中支持多进程管理的主要实现方式
  * docker中的网络
    * Docker 启动时
      * 会在主机上创建一个 `docker0` 虚拟网桥，实际上是 Linux 的一个 bridge，可以理解为一个软件交换机。
      * Docker 随机分配一个本地未占用的私有网段给 `docker0`。比如典型的 `172.17.42.1`，掩码为 `255.255.0.0`。

    * 创建一个 Docker 容器
      * 创建了一对 `veth pair` 接口（当数据包发送到一个接口时，另外一个接口也可以收到相同的数据包）
      * 这对接口一端在容器内，即 `eth0`
      * 另一端在本地并被挂载到`docker0` 网桥，名称以 `veth` 开头（例如 `vethAQI2QT`）

    * 特点：**一个私有的网络，通过 nat 连接外网，如果要让外网连接到容器中，就需要做端口映射。**

    * 结构图

     ![image.png](https://fodi.389266.xyz/images/IMG-9790f6c962b98ed0.webp)

* 基本操作：可以参考[这个指南](https://www.gitbook.com/book/yeasy/docker_practice)
  * **查看所有镜像**：`docker images`
  * **查看所有容器**（运行的，停止的）：`docker ps`
  * 创建镜像： `docker build . `使用**当前目录**下的Dockerfile文件创建，可选参数 -t 加tag。
  * **运行容器**：`docker run -p 80:80 -d [image ID or NAMES] `
    * `-d` 后台运行
    * `-p hostPort:containerPort` **映射端口**，指定映射到主机某个具体的ip也可以：`ip:hostPort:containerPort`，可以用多个-p来绑定多次。
    * `-v /local_dir:/containr_dir` 指定挂载的数据卷，可以挂载多个，指定权限
    * —name 指定一个名字
    * `-t -i` 选项让Docker分配一个伪终端（pseudo-tty）并绑定到容器的标准输入上， `-i` 则让容器的标准输入保持打开。效果是打开终端。
      * --name xxxx 为容器自定义命名
    * CMD 后面可以接一个启动运行的命令
  * 删除镜像：`docker rmi [image ID or NAMES]`
  * **删除容器**：`docker rm [container ID or NAMES]`
  * **登录容器**：`docker exec -i -t b0c5c63c4630 bash`
  * 查看容器输出 `docker logs [container ID or NAMES]`
  * 查看所有变量 `docker inspect`
  * 数据卷容器：其实就是一个正常的容器，专门用来提供数据卷供其它容器挂载的。
    *  `docker run -d --volumes-from [image ID or NAMES]` 来挂载 dbdata 容器中的数据卷。
    *  所挂载数据卷的容器自己并不需要保持在运行状态
  * 容器互联: 可以让容器之间安全的进行交互。`--link name:alias`，其中 `name` 是要链接的容器的名称，`alias` 是这个连接的别名。**会写入到容器的host中，使用它可以访问到被连接的容器。**
    *  docker run -d  --link db:db [image ID or NAMES]
* 网络配置与管理
  * 容器有自己的内部网络和 ip 地址
  * [如何多台主机的容器互联？](https://yeasy.gitbooks.io/docker_practice/content/cases/container_connect.html)

> **区分镜像与容器**：容器是镜像的实例

### docker部署实践

* Dockerfile编写：Dockerfile包含构建镜像所需的信息
  * 基础镜像信息：第一行`FROM xxx` 基础镜像
  * 维护者信息 `MAINTAINER: docker_user <docker_user at email.com>`
  * `ADD <src> <dest>`命令，复制复制指定的 `src` 到容器中的 `dest`
  * 镜像操作指令`RUN  command`，每一个相当于commit一次
  * **容器启动时执行指令**: `CMD command`
  * CMD与ENTRYPOINT https://segmentfault.com/q/1010000000417103

* docker-compose的使用

  * 什么是：可以方便的**生成多个镜像**和**启动多个docker容器**，并且**组织（compose）他们的关系**
  * 概念
    * 服务（service）：一个应用容器，实际上可以运行多个相同镜像的实例。
    * 项目(project)：由一组关联的应用容器组成的一个完整业务单元。
    * 一个项目可以由多个服务（容器）关联而成，Compose 面向项目进行管理。
  * docker-compose命令
    - docker-compose build . 编译所有的镜像
    - docker-compose up 运行所有的容器（包含编译


*   实践：
    * docker-compose.yml 文件

    ```yaml
     web1: # 第一个web容器
        build: nodeclub/. # 编译该目录下的docker镜像
        volumes: # 挂载数据卷，里面是运行的程序源码
          - "./nodeclub:/src/nodeclub"
        ports: # 指定自身的暴露端口
          - "3000"
        links: # 容器互联
          - "redis:redis" # redis
          - "db:db" #数据库
        command: nodemon -L nodeclub/app.js # 执行的启动命令

    web2: # 第二个web容器
        build: nodeclub/. # 编译该目录下的docker镜像
        volumes:
          - "./nodeclub:/src/nodeclub"
        ports: 
          - "3000"
        links:
          - "redis:redis"
          - "db:db"
        command: nodemon -L nodeclub/app.js
        
    nginx:
        restart: always
        build: nginx/.
        ports:
          - "80:80"  # 端口映射
        volumes_from: # 挂载的数据容器
          - web1
        links:
          - web1:web1 # 容器互联
          - web2:web2

    redis:
        image: redis # 直接使用这个名字的docker镜像

    db:
        image: mongo:3.0.5
        volumes: # 数据库挂载的外部持久化目录
          - "./data/db:/data/db"
    ```

*   web容器Dockerfile文件：生成镜像

    ```dockerfile
      FROM node

      RUN mkdir /src

      # 安装一些工具
      # RUN npm install express-generator -g
      RUN npm install nodemon -g

      # 拷贝package.json,安装依赖模块
      WORKDIR /src
      ADD package.json package.json
      RUN npm install --registry=https://registry.npm.taobao.org

      # 端口
      EXPOSE 3000

      # CMD node app.js
    ```

*   ngnix容器的Dockerfile文件：拷贝配置文件（参考上文），生成nginx镜像

    ```dockerfile
      FROM nginx
      # 删除默认的配置
      RUN rm /etc/nginx/conf.d/default.conf
      # 拷贝新配置
      ADD nodejs.conf /etc/nginx/conf.d/nodejs.conf
    ```

*   本地新建`data`目录作为volume目录，持久化存储数据，保持数据库.


### 学习文章

* [docker的node部署实例](http://dockone.io/article/291)
* [docker-compose一步步部署node实例项目](https://github.com/b00giZm/docker-compose-nodejs-examples)
* [docker中的多进程与init]( https://yq.aliyun.com/articles/5545)
* [docker practice gitbook](https://www.gitbook.com/book/yeasy/docker_practice/details)
* [理解网桥](http://blog.chinaunix.net/uid-18824385-id-107165.html)

## 遗留问题

* docker在生产环境的使用，集群与监控 docker swarm



##
