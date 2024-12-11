---
{"dg-publish":true,"permalink":"/003-Publish/History/版本控制工具git入门实践/","title":"版本控制工具Git入门实践","tags":["技术","Tools"],"created":"2024-12-11T18:09:35.741+08:00","updated":"2024-12-11T18:09:35.742+08:00"}
---



在软件开发中有一些是**通用技能**，无论从事什么项目，C、Java、前段、后台，都必须掌握的（甚至产品设计与UI设计都需要）：

* 操作系统的使用：MacOs/Linux/Win
* 版本协作工具：SVN/**Git**
* 文本工具：Vim/SublimeText/**MarkDown**
* IDE：Eclipse/Idea
* 效率工具：Alfred

这些技能应当优先熟练掌握，今天聊一聊版本控制工具Git。之前一直使用的是SVN，也仅仅停留在工作中使用，甚至不会使用svn命令，只能使用SVN Tortoise，也就是拉代码，解决冲突，提交代码这三个步骤，偶尔拉一下分支（甚至都没有亲自合并过分支。。）。

对于Git使用，说来已久，从一开始在Github上面copy代码（clone），到背git命令，了解git flow开发模型，都是理论大于实践，以至于最近工作的代码使用了Git后，一脸的蒙逼。本质来说没有理解Git的核心思想。

老规矩，下面是学习的一些资料，这次没有找英文的原始资料，因为这方面国内很多资料，就不折腾翻译了：

* [走进git时代系列一、二、三](https://yq.aliyun.com/articles/5843?spm=5176.100239.blogcont7441.8.GnX3p9)：基本够了


* [使用原理视角看 Git](https://segmentfault.com/a/1190000005695097)



## 概念理解

### 核心概念：**分支开发**

这是Git中最最重要的一个概念，也是与SVN等其他工具最核心的不同。

在Git的开发中，分支是一个轻量级的概念，因此新建分支，合并分支，删除分支在开发中十分常见，我们开发时应当遵循这样的基于分支的开发流程：

1. 当有一个新需求、bug需要开发时，从origin/master新建一个分支,feature01/bugfix01。
2. 切换到该分支上，开发。。修复a1，add，修复a2，add。。。
3. 开发完成后commit（一般保持一个commit记录）
4. git fetch origin master 从origin master更新代码(从网络更新代码到FETCH_HEAD,可以更新到master)
5. rebase到当前分支上
6. push



### 理解远程分支与本地分支

* 本质上，他们都是分支！


* 本地分支中的默认分支`master`也是分支，它是以`origin/master分支`初始化的一个普通分支。


* **分支本质是n个commit的list**。如果本地`master`与`origin/master`同步（不在本地master上commit，修改可以，不要提交），更新之后，那么master与origin/master就是一样的。

一个新的项目clone后有哪些分支：

![all branch](http://fodi.limuzhi.us.kg/images/IMG-af5ae175b5228177.webp)

* `master` 当前的默认分支，第一次取下来后与`origin/master `指向相同，但是后续可以在上面开发


* `origin/master` 远程分支，内容是**最后一次从远程origin获取的所有commit**的分支，无法修改数据，本地数据，但含义是

  > `remotes/origin/master`和`origin/master`的指向是相同的
  >
  > - `master` is a local branch
  > - `origin/master` is a remote branch (which is a *local copy* of the branch named "master" on the remote named "origin")

哪些指针：

* `HEAD`
* `FETCH_HEAD `
* `origin/master/HEAD` 指向`origin/master`的提交，与`HEAD`区分


### 理解本地的三个工作区

#### 1. 工作目录workspace

这个可以直接理解成**普通的文件夹，文件**， 这几个角度：

* 工作区全局只有一个，没有多个备份，它不会保存完整的状态。

* 不commit就切换分支会污染工作目录！（不commit，**且有冲突**的情况下，Git不会让你切换的，**如果没有冲突，则会合并！**）

  > error: Your local changes to the following files would be overwritten by checkout:
  >
  > **建议：切换分支前先commit**，养成commit的习惯

* 从其他分支合并，或者从网络fetch后merge（pull），都是合入工作区。因此要注意在此之前add/commit操作。

  > 泛化的理解：Git的所有操作都是先针对工作区修改，因此在执行例如合并之类的操作必须要add或者commit

#### 2. 暂存区 Stage/Index

暂存区是Git维护的，用户手动添加，暂存区只对当前分支有效，

* 工作区全局只有一个，没有多个备份，它不会保存完整的状态。
* 因此，在切换分支时，暂存区也会被污染！具体情况与上面第二点相同（不commit，**且有冲突**的情况下，Git不会让你切换的，**如果没有冲突，则会合并！**）。
* 与工作区不同，在从合入分支时，它不受影响，这也是暂存区的作用。

#### 3. 本地仓库

**分支本质是n个commit的list**。

* 本地仓库中含有n个分支。包括一个默认的本地master（映射到origin/maser）和其他本地分支（可能没有映射，也可能映射到远程对于的分支）。

* 切换分支时，只是指针切换，仓库内容不受影响，**只会更新工作目录（文件的删除添加）与暂存区（一般是清空，如果暂存区非空，且不冲突也会保留）**

* 合并操作，会修复被合入的分支内容（某个commit的提交内容，或者commit的list）




### 工作目录中文件的三种状态

* 已提交（committed）：该文件被安全地保存在了本地数据库

* 已暂存（staged）：把已修改的文件放下下次保存的清单中

* 已修改（modified）：修改了某个文件，但还没有保存，此外还有从没有add过的新文件，**未追踪untracked**

  参考下图学习：

  ![git status](http://fodi.limuzhi.us.kg/images/IMG-5bac7993b4e8db8c.webp)

几个关注点：

* 红色表示在工作目录的文件，绿色表示在暂存区的文件
* `Changes to be committed:`下的`new file`表示，暂存区与仓库对比，他们是新文件（如果有`modified`表示，暂存区与仓库对比是修改的）
* `Changes not staged for commit:`下的`modified`表示，工作区与暂存区/仓库对比是修改的
* `Untracked files:`下的是没有加入过暂存区的内容（在暂存区/仓库找不到对比对象）
* 注意有两个hell3.txt，一个是工作目录的，一个是暂存区的，这说明**加入暂存区后该文件被修改过。**



### HEAD

`HEAD`是一个指针，指向的是**本地仓库**中**当前分支**的一个commit记录！该commit是当前正在工作的commit。
什么时机HEAD会移动：

* git checkout后指向新分支的commit（跨分支）
* git commit后HEAD移动指向的commit（同一分支，向前）
* merge，rebase，fast-forward

注意：它指向的内容与工作目录和暂存区可能都不同，因此可以用HEAD来执行恢复操作。


### fast-forward merge

快进模式 一个常用概念，指的是谁forward呢？是**HEAD的快速向前移动**。

出现的情况：

* merge完成时，被合入分支的最后一个commit在待合入分支之中，移动HEAD到最后一个commit。如下图所示：

  ![fast forward](http://fodi.limuzhi.us.kg/images/IMG-9b834d017fbfa2b5.webp)

* rebase完成后，commit记录合入后，HEAD指针直接向前移动多个commit。

总之，当前分支是master，执行`git merge branch1`。master分支最好的commit是目标分支branch1的祖先commit节点时，会发生Fast-forward的merge。

> 有时为了时每一次merge都有记录，要禁用fast-forward,需要使用`git merge —no-ff`命令合并。

### merge/rebase/cherry-pick 区别与使用场景

- rebase：一般只用在本地分支上，把远程分支rebase过来。


- merge：一般用在远程分支上，把本地分支merge到远程主干上。
- cherry-pick：从其他分支拿(pick)一个commit到某个分支

顺序，先rebase orgin 再切换到主分支 merge 过去，前向合并(fast-forward merge，分支是目标分支的祖先commit节点时，会发生Fast-forward的merge)

参考这篇文章理解：http://pinkyjie.com/2014/08/10/git-notes-part-3/

### reset与revert

这两个都是后悔药，区别如下

* reset：使用撤销commit的方式恢复，使用在本地分支的恢复上（推送之前）。


* revert：使用添加commit的方式恢复，用在远程分支的恢复上。原因参考『场景分析』



### stage与stash

* stage，是暂存区，既我们git add之后存储的地方
* stash，是一个独立的存储区域，可以存放保存的文件，
  * `git stash`：存储
  * `git stash apply`： 还原





## 场景分析

### commit多次，只保留一个commit，

两种方案：

* commit修改，使用git commit —amend 提交commit，会在之前一个commit的基础上修改（记录内容），不增加新的commit节点。
* commit合并，使用git rebase 的交互模式来合并多个commit：参考：http://zerodie.github.io/blog/2012/01/19/git-rebase-i/


### 保存某个文件，要跨分支使用它。

`git stash` 命令可以满足这个需求，相当于一个存储箱。

### commit错分支

cherry-pick：
一种常见的场景就是，比如我在A分支做了几次commit以后，发现其实我并不应该在A分支上工作，应该在B分支上工作，这就需要将这些commit从A分支复制到B分支去了，这时候就需要cherry-pick命令了。

参考：http://pinkyjie.com/2014/08/10/git-notes-part-3/


### 各种场景恢复的使用reset

下面三种恢复的越来越多。

* `git reset —soft xx`： 仅仅取消commit&&移动HEAD指针到xx，不修改工作目录和暂存区。这个模式的效果是，自从<commit>以来的所有改变都会显示在git status的**"Changes to be committed"**中。使用场景：取消commit，一直到某个commit（xx），但是这些修改恢复到暂存区中。

* `git reset xx`：取消commit&&移动HEAD指针到xx，并且恢复暂存区，但是恢复工作目录。这个模式是默认模式，这个模式的效果是，工作目录中的文件的修改都会被保留，不会丢弃，但是也**不会**被标记成"Changes to be committed"。使用场景：取消commit，一直到某个commit（xx），但是这些修改恢复到工作目录中。

  > 对比上一个就是少了git add 的过程

* `git reset —hard xx`： 取消commit&&移动HEAD指针到xx，恢复工作目录&&暂存区。使用场景：恢复到xx commit，**丢弃从xx以来的所有修改。**



### 已经推送到远程分支，但是向回退某个commit

这个使用revert命令，不要使用reset取消commit方式恢复，而是应当使用revert添加commit的方式恢复。

原因如下

- 当然如果你想撤销你的修改，可以通过git reset 或 git revert ，但当你的commit已经push到远端，被别人pull了下来， 再reset push 的话，别人再pull 就会出现错误，因为这个commit 节点回退到了你本地的缓存区，不在版本系统内，会很麻烦。
- 所以这种情况下需要使用 git revert ，**它是撤销某次操作，此次操作之前和之后的commit和history都会保留，并且把这次撤销作为一次最新的提交。**将需要revert的版本的内容再反向修改回去，版本会递增（添加新的commit记录），不影响之前提交的内容，别人pull的时候不会出问题，这个很重要。



## 命令分析

我们使用的很多命令都是省略了一下参数的，而使用了默认值，有些情况我们也要认识他们，知道含义。

### git push <repository> <src>:<dst>

如我们常用:

* git push
* git push origin master

origin指定了你要push到哪个remote，master其实是一个“refspec”，正常的“refspec”的形式为”+<src>:<dst>”，**冒号前表示local branch的名字，冒号后表示remote repository下 branch的名字。**注意，如果你省略了<dst>，git就认为你想push到remote repository下和local branch相同名字的branch。

### git fetch <repository> <src>:<dst>

* git fetch == **git fetch origin master:master** == **fetch到本地master分支**
* git fetch origin master == **git fetch origin master:** == **fetch到本地FETCH_HEAD上****

与push命令类似，但是注意：**冒号前表示remote repository下 branch的名字，冒号后表示local branch的名字。**上面第一个命令是第二个命令不是fetch到本地master分支，而是FETCH_HEAD上

>  参考：
>
>  * [如何配置默认值&&git getch 与git fetch origin master 区别](http://stackoverflow.com/questions/11892517/git-fetch-vs-git-fetch-origin-master-have-different-effects-on-tracking-branch)
>  * [什么是FETCH_HEAD](http://stackoverflow.com/questions/9237348/what-does-fetch-head-in-git-mean)

### git pull  <repository> <src>:<dst>

* [git pull ==git fetch+git merge FETCH_HEAD]( https://ruby-china.org/topics/4768)

* git pull origin master 从网络获取origin分支的master，合入当前分支（没指定<dst>）

* git pull origin/master 本地操作，合并最后一次获取的远程master分支到当前分支（没指定<src>:<dst>）

  > 参考：http://stackoverflow.com/questions/2883840/differences-between-git-pull-origin-master-git-pull-origin-master

重难点理解

- 理解分支开发模型
- 理解工作目录，合理使用commit
- 暂存区，Stash 的运用场景
- reset 和 revert的差别
- merge，rebase， check-pick 的差别和运用
- [理解 master origin/master origin](http://stackoverflow.com/questions/18137175/in-git-what-is-the-difference-between-origin-master-vs-origin-master)





