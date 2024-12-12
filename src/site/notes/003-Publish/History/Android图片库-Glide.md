---
{"dg-publish":true,"permalink":"/003-Publish/History/Android图片库-Glide/","title":"Android图片库--Glide Wiki中文翻译","tags":["技术","Android","主框架"],"created":"2024-12-11T18:09:35.709+08:00","updated":"2024-12-11T18:09:35.710+08:00"}
---


## 序
最近，一直在捣鼓Glide这个图片库，确切地说应当叫『媒体框架』。为了支持Gif播放，项目中使用Glide代替了Universal Image Loader。起初我们打算使用的是Fresco方案，最终因为Lib体积的问题放弃。万万没想到apk的size这么敏感，国外很多推广服务是根据apk体积收费的，3M一个门槛，5M一个门槛。fresco引入之后即使proguard、7zip，apk的大小也增加了1M+，多平台的os库是个大头，第三世界的应用市场还不支持按平台分发😓。。
Glide的使用还不深入，仅仅停留在Api的范畴，为了解决bug，跟踪了代码，也是云里雾里。不得不说，Glide源码的设计十分NB，面向接口的理念贯彻的很彻底，是一个学习设计的好demo。
GitHub的README作为最简单的入门足够了，但是遇到具体问题还是要理解一些设计思想的，GitHub上的Wiki有一份不错的文档，可惜木有中文，正好学习的过程中翻译一下。
在看wiki之前，可以先看这两篇中文入门，了解基本的用法。

* [Google推荐的图片加载库Glide介绍](http://www.jcodecraeer.com/a/anzhuokaifa/androidkaifa/2015/0327/2650.html)
* [Glide 一个专注于平滑滚动的图片加载和缓存库](http://www.jianshu.com/p/4a3177b57949)

本文是基于**Glide 3.x**正式版的[Wiki文档](https://github.com/bumptech/glide/wiki)的翻译（原文编辑日期：2015-2-9,18 revision，当时Glide release的版本号3.6.1）。

翻译Version：1.1 简单校对

### 名词解释
Glide中有一部分单词，我不知道用什么中文可以确切的表达出含义，用英文单词可能在行文中更加合适，还有一些词在Glide中有特别的含义，我理解的可能也不深入，这里先记录一下。

* View： 一般情况下，指Android中的View及其子类控件（包括自定义的），尤其指ImageView。这些控件可在上面绘制Drawable
* Target： Glide中重要的概念，目标。它即可以指封装了一个View的Target（ViewTarget），也可以不包含View（SimpleTarget）。
* Drawable： 指Android中的Drawable类或者它的子类，如BitmapDrawable等。或者Glide中基础Drawable实现的自定义Drawable（如GifDrawable等）
* Request - 加载请求，可以是网络请求或者其他任何下载图片的请求，也是Glide中的一个类。
* Model：数据源的提供者，如Url，文件路径等，可以从model中获取InputStream。
* Signature：签名，可以唯一地标识一个对象。
* recycle()：Glide中Resource类有此方法，表示该资源不被引用，可以放入池中（此时并没有释放空间）。Android中Bitmap也有此方法，表示释放Bitmap占用的内存。

## 目录
* 主目录
* 缓存机制与缓存失效
* Glide配置
* 自定义Target
* 调试与错误处理
* 使用Glide下载自定义大小图片
* 集成库-与其他库相整合
* 在后台线程中加载与缓存数据
* Glide中的资源复用
* 快照（Snapshots）
* 图形变换（Transformations）

---------

## 主目录

### 报告问题
如果你有任何问题，可以在[Github上提出](https://github.com/bumptech/glide/issues/)或者发送e-mail到我们的[邮件列表](https://groups.google.com/forum/#!forum/glidelibrary)，也可以在IRC(Internet Relay Chat 网络中继聊天？)频道上联系我们：[irc.freenode.net#glide-library](http://webchat.freenode.net/?channels=glide-library)。

### 3.0版本的新特性
* 支持Gif动画的解码 - 与加载图片相同，只要调用Glide.with(...).load(...)，如果你加载的图片是一个可以播放的Gif，Glide会自动加载它并显示在一个自定义的Drawable上（注：GifDrawable）。此外，你还可以控制的更多，比如

	```java
// 你想加载Gif为一张静态图片
Glide.with(context).load(...).asBitmap()。
// 或者你想只有加载对象是Gif时才能加载成功
Glide.with(context).load(...).asGif()。
```

* 本地视频播放技术 - 除了解码Gif，Glide也能解码你设备上的视频（video）。使用方法和加载gif相同，Glide支持所有Android可以直接解码的视频。
* 支持缩略图加载 - 有时我们希望减少用户等待的时间又不想牺牲图片的质量，我们可以同时加载多张图片到一个View中，先加载缩略图（只有view的1/10大小），然后再加载一个完整的图像覆盖在上面。使用下面的代码

	```java
Glide.with(yourFragment).load(yourUrl).thumbnail(0.1f).into(yourView)
```

当然，你也可以传入一个Request到`.thumbnail()`函数中作为参数。

* 与生命周期集成 - 加载请求现在会自动在onStop中暂停在，onStart中重新开始。为了节约电量，Gif动画也会在onStop中自动暂停。此外，当设备的连接状态改变时，所有失败的请求会自动重试，确保Glide不会因为临时性的连接问题，导致请求永远失败。
* 转码 - 除了解码资源，Glide的`.toBytes()`和`.transcode()`方法允许你在后台正常地获取、解码、变换一张图片。你还可以在这些调用中把图片转码成更有用的格式，比如，上传一张大小为250*250像素的用户头像的图片bytes数据，代码如下

	```java
Glide.with(context)
    .load(“/user/profile/photo/path”)
    .asBitmap()
    .toBytes()
    .centerCrop()
    .into(new SimpleTarget<byte[]>(250, 250) {
        @Override
        public void onResourceReady(byte[] data, GlideAnimation anim) {
            // 在此处，将bytes数据传入后台线程，再上传他们
        }
    });
```

* 动画 - Glide3.X支持『淡入淡出』动画（`.crossFade()`）和view的属性动画(`.animate(ViewPropertyAnimation.Animator)`)。此外，还有Glide2.0就支持的android view动画。
* 支持 OkHttp 和 Volley - 现在你可以选择用OkHttp、Volley或者默认的HttpUrlConnection作为网络栈。OkHttp和Volley可以通过添加对应的集成库(integration library)和注册相应的`ModelLoaderFactory`来引入。具体查看ReadMe文件。
* 其他 - 可以使用Drawable对象作为加载中的占位图、请求优先级、覆盖自定义宽和高、可以缓存变换后的缩略图或者缓存原始文件

### 从2.0迁移到3.0
* 将所有的`Glide.load()`替换为`Glide.with([fragment/activity/context]).load()`。
* 将所有的自定义的加载调用`Glide.load(url).into(new SimpleTarget(){ ... }).with(context)`替换成`Glide.with(context).load(url).into(new SimpleTarget(width, height) { ... })`。

### 特性
除了3.0引入的新功能，Glide继承了2.0的所有功能：

* 后台图片加载
* 如果你使用了listview的复用机制，那么Glide会自动取消作业（job）
* 内存和磁盘缓存
* Bitmap和资源池来减少内存抖动
* 支持任意的图像变换

---------

## 缓存机制与缓存失效
缓存失效是一个比较复杂的话题，理想情况下，你尽可能不要考虑这个问题。这一节主要是让你粗略的了解一下Glide中cache的key是如何生成，以及提供一些关于如何合理利用缓存的提示。
### 缓存的key
`DiskCacheStrategy.RESULT`磁盘缓存策略（注：我们配置的一种磁盘缓存策略）使用的key由以下四个主要部分组成：

* DataFetcher的方法`getId()`返回的字符。典型地，DataFetcher仅仅返回由数据Model的`toString()`方法得到的值。所以，如果Model是一个**URL**，那么会返回URL的字符串，如果Model是是一个文件，那么会返回文件的**路径**。。。
* 宽和高。如果你调用过override(width,height)方法，那么就是是它传入的值。没有调用过，默认是通过**Target**的`getSize()`方法获得这个值。
* 各种编码器、解码器的`getId()`方法返回的字符串。这些编码器、解码器用于加载和缓存你的图片。仅有哪些堆bytes数据有影响的编码器、解码器才会有这些`id`值。比如，你只有一个将bytes数据写入磁盘的编码器，那么它就没有id值，因为不管怎样它都不会修改数据。
* 可选地，你可以为图片加载提供签名(Signature)，请看下面的缓存失效部分。

所有的这些key，以特定的顺序计算出hash值，并将这个值作为保存图片到磁盘上的唯一且安全的文件名。

### 缓存失效
由于文件名是hash值，没有简便的方式删除磁盘上某个特定url或者文件路径的所有缓存文件。如果仅仅缓存原文件，问题可能还比较简单，但是Glide会缓存缩略图，各种变换后的图片，所有这些缓存都会产生新文件。跟踪和删除所有文件是十分困难的。

### 自定义缓存失效
通常情况下改变缓存的标志（key）是困难的。Glide提供了`signature()`API来混入其他数据，帮助你控制缓存的key。签名（Signature）对于多媒体或者有版本信息的内容来说都很有用。

* 媒体库内容 - 对于媒体库内容，你可以使用Glide的`MediaStoreSignature`类作为签名， MediaStoreSignature类允许你混入文件修改时间、mime类型、媒体库的方向(orientation?)这些值作为缓存的key，这三个值足以让你可靠的捕捉到任何修改和更新，使你可以缓存媒体库的缩略图.?
* 文件 - 你可以使用`StringSignature`混入文件修改时间
* url - 虽然使url失效最好的方式是当内容变化时，服务器修改url并更新客户端，但是你也可以用`StringSignature`混入任意的元数据（如版本号）来使缓存失效。

使用String Signature加载数据很简单：

```java
Glide.with(yourFragment)
    .load(yourFileDataModel)
    .signature(new StringSignature(yourVersionMetadata))
    .into(yourImageView);
```

媒体库Signature可以直接使用从MeidaStore(注：一个android api）获得的数据

```java
Glide.with(fragment)
    .load(mediaStoreUri)
    .signature(new MediaStoreSignature(mimeType, dateModified, orientation))
    .into(view);
```

你还可以通过实现`key`接口来自定义签名，确保实现了`equals()`, `hashCode()`和`updateDiskCacheKey()`这几个方法

```java
public class IntegerVersionSignature implements Key {
    private int currentVersion;
 
    public IntegerVersionSignature(int currentVersion) {
         this.currentVersion = currentVersion;
    } 
 
    @Override 
    public boolean equals(Object o) {
        if (o instanceof IntegerVersionSignature) {
            IntegerVersionSignature other = (IntegerVersionSignature) o;
            return currentVersion = other.currentVersion;
        } 
        return false; 
    } 
 
    @Override 
    public int hashCode() { 
        return currentVersion;
    } 
 
    @Override 
    public void updateDiskCacheKey(MessageDigest md) {
        messageDigest.update(ByteBuffer.allocate(Integer.SIZE).putInt(signature).array());
    } 
} 
```
请牢记：为了避免性能下降，请在后台线程中批量加载版本元数据(metaData，注：一般查询数据库获得)，只有这样，才能确保当你想加载图片的时候，这些值是可用的。

如果这些方法都失败了，比如，你不能改变标识符，也不能跟踪一个合理的版本号。你还可以使用`diskCacheStrategy()`和`DiskCacheStrategy.NONE.`来完全关闭磁盘缓存。

---------

## 配置

### 懒加载配置
从Glide3.5开始，你可以使用`GlideModule`接口来懒加载配置Glide以及注册组件（如`ModelLoaders`)，这些配置将会在第一个Glide请求发起的时候被调用。

### 创建一个GlideModule
为了使用和注册GlideModule，首先需要实现该接口，加入你的配置和组件。

```java
package com.mypackage; 
 
public class MyGlideModule implements GlideModule { 
    @Override public void applyOptions(Context context, GlideBuilder builder) {
        // Apply options to the builder here. 
    } 
 
    @Override public void registerComponents(Context context, Glide glide) {
        // register ModelLoaders here. 
    } 
} 
```

然后，添加你的实现类到`proguard.cfg`文件中，让你的module可以被反射调用到。这个性能开支很少，因为每个module只会在Glide第一次请求的时候实例化一次。

```xml
-keepnames class com.mypackage.MyGlideModule
# or more generally:
#-keep public class * implements com.bumptech.glide.module.GlideModule
```

最后，添加meta-data标记到`AndroidManifest.xml`，那样Glide才能找到它。

```xml
<manifest ...>
    <!-- ... permissions -->
    <application ...>
        <meta-data
            android:name="com.mypackage.MyGlideModule"
            android:value="GlideModule" />
        <!-- ... activities and other components -->
    </application>
</manifest>
```

你可以实现任意个`GlideModule`，但是每一个都要添加到`proguard.cfg`，而且每一个GlideModule都要在manifest有自己的meta-data标记。

### Library工程
Library工程可能含有一个或多个`GlideModule`。当我们使用Gradle（或者任何支持manifest合并的构建工具）构建app时，如果我们所依赖的Library工程的manifest中含有module，那么这些module会自动并入到app中。如果构建工具不支持manifest合并，那么这些library工程中的module必须手动添加到你app的manifest中。

### GlideModule冲突
虽然Glide允许每个app注册多个`Glidemodule`，但是不会以某种特定的顺序调用这些module。所以，如果你的app中多个`GlideModules`或者依赖的library工程中有多个`GlideModules`，你必须负责避免他们之间的冲突。
如果冲突不可避免，app应该定义自己的默认module，这个module需要手动解决这些冲突，提供所有Library工程所需要的依赖。使用Gradle的开发者，或者使用manifest合并的开发者，可以通过下面的标签来排除冲突的module。

```xml
<meta-data android:name=”com.mypackage.MyGlideModule” tools:node=”remove” />
```

### 全局配置
你可以配置一些作用于所有请求的全局性配置项。请使用`GlideModule#applyOptions`方法中（注：作为参数）提供给你的`GlideBuilder`来配置。本节代码示例中的`builder`就是一个GlideModule对象。

### 磁盘缓存
你可以使用`GlideBuilder`的`setDiskCache()`方法设置磁盘缓存的位置、大小（最大值）。你也可以使用`DiskCacheAdapter`彻底关闭缓存，或者自己实现`DiskCache`接口来换掉默认实现。磁盘缓存由`DiskCache.Factory`接口的实例在后台线程中创建，使用后台线程可以避免在严格模式中出现问题。
Glide默认使用`InternalCacheDiskCacheFactory`类建立磁盘缓存。这个内部缓存工厂（internal cache factory）会把磁盘缓存放到应用程序的内部缓存目录中，并且设置最大空间是250MB，使用内部缓存的目录而不是SD卡的外部目录意味着其他应用程序无法访问到你下载的图片。具体请查看Android的[存储选项相关文档](http://developer.android.com/intl/zh-cn/guide/topics/data/data-storage.html#filesInternal)。

#### 大小
使用`InternalCacheDiskCacheFactory`设置磁盘缓存大小

```java
builder.setDiskCache(
  new InternalCacheDiskCacheFactory(context, yourSizeInBytes));
```

#### 位置
也可以设置磁盘缓存位置
你可以使用`InternalCacheDiskCacheFactory `来把你的磁盘缓存放到应用程序私有的内部存储目录中：

```java
builder.setDiskCache(
  new InternalCacheDiskCacheFactory(context, cacheDirectoryName, yourSizeInBytes));
```
还可以用`ExternalCacheDiskCacheFactory `来把你的磁盘缓存放到sd卡的公共缓存目录上。

```java
builder.setDiskCache(
  new ExternalCacheDiskCacheFactory(context, cacheDirectoryName, yourSizeInBytes));
```
如果你想用其他自定义的路径，可以用`DiskLruCacheFactory`类的构造函数来实现。

```java
// If you can figure out the folder without I/O: 
// Calling Context and Environment class methods usually do I/O. 
builder.setDiskCache( 
  new DiskLruCacheFactory(getMyCacheLocationWithoutIO(), yourSizeInBytes)); 
 
// In case you want to specify a cache folder ("glide"): 
builder.setDiskCache( 
  new DiskLruCacheFactory(getMyCacheLocationWithoutIO(), "glide", yourSizeInBytes)); 
 
// In case you need to query the file system while determining the folder: 
builder.setDiskCache(new DiskLruCacheFactory(new CacheDirectoryGetter() { 
    @Override public File getCacheDirectory() {
        return getMyCacheLocationBlockingIO(); 
    } 
}), yourSizeInBytes); 
```
注意：请牢记，写死任何值都不是个好主意，尤其是目录的路径，因为自Android4.2开始，支持单设备多用户。

如果你想完全控制缓存的创建，可以自己实现`DiskCache.Factory `接口，使用`DiskLruCacheWrapper`可以在你想要的位置创建一个新的缓存。

```java
builder.setDiskCache(new DiskCache.Factory() { 
    @Override public DiskCache build() { 
        File cacheLocation = getMyCacheLocationBlockingIO();
        cacheLocation.mkdirs();
        return DiskLruCacheWrapper.get(cacheLocation, yourSizeInBytes);
    } 
}); 
```

### 内存缓存和缓存池
`GlideBuilder`类允许你设置内存缓存大小，而且可以实现自定义的`MemoryCache`和`BitmapPool`。

#### 大小
默认大小是由`MemorySizeCalculator`类决定的。MemorySizeCalculator类会考虑该设备的屏幕大小、可用内存来计算出一个合理的默认值。如果你想调整这个默认值，请构建自己的实例。

```java
MemorySizeCalculator calculator = new MemorySizeCalculator(context);
int defaultMemoryCacheSize = calculator.getMemoryCacheSize();
int defaultBitmapPoolSize = calculator.getBitmapPoolSize();
```
如果你想在应用的某个阶段动态调整Glide的内存占用，你可以选择一个`MemoryCategory`并使用`setMemoryCategory()`方法传入Glide中：

```java
Glide.get(context).setMemoryCategory(MemoryCategory.HIGH);
```

#### 内存缓存
Glide的内存缓存会在内存中持有资源，它的作用是，我们可以不进行IO操作而快速获得可用资源。
你可以使用`GlideBuilder`的`setMemoryCache()`方法设置大小，或者设置你关于内存缓存的自定义实现（自定义类）。`LruResourceCache`类是Glide的默认实现。你可以通过`LruResourceCache`的构造函数来配置内存占用的bytes的最大值。

```java
builder.setMemoryCache(new LruResourceCache(yourSizeInBytes));
```

#### Bitmap池
Glide的Bitmap池主要作用是，可以让各种尺寸的Bitmap被复用，可以可观地减少由垃圾回收引起的内存抖动。在解码图片时，需要给像素数组分配空间，这会触发垃圾回收。
你可以使用`GlideBuilder`的`setBitmapPool()`方法设置大小，或者设置你关于Bitmap池的自定义实现，`LruBitmapPool`类是Glide的默认实现。LruBitmapPool类使用了LRU算法维护最近最常使用的Bitmap的尺寸。你可以通过`LruBitmapPool`的构造函数配置内存占用的bytes的最大值。

```java
builder.setBitmapPool(new LruBitmapPool(sizeInBytes));
```

### Bitmap格式
`GlideBuilder` 类也允许你配置一个App全局使用的Bitmap的Config属性。
Glide默认使用RGB_565，因为它每个像素只需要2bytes（16bit）的空间，它仅需要高质量图片（既系统默认的`ARGB_8888 `）一半的内存空间。但是，这对于某些图片这可能会出现『条带』的问题，而且它也不支持透明度。
如果在你的应用中『条带』是一个重要的问题，或者你需要尽可能好的图片质量，你可以使用`GlideBuilder`的`setDecodeFormat`方法设置DecodeFormat.ALWAYS_ARGB_8888作为首选配置。

```java
builder.setDecodeFormat(DecodeFormat.ALWAYS_ARGB_8888);
```

---------

## 自定义目标（Targets）
除了可以加载图像，视频剧照，gif动画到View中，你还可以加载他们到实现了Target接口的自定义目标中。

### SimpleTarget
如果你只是想加载一个Bitmap，并不想直接展示给用户而是有一些特殊用途，比如在通知栏中显示或者作为头像上传。
Glide也可以做到。
SimpleTarget为Target接口提供了大部分的默认实现。你可以专注于处理加载的结果。
为了使用SimpleTarget，你需要在它的构造函数中提供你要加载的资源的宽和高（单位像素），你还需要实现` onResourceReady(T resource, GlideAnimation animation)`方法。
一个典型的使用SimpleTarget的例子如下：

```java
int myWidth = 512;
int myHeight = 384;
 
Glide.with(yourApplicationContext)) 
    .load(youUrl) 
    .asBitmap() 
    .into(new SimpleTarget<Bitmap>(myWidth, myHeight) {
        @Override 
        public void onResourceReady(Bitmap bitmap, GlideAnimation anim) {
            // Do something with bitmap here. 
        } 
    }; 
```

#### 一些警告
正常情况下，你加载一个资源会把他们放到view中。当你的Activity或者Fragment被pause或者destroy时，Glide会暂停或取消加载，确保你不会加载那些根本不会显示的资源。
可是当我们使用SimpleTarget的时候，这可能并不是我们希望的行为。所以，当你调用`Glide.with(context)`的时候，你可以传入Application的context，而不是传入Activity或者Fragment。
此外，考虑到长时间的加载操作可能导致内存泄漏，请考虑使用静态内部类，而不是匿名内部类。

### ViewTarget
如果你想加载一个图片到View中，但是你想观察或者覆盖Glide的默认行为。你可以覆盖ViewTarget或者它的子类。
当你想让Glide来获取view的的大小，但是由自己来启动动画和设置资源到view中，ViewTarget是个不错的选择。如果你要加载一个图片到ImageView之外的自定义view中，那么ImageViewTarget或者它的子类就不能满足你的要求，此时继承ViewTarget就特别合适。
你可以静态的定义一个ViewTarget的子类，或者传递一个匿名内部类到你的加载调用里：

```java
Glide.with(yourFragment)
    .load(yourUrl)
    .into(new ViewTarget<YourViewClass, GlideDrawable>(yourViewObject) {
        @Override
        public void onResourceReady(GlideDrawable resource, GlideAnimation anim) {
            YourViewClass myView = this.view;
            // Set your resource on myView and/or start your animation here.
        }
    });
```
注意，如果你想指定加载Bitmap还是GifDrawable，请在`.load(yourUrl)`调用后面直接添加`.asBitmap()`或者`.asGif()`，同时将ViewTarget的类型参数`GlideDrawable`换成对应加载的类型。
为了更多控制，你也可以在Target实现`LifecycleListener`回调，`onStart()`、`onStop()`或者`onDestroy()`会和你view所在的fragment的生命周期保持同步。

### 覆盖默认行为
如果你只想观察不想修改Glide的默认行为，你可以继承任何一个Glide对ImageViewTargets的默认实现。

* GlideDrawableImageViewTarget - 默认的Target，用于正常的加载和`asGif()`。
* BitmapImageViewTarget - 当使用`asBitmap()`加载时，使用的默认Target。

只有你在每个方法里面调用`super()`，将会保留默认的行为，同时还可以添加一些你希望的功能。

例如，想要生成一个[调色板](http://chris.banes.me/2014/07/04/palette-preview/)，你可以这样做。

```java
Glide.with(yourFragment) 
    .load(yourUrl) 
    .asBitmap() 
    .into(new BitmapImageViewTarget(yourImageView)) { 
        @Override 
        public void onResourceReady(Bitmap bitmap, GlideAnimation anim) {
            super.onResourceReady(bitmap, anim);
            Palette.generateAsync(bitmap, new Palette.PaletteAsyncListener() {  
                @Override 
                public void onGenerated(Palette palette) {
                    // Here's your generated palette 
                } 
            }); 
        } 
    }); 
```
虽然这个例子还不错，但是，通常情况下，我不推荐用这个方式生成调色板。请查看Glide的 `ResourceTranscoder` 接口和`.transcode()`方法，考虑返回一个包含Bitmap和调色板的自定义资源。调色板可在在后台线程生成。？？？？？更多内容会在以后推出。。。

---------

## 调试和错误处理
Glide在加载过程中出现异常默认情况下不会打日志。Glide为你提供了两种方式查看和处理这些异常。

### 调试
仅仅为了查看异常的话，你可以为`GenericRequest`类打开Debug日志。这个类处理所有媒体资源的加载响应（response）。你可以在命令行里运行：

```xml
adb shell setprop log.tag.GenericRequest DEBUG
```
想要包括详细的请求时序信息，你可以把`DEBUG`缓存`VERBOSE`。

关闭日志使用：

```xml
adb shell setprop log.tag.GenericRequest ERROR
```

### 调试[工作流](https://docs.google.com/drawings/d/1KyOJkNd5Dlm8_awZpftzW7KtqgNR6GURvuF6RfB210g/edit)
为了查看内部引擎（engine）如何以及何时查找到资源，你可以打开日志：

```xml
adb shell setprop log.tag.Engine VERBOSE
adb shell setprop log.tag.EngineJob VERBOSE
adb shell setprop log.tag.DecodeJob VERBOSE
```
打开这个有助于你找出为什么某个资源没有从内存缓存中加载，为什么请求再次从外部的url下载数据。这也可以帮助你了解：如果想命中磁盘缓存，什么样的参数需要匹配。启用`DecodeJob`日志也可以帮助你去定位自定义transformation/decoder/encoder相关的问题。

### 监听请求-RequestListener
虽然启用debug日志很简单，但是只有在你可以连接到设备时才能这样干。为了把Glide集成到已有的更专业的错误日志系统。你可以使用`RequestListener`类的`onException()`。当请求失败时，该方法会告知你导致失败的`异常`(Exception)，如果解码器（Decoder）不能从数据中解析出任何有用的信息，Exception也可能传`null`。你可以使用`listener()`API传一个你的监听器（listener）到每一个请求中。
请确保`onException()`返回`false`，以免覆盖了Glide的默认错误处理行为（例如，它默认会通知`Target`这个error）。
这是一个快速调试的例子：

```java
// example usage: .listener(new LoggingListener<String, GlideDrawable>()) 
public class LoggingListener<T, R> implements RequestListener<T, R> {
    @Override public boolean onException(Exception e, Object model, Target target, boolean isFirstResource) {
        android.util.Log.d("GLIDE", String.format(Locale.ROOT,
                "onException(%s, %s, %s, %s)", e, model, target, isFirstResource), e);
        return false; 
    } 
    @Override public boolean onResourceReady(Object resource, Object model, Target target, boolean isFromMemoryCache, boolean isFirstResource) {
        android.util.Log.d("GLIDE", String.format(Locale.ROOT,
                "onResourceReady(%s, %s, %s, %s, %s)", resource, model, target, isFromMemoryCache, isFirstResource));
        return false; 
    } 
} 
```
**确保发版前移除相关代码**

### 更多日志
这个列表是给3.6.0版本用的，可能不完整。

```xml
cd .../android-sdk/platform-tools
adb shell setprop log.tag.AnimatedGifEncoder VERBOSE
adb shell setprop log.tag.AssetUriFetcher VERBOSE
adb shell setprop log.tag.BitmapEncoder VERBOSE
adb shell setprop log.tag.BufferedIs VERBOSE
adb shell setprop log.tag.ByteArrayPool VERBOSE
adb shell setprop log.tag.CacheLoader VERBOSE
adb shell setprop log.tag.ContentLengthStream VERBOSE
adb shell setprop log.tag.DecodeJob VERBOSE
adb shell setprop log.tag.DiskLruCacheWrapper VERBOSE
adb shell setprop log.tag.Downsampler VERBOSE
adb shell setprop log.tag.Engine VERBOSE
adb shell setprop log.tag.EngineRunnable VERBOSE
adb shell setprop log.tag.GenericRequest VERBOSE
adb shell setprop log.tag.GifDecoder VERBOSE
adb shell setprop log.tag.GifEncoder VERBOSE
adb shell setprop log.tag.GifHeaderParser VERBOSE
adb shell setprop log.tag.GifResourceDecoder VERBOSE
adb shell setprop log.tag.Glide VERBOSE
adb shell setprop log.tag.ImageHeaderParser VERBOSE
adb shell setprop log.tag.ImageVideoDecoder VERBOSE
adb shell setprop log.tag.IVML VERBOSE
adb shell setprop log.tag.LocalUriFetcher VERBOSE
adb shell setprop log.tag.LruBitmapPool VERBOSE
adb shell setprop log.tag.MediaStoreThumbFetcher VERBOSE
adb shell setprop log.tag.MemorySizeCalculator VERBOSE
adb shell setprop log.tag.PreFillRunner VERBOSE
adb shell setprop log.tag.ResourceLoader VERBOSE
adb shell setprop log.tag.RMRetriever VERBOSE
adb shell setprop log.tag.StreamEncoder VERBOSE
adb shell setprop log.tag.TransformationUtils VERBOSE
```

---------

## 使用Glide下载自定义大小的图片
开发者可以通过Glide的ModelLoader接口获得图片大小，并根据这个大小来加载一个合适尺寸的图片url。
使用合适尺寸的图片可以节约带宽，设备的存储空间，还可以提升app性能。
2014年的Google I/O app团队写了一篇关于如何使用ModelLoader接口调整加载的图片大小的文章。请在GitHub上查看 [I/O app的源码](https://github.com/google/iosched/blob/master/doc/IMAGES.md)。
为了实现自定义的ModelLoader来通过http/https下载图片，可以继承BaseGlideUrlLoader这个类

```java
public interface MyDataModel { 
    public String buildUrl(int width, int height);
}  
 
public class MyUrlLoader extends BaseGlideUrlLoader<MyDataModel> { 
    @Override 
    protected String getUrl(MyDataModel model, int width, int height) {
        // Construct the url for the correct size here. 
        return model.buildUrl(width, height);
    } 
} 
```

然后，你就可以使用自定义的ModelLoader来加载图片了，其他的事情会自动完成：

```java
Glide.with(yourFragment)
    .using(new MyUrlLoader())
    .load(yourModel)
    .into(yourView);
```

如果你想避免调用`.using(new  MyUrlLoader())`，你可以实现一个自定义的`ModelLoaderFactory`，并在`GlideModule`中注册它。

```java
public class MyGlideModule implements GlideModule { 
    ... 
    @Override 
    public void registerComponents(Context context, Glide glide) {
        glide.register(MyDataModel.class, InputStream.class, 
            new MyUrlLoader.Factory()); 
    } 
} 
```

注册ModelLoaderFactory之后，你就不用调用`.using()`了：

```java
Glide.with(yourFragment)
    .load(yourModel)
    .into(yourView);
```

其他的例子，关于如何使用自定义ModelLoader加载各种尺寸的图片，请查看[Flicker示例应用](https://github.com/bumptech/glide/blob/master/samples/flickr/src/main/java/com/bumptech/glide/samples/flickr/FlickrModelLoader.java)，和[Giphy示例应用](https://github.com/bumptech/glide/blob/master/samples/giphy/src/main/java/com/bumptech/glide/samples/giphy/GiphyModelLoader.java)。

---------

## 集成（Integration）库-Glide与其他库整合

### 介绍

####什么是集成库（Integration Library）
Glide包括一系列小巧可配置的集成库，这些库可以让Glide与外部的库相结合。现在集成库可以使Glide支持Volley和OkHttp的http/https请求。

我们坚信，您所选择的客户端媒体库既不应该决定你在你的应用程序中使用的网络库，也不需要你仅仅为了加载图像额外添加网络库。集成库和Glide的ModelLoader系统，允许开发人员对所有网络请求使用统一的网络库。

#### 为什么没有XXX库的实现
因为我们还没有为这个库写集成库。OkHttp和Volley是十分流行的库，许多开发者都使用，但是，我们不是排斥其他库，如果你写了其他库的`ModelLoader`并且打算开源，我们很乐意看到这样的提交请求（pull request）。

#### 我如何依赖一个集成库呢？
依赖任何的集成库需要两g个步骤。

1. 添加相应的继承库的Maven/Gradle/jar依赖，因为作为可选项的集成库并不包含在glide的jar依赖中。
2. 确保app包含了集成库的GlideModule，具体内容看[配置wiki](https://github.com/bumptech/glide/wiki/Configuration)部分。对于Glide集成库的具体说明见下面。

#### 我应当选择哪个版本？
集成库的版本跟随着Glide的release版本。但是多了一个不同的数字。确保选择与所依赖的Glide对应的集成库。在[release页](https://github.com/bumptech/glide/releases)查看.
网络库有自己的版本号。继承库会依赖的库就是你在Maven/Gradle中指定的库。你可以通过版本号来指定具体依赖的网络库。？？

### Volley
Volley是一个Http库，可以使Android上的网络请求更简单，更快速。

#### Gradle中使用Volley

```java
dependencies {
    compile 'com.github.bumptech.glide:volley-integration:1.3.1@aar'
    //compile 'com.mcxiaoke.volley:library:1.0.8'
}
```
集成库的`GlideModule`会自动合并到你app的manifest中。

#### Maven中是Volley

```xml
<dependency>
    <groupId>com.github.bumptech.glide</groupId>
    <artifactId>volley-integration</artifactId>
    <version>1.3.1</version>
    <type>aar</type>
</dependency>
<dependency>
    <groupId>com.mcxiaoke.volley</groupId>
    <artifactId>library</artifactId>
    <version>1.0.8</version>
    <type>aar</type>
</dependency>

```
请查看对应的manifest章节，了解如何添加对应的GlideModule。

#### 手动添加Volley
从[release页](https://github.com/bumptech/glide/releases)下载[glide-volley-integration-<version>.jar ](https://github.com/bumptech/glide/releases/download/v3.6.1/glide-volley-integration-1.3.1.jar)。并添加到你app的编译路径中(compile classpath)。

请查看对应的manifest章节，了解如何添加对应的GlideModule。

#### Volley的Manifest
如果你使用那些不支持manifest合并的编译系统（如Maven，Ant），你必须手动添加`GlideModule`的metadata标签到`AndroidManifest.xml`中。

```xml
<meta-data
    android:name="com.bumptech.glide.integration.volley.VolleyGlideModule"
    android:value="GlideModule" />
```

#### Volley的混淆设置
无论使用什么编译系统，不要混淆`VolleyGlideModule`类，它需要被反射来实例化。添加下面的代码到`proguard.cfg`文件（或者查看“通用部分”）。

```java
-keep class com.bumptech.glide.integration.volley.VolleyGlideModule
```

### OkHttp
OKHttp是一个高效且易于使用的Http客户端。

#### Gradle中使用OkHttp

```java
dependencies {
    compile 'com.github.bumptech.glide:okhttp-integration:1.3.1@aar'
    //compile 'com.squareup.okhttp:okhttp:2.2.0'
}
```
集成库的`GlideModule`会自动合并到你app的manifest中。

#### Maven中是OkHttp

```xml
<dependency>
    <groupId>com.github.bumptech.glide</groupId>
    <artifactId>okhttp-integration</artifactId>
    <version>1.3.1</version>
    <type>aar</type>
</dependency>
<!--
<dependency>
    <groupId>com.squareup.okhttp</groupId>
    <artifactId>okhttp</artifactId>
    <version>2.2.0</version>
    <type>jar</type>
</dependency>
-->

```
请查看对应的manifest章节，了解如何添加对应的GlideModule。

#### 手动添加OkHttp
从[release页](https://github.com/bumptech/glide/releases)下载[glide-okhttp-integration-<version>.jar ](https://github.com/bumptech/glide/releases/download/v3.6.1/glide-okhttp-integration-1.3.1.jar)。并添加到你app的编译路径中(compile classpath)。

请查看对应的manifest章节，了解如何添加对应的GlideModule。

#### OkHttp的Manifest
如果你使用那些不支持manifest合并的编译系统（如Maven，Ant），你必须手动添加`GlideModule`的metadata标签到`AndroidManifest.xml`中。

```xml
<meta-data
    android:name="com.bumptech.glide.integration.okhttp.OkHttpGlideModule"
    android:value="GlideModule" />
```

#### OkHttp的混淆设置
无论使用什么编译系统，不要混淆`OkHttpGlideModule`类，它需要被反射来实例化。添加下面的代码到`proguard.cfg`文件（或者查看“通用部分”）。

```java
-keep class com.bumptech.glide.integration.okhttp.OkHttpGlideModule
```

### 更多选项

#### 通用的混淆配置
你也可以使用下面的配置来避免混淆所有的`GlideModule`。

```java
-keep public class * implements com.bumptech.glide.module.GlideModule
```
这种方式有其他的好处，当修改集成库或者自定义集成库的行为时，不需要修改。当你添加或者移动其他module的时候，也不需要修改什么。

#### 覆盖默认的行为
如果默认配置无法满足你，所有的集成库还有一些额外的选项。比如添加重试行为，请查看集成库的`GlideModule`的源码（位于[/integration/<lib>/src/main/java/<package>](https://github.com/bumptech/glide/tree/3.0/integration)）了解默认的注册做了些什么。你可以通过在自定义的`GlideModule`中修改参数为`UrlLoader.Factory`类来改变默认行为。
当你要覆盖默认行为时，请确保自定义的`GlideModule`被注册，且默认的GlideModule被排除在外。排除GlideModule可能是从manifest中移除相应的的metadata，或者使用jar包依赖代替aar依赖。关于`GlideModule`的更多信息请查看[配置的wiki页](https://github.com/bumptech/glide/wiki/Configuration)\

---------

## 在后台线程中加载和缓存
为了使后台加载资源和与媒体交互更加容易，除了`Glide.with(fragment).load(url).into(view)`这个API外，Glide还提供了额外两个API。

* `downloadOnly(int, int)`
* `into(int, int)`

### downloadOnly方法
Glide的`downloadOnly(int, int)`方法允许你把图片bytes下载到磁盘缓存中，以便以后获取使用。你可以在UI线程异步地调用`downloadOnly()`，也可以在后台线程同步的使用。但是，注意他们的参数有些不同，异步API的参数是`Target`，同步api的参数是宽和高的整数值。	
为了在后台线程下载图片，你必须使用同步方法

```java
FutureTarget<File> future = Glide.with(applicationContext)
    .load(yourUrl)
    .downloadOnly(500, 500);
File cacheFile = future.get();
```
当future返回时，图片的bytes数据就在缓存中可用了。一种典型的情况是，你使用`downloadOnly()`API只是为了确保数据下载到了磁盘上。一般情况下，虽然你有权访问底层的缓存文件，但是你不需要直接和它交互。
而是，当你想要获取这个图片时，你只要像平常加载图片时那样调用就行，**只有一点不同**：

```java
Glide.with(yourFragment)
    .load(yourUrl)
    .diskCacheStrategy(DiskCacheStrategy.ALL)
    .into(yourView);
```
通过传入 `DiskCacheStrategy.ALL`或者`DiskCacheStrategy.SOURCE`，确保Glide可以使用你通过`downloadOnly()`下载的数据。

### into方法
如果你想在一个**后台线程**与一张已经解码的图片交互。你可以使用这个版本的`into()`方法来返回一个`FutureTarget`。例如，获取一张中心剪裁后的500*500像素的图片：

```java
Bitmap myBitmap = Glide.with(applicationContext)
    .load(yourUrl)
    .asBitmap()
    .centerCrop()
    .into(500, 500)
    .get()
```
虽然`into(int, int)`方法在后台线程中很有效，但是，你不能把它用在主线程中。即使这个同步方法在你的主线程中不会抛出异常，调用get()也会阻塞主线程。会使你的APP性能变差，反应迟钝。

---------

## Glide中的资源复用

### 为什么 - 资源复用的作用
Glide通过复用资源避免不必要的内存分配。Dalvik虚拟机（在Lollipop之前）有两种基本的垃圾回收方式，`GC_CONCURRENT` 和`GC_FOR_ALLOC`。每次`GC_CONCURRENT`会阻塞主线程5ms。由于每次操作的时间少于16ms（1帧的时间），`GC_CONCURRENT`并不会引起掉帧。相反的是`GC_FOR_ALLOC`，他会停止所有操作，阻塞主线程125+ms，事实上，GC_FOR_ALLOC总是会让你的app掉很多帧。尤其是在滑动时，导致明显的卡顿。
很不幸，即使只是分配适当的空间（比如16kb的buffer）Dalvik表现的也很糟糕。不断的小内存分配，或者一次大的内存分配（比如说bitmap），将会引起GC_FOR_ALLOC。因此，你分配内存的越多，你会遇到垃圾回收器阻塞应用的情况就越多，应用掉帧就越严重。
通过适度复用大块资源，Glide可以避免内存抖动，减少垃圾回收器阻塞app的次数。

### 怎么做 - Glide是如何复用资源
Glide的资源复用策略比较宽松。这意味着，当Glide认为该资源可以被安全的复用时，才有几率去复用它，并不需要开发者在每个request后面去手动释放资源。

#### 标志-哪些资源可复用
Glide有两个简单的标志来识别可复用的资源。

1. `Glide.clear()` 

	在`View`或者`Target`上调用`clear()`方法都表示，Glide要取消加载，可以安全地把Target占用的所有资源（Bitmap，bytes数组等）放入资源池中（pool）。用户可以在任何时候安全地手动调用`clear()`方法，但是典型情况下，我们不需要这样做，看第二条。

2. View或者Target的复用 

	当用户把图片加载到一个已经存在的View或者Target上时（注：确切的说是调用`into(xxx)`方法之后），Glide会先调用`clear()`清空该View/Target上的加载请求并复用已经显示过的资源。因此，如果你的ListView或者RecyclerView中的view使用了复用机制（注：如ViewHolder），那么Glide会自动为他们缓存资源和管理加载请求。
 
#### 引用计数
如果两个请求指向同一个资源，为了避免额外的工作，Glide会把单个资源分配给他们。这就导致一个问题，当Glide得知某个资源不被某个调用者使用，这并不表示它不会被其他调用者使用。为了避免回收调依旧被使用的资源，Glide使用引用计数来跟踪资源。
当把资源提供给View/Target时，该资源的引用计数加1，当清空View/Target时，引用计数减1。当引用计数为0时，Glide会回收资源，并把它的内容放回可用内存池中。

#### 放入缓存池
Glide的Resource API有一个`recycle()`方法，当Glide认为资源不再被引用时，会调用该方法，资源会放入缓存池中。

Glide提供的BitmapPool接口可以让Resource获取`Bitmap`和复用Bitmap对象。Glide的BitmapPool可以通过Glide单例获得：

```java
Glide.get(context).getBitmapPool();
```
ResourceDecoder可以返回Resource的任何实现。所有，开发者可以实现他们自己的Resource和ResourceDecoder来自定义地缓存一些特有类型的数据。
同样地，开发者如果想更多地控制Bitmap缓存，可以实现自己的BitmapPool，然后通过GlideModule配置到Glide中。

### 常见的错误
不幸的是，缓存池设计使我们很难判断开发者是否误用了资源或者bitmap。但是，在Glide中有两个主要的现象会暗示你某些地方可能出了问题。

#### 现象

1. `Cannot draw a recycled Bitmap`

	Glide有个固定大小的Bitmap池。当Bitmap不再被复用（注：不是使用，区分使用和复用），会从池中移走。Glide会调用[`recycle()`](http://developer.android.com/reference/android/graphics/Bitmap.html#recycle())（注：指的是Bitmap真正的recycle，不是Resource类的recycle）。你告诉Glide放心地回收某个Bitmap，但是，你的应用不小心还持有这个Bitmap的引用，应用程序可能会绘制这个Bitmap，导致崩溃。

2. View在多张图片之间闪烁，或者同样的图片出现在多个View中

	如果一张图片被放入BitmapPool中多次。或者虽然一张图片被放入了pool中，但是某个View依然持有这个图的引用，与此同时，另一张图片被解析成了Bitmap（注：此Bitmap正好用了刚才那张图片的控件来存放解析后的数据）。如果发生这种事情，Bitmap的内容就会被换成了新的图片内容。此时，View依然尝试着绘制Bitmap，导致原来的View中显示了一张新的图片！

#### 原因
这些问题主要有两个原因：

1. 尝试加载两个不同的资源到同一Target中

	在Glide中，没有安全的方法来加载多个资源到单一的Target中。开发者可以使用`thumbnail()`来加载一系列资源到到某个Target中，但是对于每一个加载的资源来说，只有在下一个`onResourceReady()`被调用前，它的引用才是安全的。
开发者如果想加载多个资源到同一个View中，可以使用两个独立的Target。为了确保加载过程不相互取消，开发者要么不使用ViewTarget的子类，要么在继承ViewTarget时，复写`setRequest()` 和 `getRequest()`，不要使用tag来存储Request。（注：需要一个demo)

	**译者注：对于同一个view，调用两次Glide.xxx.into(view)，第二次调用会先清空第一个加载的图片（出现空白），再去下载新的图片，如果想要在第二张图片下载下来之前依旧显示之前的，需要一些技巧**
2. 加载一个资源放入到Target，然后清空或者复用了Target，但是依然引用这这个资源。

	最简单的避免这个错误的方法是在`onLoadCleared()`方法中把所有对资源对象的引用置null。一般情况下，加载一个Bitmap，然后引用它的Target是安全的。不安全的是，你清空了这个Target，却依然引用着这个Bitmap。
	
---------
	
## 使用快照

### 关于快照
对于那些不想等待Glide下个正式版本而愿意尝鲜的用户，我们在[Sonatype](https://travis-ci.org/bumptech/glide)上部署了这个库的快照。
每一次我们push代码待GitHub的master分支，[travis-ci](https://oss.sonatype.org/content/repositories/snapshots/)会构建Glide.如果构建成功，会自动部署最新版本的库到Sonatype上。
和Glide主库一样，每个Intergration库也有自己的快照，如果你使用快照版本的Glide库，请使用快照版本的Intergration库，反之依然。

### 获取快照
Sonatype的快照库和其他的maven库一样，提供jar，maven，gradle等版本。

#### Jar
Jar包可以直接从Sonatype上下载，再次检查一下日期，确保使用最新版本

#### Gradle
在仓库列表中添加快照仓库

```xml
repositories {
  jcenter()
  maven {
    url 'http://oss.sonatype.org/content/repositories/snapshots'
  }
}
```
然后修改依赖为快照版本

```xml
dependencies {
  compile "com.github.bumptech.glide:glide:3.6.0-SNAPSHOT"
  compile "com.github.bumptech.glide:okhttp-integration:1.3.0-SNAPSHOT"
}
```

#### Maven
这种方式没有测试，直接从[StackOverflow](http://stackoverflow.com/questions/7715321/how-to-download-snapshot-version-from-maven-snapshot-repository)拷过来的。欢迎改进下面的内容。
添加下面的代码到`~/.m2/settings.xml`中：

```xml
<profiles>
  <profile>
     <id>allow-snapshots</id>
     <activation><activeByDefault>true</activeByDefault></activation>
     <repositories>
       <repository>
         <id>snapshots-repo</id>
         <url>https://oss.sonatype.org/content/repositories/snapshots</url>
         <releases><enabled>false</enabled></releases>
         <snapshots><enabled>true</enabled></snapshots>
       </repository>
     </repositories>
   </profile>
</profiles>
```
修改依赖为快照版本

```xml
<dependency>
  <groupId>com.github.bumptech.glide</groupId>
  <artifactId>glide</artifactId>
  <version>3.6.0-SNAPSHOT</version>
</dependency>
<dependency>
  <groupId>com.github.bumptech.glide</groupId>
  <artifactId>okhttp-integration</artifactId>
  <version>1.3.0-SNAPSHOT</version>
</dependency>
```

---------

## 图形变换

### 默认的变换
Glide包含两种默认的图像变换方式，fitCenter和centerCrop。如果开发者需要其他类型的变换，可以考虑使用这个独立的[变换库](https://github.com/wasabeef/glide-transformations)。

#### Fit center
FitCenter会按原始比例缩小图像，使图像可以在放在给定的区域内。FitCenter会尽可能少地缩小图片，使宽或者高的一边等于给定的值。另外一边会等于或者小于给定值。
FitCenter和Android中的ScaleType.FIT_CENTER效果相同。
#### CenterCrop
CenterCrop会按原始比例缩小图像，使宽或者高的一边等于给定的值，另外一边会等于或者大于给定值。CenterCrop会裁剪掉多余部分。
CenterCrop和Android中的ScaleType.CENTER_CROP效果相同。

### 使用
fit center效果使用`.fitCenter()`：

```java
Glide.with(yourFragment)
    .load(yourUrl)
    .fitCenter()
    .into(yourView);
```
center crop效果使用`.centerCrop()`：

```java
Glide.with(yourFragment)
    .load(yourUrl)
    . centerCrop()
    .into(yourView);
```
如果你只加载Bitmap或者Gif，也可以使用这个变换：

```java
// For Bitmaps:
Glide.with(yourFragment)
    .load(yourUrl)
    .asBitmap()
    .centerCrop()
    .into(yourView);

// For gifs:
Glide.with(yourFragment)
    .load(yourUrl)
    .asGif()
    .fitCenter()
    .into(yourView);
```
当在类型间转码时，也可以使用这些变换。例如，获取一个变形后的jpeg图片的bytes数据：

```java
Glide.with(yourFragment)
    .load(yourUrl)
    .asBitmap()
    .toBytes()
    .centerCrop()
    .into(new SimpleTarget<byte[]>(...) { ... });

```
自定义变换
除了两个内置的变换，你还可以自定义变换。
最简单的方式是继承BitmapTransformation。

```java
private static class MyTransformation extends BitmapTransformation { 
 
    public MyTransformation(Context context) {
       super(context);
    } 
 
    @Override 
    protected Bitmap transform(BitmapPool pool, Bitmap toTransform, 
            int outWidth, int outHeight) {
       Bitmap myTransformedBitmap = ... // apply some transformation here. 
       return myTransformedBitmap; 
    } 
 
    @Override 
    public String getId() {
        // Return some id that uniquely identifies your transformation. 
        return "com.example.myapp.MyTransformation"; 
    } 
} 
```

之后一就可以用同样的方式使用它。使用`.transform(...)`代替`.fitCenter()`/`.centerCrop()`。

```java
// For the default drawable type:
Glide.with(yourFragment)
    .load(yourUrl)
    .transform(new MyTransformation(context))
    .into(yourView);

// For Bitmaps:
Glide.with(yourFragment)
    .load(yourUrl)
    .asBitmap()
    .transform(new MyTransformation(context))
    .into(yourView);

// For Gifs:
Glide.with(yourFragment)
    .load(yourUrl)
    .asGif()
    .transform(new MyTransformation(context))
    .into(yourView);
```

#### 调整大小
你可能注意到上面的例子没有传入具体的尺寸，那么，Transformation中的尺寸是如何获得的呢？
Transformation中的尺寸就是View或者Target的大小。Glide会根据布局文件中的weight，match_parent或者具体值算出View的大小。，当你拥有View/Target的具体大小，又拥有原始图片的大小时，就可以通过图像变换生成一个正确大小的图片。
如果你想指定View/Target的自定义大小，可以使用`.override(int, int)`方法，如果你想加载一个图片有其他用途，而不是显示在View中，请查看『自定义Target』章节。

#### Bitmap 复用
为了减少垃圾回收，你可以说使用`BitmapPool`接口来释放不想要的Bitmap或者复用存在的Bitmap。一个在Transformation中复用Bitmap典型的例子：从pool中获取Bitmap，使用该Bitmap创建一个`Canvas`，然后使用Matrix/Paint/Shader来变换图像并绘制到Canvas上。为了正确有效地在Transformation中复用Bitmap中，遵守下面的规则：

1. 在`transform()`不要回收资源或者把Bitmap放入Bitmap池中，这些步骤会自动完成。
2. 如果你从Bitmap池中获取了多个Bitmap，或者没有使用从pool中获取到的Bitmap，确保把多余的Bitmap放回pool中。
3. 如果你的Trasformation并没有替换调原始图片（比如，图片已经满足你要求的大小，直接返回了），请在`transform()`方法中返回原始的资源或者Bitmap。

一个典型的用法如下：

```java
protected Bitmap transform(BitmapPool bitmapPool, Bitmap original, int width, int height) {
    Bitmap result = bitmapPool.get(width, height, Bitmap.Config.ARGB_8888);
    // If no matching Bitmap is in the pool, get will return null, so we should allocate. 
    if (result == null) {
        // Use ARGB_8888 since we're going to add alpha to the image. 
        result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    } 
    // Create a Canvas backed by the result Bitmap. 
    Canvas canvas = new Canvas(result);
    Paint paint = new Paint();
    paint.setAlpha(128);
    // Draw the original Bitmap onto the result Bitmap with a transformation. 
    canvas.drawBitmap(original, 0, 0, paint);
    // Since we've replaced our original Bitmap, we return our new Bitmap here. Glide will 
    // will take care of returning our original Bitmap to the BitmapPool for us.  
    return result;
} 
```
