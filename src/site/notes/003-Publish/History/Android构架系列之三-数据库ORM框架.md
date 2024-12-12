---
{"dg-publish":true,"dg-path":"History/Android构架系列之三-数据库ORM框架.md","permalink":"/History/Android构架系列之三-数据库ORM框架/","title":"Android构架系列之三--数据库ORM框架","tags":["技术","Android","主框架"],"created":"2024-12-11T18:09:35.736+08:00","updated":"2024-12-11T18:09:35.737+08:00"}
---



## [数据库方案](https://www.zhihu.com/question/27977160)

* 原生sqlite
* orm框架
* ContentProvider+LoaderManager
* 结合使用

最终选择了DBFlow框架，基于以下几点优势：

* 速度快，没有反射，使用编译时生成相关类。
* **支持Content Provider**，与schematic库相集成。
* 流式的sql API，接口友好。
* Google在Demo中使用，算是官方推荐的库。

下面介绍一下ContentProvider和DBFlow框架

## ContentProvider

### 何时使用
Google建议：http://developer.android.com/intl/zh-cn/guide/topics/providers/content-provider-creating.html


### 使用ContentProvider
* 一个app多个ContentProvider，一个ContentProvider，对应一个数据库，一个数据库中含有n个table（表）
* 快速生成ContentProvider的工具-[schematic](https://github.com/SimonVT/schematic)，[demo](https://github.com/SimonVT/schematic/blob/master/schematic-samples%2Fsrc%2Fmain%2Fjava%2Fnet%2Fsimonvt%2Fschematic%2Fsample%2Fdatabase%2FNotesProvider.java)
* 事务操作：创建一个 ContentProviderOperation 数组，然后将相应的操作添加到数组中，最后通过 ContentResolver.applyBatch 实现批量操作。

### 注意点
* 进程安全，但不是线程安全，query 等方法可以同时被多个线程调用，所以这些方法必须线程安全。也就是不能使用全局变量、静态变量。（如果非要使用，必须加锁）。但是，数据库dbhelp自身会对操作排队，所以不用担心数据库的线程安全。
* 避免在 onCreate() 中执行长时间操作，只构建DbHelper。。不getdb，这很快。
* 外部应用访问数据有多重方式，通过路径开发部分内容、临时权限，intent访问等


## DBFlow功能介绍
这个库的基本使用，先学习[官方文档](https://github.com/Raizlabs/DBFlow)，[中文](https://www.gitbook.com/book/3yumenokanata/dbflow-tutorials/details)，不再赘述。
### 基本使用
步骤：

1. Application中初始化
2. 建立DataBase。一般情况下，一个App只有一个DB，然后有多张表。
3. 根据业务需求建立Table,必须实现Model接口（更多是继承BaseModel类）

实际开发中，可能还需要建立表之间的关联（Relation），在建立Table时，通过`foreignKey`来实现（1对1，1对多，多对多-3.0以上）

其他几个概念:

* 类型转换器：`@Column`只支持基本数据类型/相关封装类/Date，使用自定义的转换器，可以支持自己实现的类。
* _Table类：DBFlow会自动生成的类，里面的字段是查询条件常用的
* _Adapter类:

### 流式API
使用DBFlow掌握了[流式的API](https://github.com/Raizlabs/DBFlow/blob/master/usage/SQLQuery.md)基本掌握了它的使用：
下面两种使用方式功能相同：

* `SQLite.select().from(someTable).where(conditions).query();`
* `new Select().from(someTable).where(conditions).query();`

	其中
	* conditions为查询条件，一般为`someTable_Table.name.eq("muz")`，从`_Table`类中找。
	* 同样适用于insert update delete。注意：**不用忘了最后都要调用**`query()`。

### 面向对象API
另外一种使用DBFlow的方式，没有上面那种方式灵活，但是更加友好。
对于我们使用`@Table`注解的类，可以直接新建一个对象，如下：

```java
SomeTable oneRecord = new SomeTable();
oneRecord.id=1
oneRecord.load();
```
注意：局限性只能查找`@PrimaryKey `属性
同样的，有update,insert等方法。

### 一些使用要点

* select中`query()`与`queryList()`：前者返回Cursor，后者直接返回对象List，**更安全**,不用close Cursor。
* 同步与异步：**默认是同步查询**，需要异步加上`async()`和回调`query(listener)`
* 如果需要多个查询条件，两种方式:
	1. 在where后添加and(xxx)或者or(xxx)--**更常用**
	2. 使用组合查询条件ConditionGroup

	```java
SQLite.select()
  .from(MyTable.class)
  .where(MyTable_Table.someColumn.is("SomeValue"))
  .and(MyTable_Table.anotherColumn.is("ThisValue"));
// 或者
SQLite.select()
   .from(MyTable.class)
   .where(ConditionGroup.clause()
   .and(MyTable_Table.someColumn.is("SomeValue")
   .or(MyTable_Table.anotherColumn.is("ThisValue"));
```

### 其他的高级功能（有用的）
#### 数据迁移 - 数据库升级
DBFlow支持使用`Migration `支持升级数据库`@Database`版本.

* 继承`AlterTableMigration<SomeTable>`，修改SomeTable表操作

	```java
@Migration(version = 2, database = AppDatabase.class)
public class Migration1 extends AlterTableMigration<TestModel> {
  @Override
  public void onPreMigrate() {
    // 一些数据库操作，如修改表，需要测试！
    addColumn(Long.class, "timestamp");
  }
}

* 继承`UpdateTableMigration<SomeTable>`，修改SomeTable表中内容操作

	```java
@Migration(version = 2, database = AppDatabase.class)
public class Migration1 extends UpdateTableMigration<TestModel> {

    @Override
    public void onPreMigrate() {
      // UPDATE TestModel SET deviceType = "phablet" WHERE screenSize > 5.7 AND screenSize < 7;
      set(TestModel_Table.deviceType.is("phablet"))
        .where(TestModel_Table.screenSize.greaterThan(5.7), TestModel_Table.screenSize.lessThan(7));
    }
    ```

扩展功能：
注意：如果**想要在第一次创建时对数据库进行初始化操作**，只需要使用**`version = 0`**

#### 支持ContentProvider
支持ContentProvider只有很少的修改。

1. 将有@Database的类添加@ContentProvider
	
	```
	@ContentProvider(authority = TestDatabase.AUTHORITY,
        database = TestDatabase.class,
        baseContentUri = TestDatabase.BASE_CONTENT_URI)
@Database(name = TestDatabase.NAME, version = TestDatabase.VERSION)
public class TestDatabase {

    public static final String NAME = "TestDatabase";

    public static final int VERSION = "1";

    public static final String AUTHORITY = "com.raizlabs.android.dbflow.test.provider";

    public static final String BASE_CONTENT_URI = "content://";
}
	```
2. 在Manifest添加provider，注意名称为`类名$Provider`。

	```xml
	<provider
    android:authorities="com.raizlabs.android.dbflow.test.provider"
    android:exported="true|false"
    android:name=".provider.TestContentProvider$Provider"/>
	```
3. 至少添加一个 `@TableEndpoint`:在`@Table`类中指定content provider的类名，代码如下，注意`@ContentUri `

	```java
	@TableEndpoint(name = ContentProviderModel.NAME, contentProviderName = "ContentDatabase")
@Table(database = ContentDatabase.class, tableName = ContentProviderModel.NAME)
public class ContentProviderModel extends BaseProviderModel<ContentProviderModel> {

    public static final String NAME = "ContentProviderModel";

    @ContentUri(path = NAME, type = ContentUri.ContentType.VND_MULTIPLE + NAME)
    public static final Uri CONTENT_URI = ContentUtils.buildUri(ContentDatabase.AUTHORITY);

    @Column
    @PrimaryKey(autoincrement = true)
    long id;

    @Column
    String notes;

    @Column
    String title;

    @Override
    public Uri getDeleteUri() {
        return TestContentProvider.ContentProviderModel.CONTENT_URI;
    }

    @Override
    public Uri getInsertUri() {
        return TestContentProvider.ContentProviderModel.CONTENT_URI;
    }

    @Override
    public Uri getUpdateUri() {
        return TestContentProvider.ContentProviderModel.CONTENT_URI;
    }

    @Override
    public Uri getQueryUri() {
        return TestContentProvider.ContentProviderModel.CONTENT_URI;
    }
}
	```
4. (非必须，会使在Model上的操作使用ContentProvider）修改Table继承的类为下面的一种，如上的代码。**强烈推荐第二个`BaseSyncableProviderModel `****????疑问ContentProvider不写数据库吗**
	* BaseProviderModel：Model上的所有操作都是通过ContentProvider
	* BaseSyncableProviderModel：与上面相同，**除此之外，还会同步到本地的App数据库，即本地数据库会有相同数据**
5. 使用，有两种方式
	* 通用方式如下：
	
		```java
	ContentProviderModel contentProviderModel = ...; // some instance
int count = ContentUtils.update(getContentResolver(), ContentProviderModel.CONTENT_URI, contentProviderModel);
Uri uri = ContentUtils.insert(getContentResolver(), ContentProviderModel.CONTENT_URI, contentProviderModel);
int count = ContentUtils.delete(getContentResolver(), someContentUri, contentProviderModel);
		```
	* 如果使用了上面步骤4，还可以像下面这样:
	
		```java
		MyModel model = new MyModel();
model.id = 5;
model.load(); // queries the content provider
model.someProp = "Hello"
model.update(false); // runs an update on the CP
model.insert(false); // inserts the data into the CP
		```


#### DBFlow的特殊List
DBFlow可以直接从数据库获取的数据用以下两种数据结构返回。具体使用查看[文档中代码](https://github.com/Raizlabs/DBFlow/blob/master/usage/TableList.md)

* FlowCursorList：用在大量数据查询与Android中的`BaseAdapter`绑定，不需要一次性加载数据到List中，而是，只加载使用到的（用户看到的）数据。**十分有用！**
* FlowQueryList：用在List中数据与数据库中的数据同步修改

#### 事务
[参考官方文档](https://github.com/Raizlabs/DBFlow/blob/master/usage/Transactions.md)

#### 其他功能
* ModelCache缓存：select操作可以先从内存缓存中读取，需要手动打开
	注意：
	* FlowCursorList与FlowQueryList使用独立的ModelCache缓存，并不与@Tabel/Model类的缓存通用
* 支持索引：设置某些列使用索引，加快select速度
* 触发器：在数据库的某些操作之前或之后自动触发执行的一些动作

### 其他数据库操作常识
* 不要在循环中操作数据库，而是使用事务
