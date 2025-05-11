# MassLib4Search
用于快速构建可搜索质谱数据集的Python库，并提供常见的搜索任务与工具函数.

## 库结构
- `masslib4search.snaps.MassSearchTools`：定义了常见的搜索任务与工具函数。

## 工具函数组
工具函数是MassLib4Search库对外暴露的最低级别的API，用户可以通过组合这些函数来构建自己的搜索业务逻辑。
同时，相同功能但存在细微差别的函数群可以组成工具函数组。MassLib4Search库把它们聚合在一起，并抽象为`ToolBox`类，方便用户调用。
本质上来说，`ToolBox`类是一个具有`run`方法和`run_by_queue`的`BaseModel`对象的子类，分别用于处理**单组数据**和**队列数据**。
- `run`方法：处理数据的基本方法，输入为单组数据，输出为处理后的单组数据。
- `run_by_queue`方法：在实际工作中，数据可能会分组，每一组数据之间是互相隔离的，这种情况对于需要query和ref进行配对的搜索任务来说是非常常见的。该方法主要针对这种情况设计。输入为队列数据，输出为处理后的队列数据。在这个过程中，分组结构保持不变。

以下是可用的工具函数组：

- [`Binning`](#binning-anchor)：质谱数据分箱处理工具，可将离散质谱峰（m/z）聚合为规整特征向量。

### <a id="binning-anchor"></a> Binning
质谱数据分箱处理工具，可将离散质谱峰（m/z）聚合为规整特征向量。

#### 核心功能
- 支持CPU和CUDA两种实现
- 提供三种峰值聚合方式：求和(sum)、最大值(max)、平均值(avg)
- 支持批量并行处理，优化内存使用
- 处理嵌套数据结构，保持原始层级关系

#### 配置参数
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| binning_window | Tuple[float, float, float] | (50.0, 1000.0, 1.0) | 分箱范围及精度(最小值, 最大值, 步长) |
| pool_method | Literal['sum','max','avg'] | "sum" | 峰值聚合方式 |
| batch_size | int | 128 | 并行计算批次大小 |
| num_workers | int | 4 | 数据预处理并行度 |
| work_device | Union[str, torch.device, Literal['auto']] | 'auto' | 计算设备(自动推断策略) |
| output_device | Union[str, torch.device, Literal['auto']] | 'auto' | 结果存储设备(默认与计算设备一致) |

#### 数据输入
`Binning`工具支持多种输入模式：

1. **Spec模式** (推荐):
   - 输入格式: `List[torch.Tensor]`，每个张量形状为`(n_peaks, 2)`
   - 数据组织: 每行包含`[m/z值, 强度值]`对
   - 示例:
     ```python
     [
         torch.tensor([[100.0, 0.5], [200.0, 0.8]]),  # 谱图1
         torch.tensor([[150.0, 0.3], [300.0, 0.6]])   # 谱图2
     ]
     ```

2. **MZ模式**:
   - 需要两个独立列表:
     - `m/z列表`: `List[torch.Tensor]`，形状为`(n_peaks,)`
     - `强度列表`: `List[torch.Tensor]`，形状为`(n_peaks,)`
   - 示例:
     ```python
     mzs = [
         torch.tensor([100.0, 200.0]),  # 谱图1的m/z
         torch.tensor([150.0, 300.0])   # 谱图2的m/z
     ]
     intensities = [
         torch.tensor([0.5, 0.8]),      # 谱图1的强度
         torch.tensor([0.3, 0.6])       # 谱图2的强度
     ]
     ```

3. **嵌套结构输入**:
对于多组数据(如多个样本批次)，支持嵌套列表结构 （仅限`run_by_queue`方法使用）：
```python
[
    [tensor1, tensor2, ...],  # 样本组1
    [tensor3, tensor4, ...],  # 样本组2
    ...
]
```

#### 数据输出
`Binning`工具的输出格式根据调用方法不同而有所区别：

1. **run()方法输出** (单层结构处理):
   - 输出类型: `torch.Tensor`
   - 形状: `(num_spectra, num_bins)`
     - `num_spectra`: 输入光谱数量
     - `num_bins`: 根据`binning_window`参数计算的分箱数量
   - 数据类型: `torch.float32`
   - 设备位置: 默认与计算设备相同，可通过`output_device`参数指定
   - 示例:
     ```python
     # 输入2个光谱，分箱范围50-1000 m/z，步长1 → 950个分箱
     output.shape  # torch.Size([2, 950])
     ```

2. **run_by_queue()方法输出** (嵌套结构处理):
   - 输出类型: `List[torch.Tensor]`
   - 结构特点: 保持与输入相同的嵌套层级
   - 每个元素的形状: `(num_spectra_in_group, num_bins)`
   - 示例:
     ```python
     # 输入结构: [ [光谱1,光谱2], [光谱3] ]
     output = [
         torch.Size([2, 950]),  # 第一组结果
         torch.Size([1, 950])   # 第二组结果
     ]
     ```

**输出内容说明**:
- 每个分箱的值取决于`pool_method`参数:
  - `sum`: 该m/z范围内所有峰强度的总和
  - `max`: 该m/z范围内最大峰强度
  - `avg`: 该m/z范围内峰强度的平均值
- 空分箱(无对应m/z值)将填充为0
- 输出张量按m/z升序排列(从`binning_window[0]`开始)

**设备控制**:
- 默认情况下，输出张量位于计算设备(`work_device`)上
- 可通过设置`output_device`参数将结果转移到指定设备
- 支持设备间自动转换(如从CUDA到CPU)

**性能提示**:
- 对于CUDA设备，建议将`output_device`设为"cpu"以减少GPU内存占用
- 嵌套结构的输出会保持原始分组，适合后续分组处理


#### 使用示例
```python
from masslib4search.snaps.MassSearchTools.utils.toolbox import Binning

# 初始化工具
binning_tool = Binning(
    binning_window=(50.0, 1000.0, 1.0),
    pool_method="sum",
    batch_size=128,
    num_workers=4,
    work_device="auto"
)

# 处理单层结构数据
result = binning_tool.run(spec_data)

# 处理嵌套结构数据
nested_result = binning_tool.run_by_queue(nested_spec_data)
```
