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

- [`SpectrumBinning`](#binning-anchor)：质谱数据分箱处理工具，可将离散质谱峰（m/z）聚合为规整特征向量。
- [`EmbeddingSimilarity`](#embedding-similarity-anchor)：嵌入向量相似度计算工具，输出二维相似度矩阵。
- [`SpectrumSimilarity`](#spectrum-similarity-anchor)：谱图相似度计算工具，输出二维相似度矩阵。
- [`PeakMZSearch`](#peak-mz-search-anchor)：基于m/z值的质谱峰搜索工具箱，封装m/z峰搜索流程，提供配置化的计算参数管理。
- [`PeakPatternSearch`](#peak-pattern-search-anchor)：基于质谱图结构的模式匹配搜索工具箱，封装质谱图结构的模式匹配搜索流程，提供配置化的计算参数管理。
- [`EmbeddingSimilaritySearch`](#embedding-similarity-search-anchor)：嵌入向量相似度搜索工具，输出最相关的参考向量的索引和相似度分数。

对于这些工具函数组的命名上也有一些约定：
- Embedding~：表示这个工具函数组的输入为规整的向量格式，最小处理单元为embedding-chunk。
- Spectrum~：表示这个工具函数组的输入为质谱图张量格式，这种输入往往是nested结构，即每个谱图的长度不一致，最小处理单元为单个谱图。
- ~Similarity：相似度计算工具组，用于计算全量相似度矩阵，这种工具会对输入的Query和Ref进行pairwise计算，并计算出二维相似度矩阵。
- ~Search：搜索工具组，用于搜索质谱数据，提供配置化的计算参数管理。这种工具的一般输出为`Tuple[Ref-Index,Score]`，如果搜索结果是基于离散逻辑的（如PeakPatternSearch），则没有`Score`信息的输出。

### <a id="binning-anchor"></a> SpectrumBinning
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
`SpectrumBinning`工具支持多种输入模式：

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
`SpectrumBinning`工具的输出格式根据调用方法不同而有所区别：

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
from masslib4search.snaps.MassSearchTools.utils.toolbox import SpectrumBinning

# 初始化工具
binning_tool = SpectrumBinning(
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

### <a id="embedding-similarity-anchor"></a> EmbeddingSimilarity
嵌入向量相似度计算工具盒，输入向量并指定相似度，计算Query与Ref之间的相似度矩阵。

#### 核心功能
- 封装嵌入向量相似度计算流程
- 提供配置化的计算参数管理
- 支持设备自动分配策略
- 批处理并行计算
- 嵌套数据结构处理

#### 配置参数
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| sim_operator | Type[EmbbedingSimilarityOperator] | CosineOperator | 用于计算嵌入向量之间相似度的算子 |
| chunk_size | int | 5120 | 并行计算时每次处理的向量块大小 |
| batch_size | int | 128 | 并行计算批次大小 |
| num_workers | int | 4 | 数据预处理并行度 |
| work_device | Union[str, torch.device, Literal['auto']] | 'auto' | 计算设备（自动推断策略：优先使用输入数据所在设备） |
| output_device | Union[str, torch.device, Literal['auto']] | 'auto' | 结果存储设备（默认与计算设备一致） |
| operator_kwargs | Optional[dict] | None | 相似度计算算子的额外参数 |

**相似度算子**

在masslib4search/snaps/MassSearchTools/utils/similarity/operators.py中，我们预定义了以下算子：
  - CosineOperator (默认使用)：余弦相似度
  - DotOperator：点积
  - JaccardOperator：杰卡德相似度
  - TanimotoOperator：谷本系数
  - PearsonOperator：皮尔逊相关系数

所有计算向量相似度的算子均是`EmbbedingSimilarityOperator`的子类，用户可以自定义算子实现自定义相似度计算。
`EmbbedingSimilarityOperator`类的详细接口见masslib4search/snaps/MassSearchTools/utils/similarity/operators/ABC_operator.py


#### 数据输入
`EmbeddingSimilarity`工具支持两种输入模式：

1. **单层结构输入**:
   - 输入格式: `query: torch.Tensor`, `ref: torch.Tensor`
   - 形状: 
     - `query`: (n_q, dim)
     - `ref`: (n_r, dim)
   - 数据类型: `torch.float32`
   - 示例:
     ```python
     query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
     ref = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
     ```

2. **嵌套结构输入**:
   - 输入格式: `query_queue: List[torch.Tensor]`, `ref_queue: List[torch.Tensor]`
   - 形状: 
     - 每个`query_queue`元素: (n_q, dim)
     - 每个`ref_queue`元素: (n_r, dim)
   - 数据类型: `torch.float32`
   - 适用于处理多个样本或实验的批量相似度计算操作
   - 示例:
     ```python
     query_queue = [
         torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 查询组1
         torch.tensor([[5.0, 6.0], [7.0, 8.0]])   # 查询组2
     ]
     ref_queue = [
         torch.tensor([[9.0, 10.0], [11.0, 12.0]]),  # 参考组1
         torch.tensor([[13.0, 14.0], [15.0, 16.0]])   # 参考组2
     ]
     ```

#### 数据输出
`EmbeddingSimilarity`工具的输出格式根据调用方法不同而有所区别：

1. **run()方法输出** (单层结构处理):
   - 输出类型: `torch.Tensor`
   - 形状: `(n_q, n_r)`
   - 数据类型: `torch.float32`
   - 设备位置: 默认与计算设备相同，可通过`output_device`参数指定
   - 示例:
     ```python
     # 输入2个查询向量和2个参考向量 → 相似度矩阵 (2x2)
     output.shape  # torch.Size([2, 2])
     ```

2. **run_by_queue()方法输出** (嵌套结构处理):
   - 输出类型: `List[torch.Tensor]`
   - 结构特点: 保持与输入相同的嵌套层级
   - 每个元素的形状: `(n_q, n_r)`
   - 示例:
     ```python
     # 输入结构: [ [查询组1,查询组2], [查询组3] ]
     output = [
         torch.Size([2, 2]),  # 第一组结果
         torch.Size([1, 2])   # 第二组结果
     ]
     ```

#### 使用示例
```python
from masslib4search.snaps.MassSearchTools.utils.toolbox import EmbeddingSimilarity

# 初始化工具
embedding_similarity_tool = EmbeddingSimilarity(
    sim_operator=CosineOperator,
    chunk_size=5120,
    batch_size=128,
    num_workers=4,
    work_device='auto',
    operator_kwargs=None
)

# 处理单层结构数据
result = embedding_similarity_tool.run(query, ref)

# 处理嵌套结构数据
nested_result = embedding_similarity_tool.run_by_queue(query_queue, ref_queue)
```

### <a id="spectrum-similarity-anchor"></a> SpectrumSimilarity
谱图相似度计算工具盒，输入质谱谱图数据并指定相似度算子，计算查询谱图与参考谱图之间的相似度矩阵。

#### 核心功能
- 封装谱图相似度计算全流程
- 提供可配置化的计算参数管理
- 支持设备自动分配策略（CPU/CUDA）
- 基于Dask的批处理并行计算
- 嵌套数据结构层级保持

#### 配置参数
| 参数 | 类型 | 默认值 | 约束 | 描述 |
|------|------|--------|------|------|
| sim_operator | Type[SpectramSimilarityOperator] | MSEntropyOperator | - | 谱图相似度计算核心算子 |
| num_cuda_workers | int | 4 | ≥0 | CUDA并行度（每个Worker含3个CUDA流） |
| num_dask_workers | int | 4 | ≥0 | Dask并行线程数 |
| work_device | Union[str, torch.device, 'auto'] | 'auto' | - | 自动设备选择策略（优先输入设备） |
| output_device | Union[str, torch.device, 'auto'] | 'auto' | - | 结果存储设备选择策略 |
| operator_kwargs | Optional[dict] | None | - | 算子的额外参数配置 |
| dask_mode | Literal["threads", "processes", "single-threaded"] | "threads" | - | Dask任务调度模式 |

**预定义算子**
在masslib4search/snaps/MassSearchTools/utils/similarity/operators中预定义：
  - MSEntropyOperator (默认使用)：质谱熵相似度算法

所有谱图相似度算子均继承自`SpectramSimilarityOperator`，支持通过继承实现自定义算法，详细接口参见masslib4search/snaps/MassSearchTools/utils/similarity/operators/ABC_operator.py

#### 数据输入
`SpectrumSimilarity`工具支持两种谱图数据结构输入模式：

1. **单层结构输入**:
   - 输入格式: `query: List[torch.Tensor]`, `ref: List[torch.Tensor]`
   - 形状: 
     - `query`: List[(n_peaks, 2)]
     - `ref`: List[(n_peaks, 2)]
   - 数据类型: `torch.float32`
   - 示例:
     ```python
      query = [
          torch.tensor([[100.0, 0.5], [200.0, 0.8]]),  # 查询谱图1
          torch.tensor([[150.0, 0.3], [300.0, 0.6]])   # 查询谱图2
      ]
      ref = [
          torch.tensor([[100.0, 0.5], [200.0, 0.8]]),  # 参考谱图1
          torch.tensor([[150.0, 0.3], [300.0, 0.6]])   # 参考谱图2
      ]
     ```

2. **嵌套结构输入**:
   - 输入格式: `query_queue: List[List[torch.Tensor]]`, `ref_queue: List[List[torch.Tensor]]`
   - 形状: 
     - 每个`query_queue`元素: List[(n_peaks, 2)]
     - 每个`ref_queue`元素: List[(n_peaks, 2)]
   - 数据类型: `torch.float32`
   - 适用于处理多个样本或实验的批量相似度计算操作
   - 示例:
     ```python
      query_queue = [
          [torch.tensor([[100.0, 0.5], [200.0, 0.8]]), torch.tensor([[150.0, 0.3], [300.0, 0.6]])],  # 查询组1
          [torch.tensor([[250.0, 0.4], [400.0, 0.7]])]  # 查询组2
      ]
      ref_queue = [
          [torch.tensor([[100.0, 0.5], [200.0, 0.8]]), torch.tensor([[150.0, 0.3], [300.0, 0.6]])],  # 参考组1
          [torch.tensor([[250.0, 0.4], [400.0, 0.7]])]  # 参考组2
      ]
     ```

#### 数据输出
`SpectrumSimilarity`工具的输出格式根据调用方法不同而有所区别：

1. **run()方法输出** (单层结构处理):
   - 输出类型: `torch.Tensor`
   - 形状: `(n_q, n_r)`
   - 数据类型: `torch.float32`
   - 设备位置: 默认与计算设备相同，可通过`output_device`参数指定
   - 示例:
     ```python
      # 输入2个查询谱图和2个参考谱图 → 相似度矩阵 (2x2)
      output = embedding_similarity_tool.run(query, ref)
      print(output)
      # 输出: 
      # tensor([[0.9876, 0.1234],
      #         [0.5678, 0.8765]])
     ```
     
2. **run_by_queue()方法输出** (嵌套结构处理):
   - 输出类型: `List[torch.Tensor]`
   - 结构特点: 保持与输入相同的嵌套层级
   - 每个元素的形状: `(n_q, n_r)`
   - 示例:
     ```python
      # 输入结构: [ [查询组1,查询组2], [查询组3] ]
      output = embedding_similarity_tool.run_by_queue(query_queue, ref_queue)
      print(output)
      # 输出:  [
      #     tensor([[0.9876, 0.1234],
      #             [0.5678, 0.8765]]),
      #     tensor([[0.2345, 0.6789]])
      # ]
     ```

#### 使用示例
```python
from masslib4search.snaps.MassSearchTools.utils.toolbox import SpectrumSimilarity
from masslib4search.snaps.MassSearchTools.utils.similarity.operators import MSEntropyOperator

# 初始化工具
spectrum_similarity_tool = SpectrumSimilarity(
    sim_operator=MSEntropyOperator,
    num_cuda_workers=4,
    num_dask_workers=4,
    work_device='auto',
    operator_kwargs=None,
    dask_mode='threads'
)

# 处理单层结构数据
result = spectrum_similarity_tool.run(query, ref)

# 处理嵌套结构数据
nested_result = spectrum_similarity_tool.run_by_queue(query_queue, ref_queue)
```

### <a id="peak-mz-search-anchor"></a> PeakMZSearch
基于m/z值的质谱峰搜索工具箱，封装m/z峰搜索流程，提供配置化的计算参数管理。

#### 核心功能
- 封装m/z峰搜索流程
- 提供配置化的计算参数管理
- 支持设备自动分配策略
- 批处理并行计算
- 嵌套数据结构处理

#### 配置参数
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| mz_tolerance | float | 3 | m/z容差值（ppm或Da） |
| mz_tolerance_type | Literal['ppm', 'Da'] | 'ppm' | 容差类型 |
| RT_tolerance | float | 0.1 | RT容差值（以min为单位） |
| adduct_co_occurrence_threshold | int | 1 | 加成共现过滤阈值 |
| chunk_size | int | 5120 | 并行处理分块大小 |
| work_device | Union[str, torch.device, Literal['auto']] | 'auto' | 计算设备（自动推断策略：优先使用输入数据所在设备） |
| output_device | Union[str, torch.device, Literal['auto']] | 'auto' | 输出设备（默认与计算设备一致） |

#### 数据输入
`PeakMZSearch`工具支持两种输入模式：

1. **单层结构输入**:
   - 输入格式: `qry_ions: torch.Tensor`, `ref_mzs: torch.Tensor`, `qry_RTs: Optional[torch.Tensor]`, `ref_RTs: Optional[torch.Tensor]`
   - 形状: 
     - `qry_ions`: (N_q,)
     - `ref_mzs`: (N_r,)
     - `qry_RTs`: (N_q,)
     - `ref_RTs`: (N_r,)
   - 数据类型: `torch.float32`
   - 示例:
     ```python
     qry_ions = torch.tensor([1.0, 2.0, 3.0])
     ref_mzs = torch.tensor([1.1, 2.1, 3.1])
     qry_RTs = torch.tensor([0.1, 0.2, 0.3])
     ref_RTs = torch.tensor([0.11, 0.21, 0.31])
     ```

2. **嵌套结构输入**:
   - 输入格式: `qry_ions_queue: List[torch.Tensor]`, `ref_mzs_queue: List[torch.Tensor]`, `qry_RTs_queue: Optional[List[torch.Tensor]]`, `ref_RTs_queue: Optional[List[torch.Tensor]]`
   - 形状: 
     - `qry_ions_queue`: 每个元素为 (N_q,)
     - `ref_mzs_queue`: 每个元素为 (N_r,)
     - `qry_RTs_queue`: 每个元素为 (N_q,)
     - `ref_RTs_queue`: 每个元素为 (N_r,)
   - 数据类型: `torch.float32`
   - 适用于处理多个样本或实验的批量峰搜索操作
   - 示例:
     ```python
     qry_ions_queue = [
         torch.tensor([1.0, 2.0, 3.0]),  # 查询组1
         torch.tensor([4.0, 5.0, 6.0])   # 查询组2
     ]
     ref_mzs_queue = [
         torch.tensor([1.1, 2.1, 3.1]),  # 参考组1
         torch.tensor([4.1, 5.1, 6.1])   # 参考组2
     ]
     qry_RTs_queue = [
         torch.tensor([0.1, 0.2, 0.3]),  # 查询组1的RT
         torch.tensor([0.4, 0.5, 0.6])   # 查询组2的RT
     ]
     ref_RTs_queue = [
         torch.tensor([0.11, 0.21, 0.31]),  # 参考组1的RT
         torch.tensor([0.41, 0.51, 0.61])   # 参考组2的RT
     ]
     ```

#### 数据输出
`PeakMZSearch`工具的输出格式根据调用方法不同而有所区别：

1. **run()方法输出** (单层结构处理):
   - 输出类型: `Tuple[torch.Tensor, torch.Tensor]`
   - 形状: 
     - 匹配索引张量：(M, 2)，每行为 (qry_idx, ref_idx)
     - 匹配误差张量：(M,)，对应每个匹配的 m/z 误差（ppm或Da）
   - 数据类型: `torch.long` for indices, `torch.float32` for deltas
   - 设备位置: 默认与计算设备相同，可通过`output_device`参数指定
   - 示例:
     ```python
     indices, deltas = peak_mz_search_tool.run(qry_ions, ref_mzs)
     # indices.shape  # torch.Size([M, 2])
     # deltas.shape   # torch.Size([M,])
     ```

2. **run_by_queue()方法输出** (嵌套结构处理):
   - 输出类型: `List[Tuple[torch.Tensor, torch.Tensor]]`
   - 结构特点: 保持与输入相同的嵌套层级
   - 每个元素的形状: 
     - 匹配索引张量：(M, 2)，每行为 (qry_idx, ref_idx)
     - 匹配误差张量：(M,)，对应每个匹配的 m/z 误差（ppm或Da）
   - 示例:
     ```python
     # 输入结构: [ [查询组1,查询组2], [查询组3] ]
     nested_result = peak_mz_search_tool.run_by_queue(qry_ions_queue, ref_mzs_queue, qry_RTs_queue, ref_RTs_queue)
     # nested_result = [ (torch.Size([M1, 2]), torch.Size([M1,])), (torch.Size([M2, 2]), torch.Size([M2,])) ]
     ```

3. **非标准的使用情况**

    无论是对于`run`还是`run_by_queue`,均支持多维度张量的输入。区别于标准使用方式输入两个一维张量，在输入多维张量作为Query和Ref时，输出张量的结构也会被拓展：
    ```python
    query: torch.Tensor = ... # (dim_0,dim_1,...,dim_q)
    ref: torch.Tensor = ... # (dim_0,dim_1,...,dim_r)
    indices, deltas = peak_mz_search_tool.run(query, ref)
    indices.shape  # (dim_0,dim_1,...,dim_q,dim_q+1,dim_q+2,...,dim_q+dim_r+1)
    deltas.shape   # (dim_0,dim_1,...,dim_q,dim_q+1,dim_q+2,...,dim_q+dim_r+1)
    ```
    如上述例子所示，在计算过程中，结果张量其实是输入张量的一个错位广播的结果，在合理设计的情况下，用户可以给每一个dim设置特定的“意义”，这样就会返回每一种“意义”下的索引结果。
    
    一个常见的情况是带加合物推理的碎片搜索：
    ```python
    query: torch.Tensor = ... # (n_query,)
    ref: torch.Tensor = ... # (n_ref,n_adduct)
    indices, deltas = peak_mz_search_tool.run(query, ref)
    indices.shape  # (n_query,n_ref,n_adduct)
    deltas.shape   # (n_query,n_ref,n_adduct)
    ```
    通过以上技巧，即可快速在搜索过程中确定命中来源何种加合物。

#### 使用示例
```python
from masslib4search.snaps.MassSearchTools.utils.toolbox import PeakMZSearch

# 初始化工具
peak_mz_search_tool = PeakMZSearch(
    mz_tolerance=3,
    mz_tolerance_type='ppm',
    RT_tolerance=0.1,
    adduct_co_occurrence_threshold=1,
    chunk_size=5120,
    work_device='auto',
    output_device='auto'
)

# 处理单层结构数据
result_indices, result_deltas = peak_mz_search_tool.run(qry_ions, ref_mzs, qry_RTs, ref_RTs)

# 处理嵌套结构数据
nested_result = peak_mz_search_tool.run_by_queue(qry_ions_queue, ref_mzs_queue, qry_RTs_queue, ref_RTs_queue)
```

### <a id="peak-pattern-search-anchor"></a> PeakPatternSearch
基于质谱图结构的模式匹配搜索工具箱，封装质谱图结构的模式匹配搜索流程，提供配置化的计算参数管理，支持设备自动分配策略、批处理并行计算以及嵌套数据结构处理。

#### 核心功能
- 封装质谱图结构的模式匹配搜索流程
- 提供配置化的计算参数管理
- 支持设备自动分配策略
- 批处理并行计算
- 嵌套数据结构处理

#### 配置参数
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| loss_tolerance | float | 0.1 | 中心丢失边匹配的容差阈值（单位：Da） |
| mz_tolerance | float | 3 | m/z容差值（ppm或Da） |
| mz_tolerance_type | Literal['ppm', 'Da'] | 'ppm' | 容差类型 |
| chunk_size | int | 5120 | 并行处理分块大小 |
| num_workers | Optional[int] | 4 | 工作进程数 |
| work_device | Union[str, torch.device, Literal['auto']] | 'auto' | 计算设备（自动推断策略：优先使用输入数据所在设备） |
| output_device | Union[str, torch.device, Literal['auto']] | 'cpu' | 输出设备（强制保留在CPU） |

#### 数据输入
`PeakPatternSearch`工具支持嵌套结构输入：

- 输入格式: `qry_mzs_queue: List[List[torch.Tensor]]`, `refs_queue: List['SpectrumPatternWrapper']`
- 形状: 
  - `qry_mzs_queue`: 外层列表表示不同批次，内层列表为单批次的查询，每个元素为 (N_q,) 的张量
  - `refs_queue`: 对应每个批次的参考图包装类列表，每个包装类包含预定义的质谱图和损失值
- 数据类型: `torch.float32`
- 示例:
  ```python
  query_mzs_queue = [
      [torch.tensor([149.05970595, 277.15869171, 295.1692564, 313.17982108, 373.20095045, 433.22207982, 581.27450931])]  # 查询组1
  ]
  graph_0 = nx.Graph()
  graph_0.add_edges_from([
      (0, 1, {'type':0}),
  ])
  graph_1 = nx.Graph()
  graph_1.add_edges_from([
      (0, 1, {'type':0}),
      (1, 2, {'type':0}),
  ])
  losses = pd.Series([60.0],index=[0])
  refs_queue = [
      SpectrumPatternWrapper(
          graphs=pd.Series([graph_0, graph_1]),
          losses=losses
      )
  ]
  ```

#### 数据输出

- **run()方法输出** (单层结构处理):
  - 输出类型: `List[torch.Tensor]`
  - 形状: 
    - 匹配索引张量：(M,), 对应每个匹配的参考图索引
  - 设备位置: 输出结果强制保留在CPU
  - 示例:
    ```python
    # 输入结构: [ [查询组1] ]
    result = peak_pattern_search_tool.run(query_mzs_queue[0][0], refs_queue[0])
    # result = [torch.Size([M,])]
    ```

- **run_by_queue()方法输出** (嵌套结构处理):
  - 输出类型: `List[List[torch.Tensor]]`
  - 结构特点: 保持与输入相同的嵌套层级
  - 每个元素的形状: 
    - 匹配索引张量：(M,), 对应每个匹配的参考图索引
  - 设备位置: 输出结果强制保留在CPU
  - 示例:
    ```python
    # 输入结构: [ [查询组1], [查询组2] ]
    nested_result = peak_pattern_search_tool.run_by_queue(query_mzs_queue, refs_queue)
    # nested_result = [ [torch.Size([M1,])], [torch.Size([M2,])] ]
    ```

#### 使用示例
```python
from masslib4search.snaps.MassSearchTools.utils.toolbox import PeakPatternSearch, SpectrumPatternWrapper
import torch
import pandas as pd
import networkx as nx

# 构建测试数据
query_mzs_queue = [
    [torch.tensor([149.05970595, 277.15869171, 295.1692564, 313.17982108, 373.20095045, 433.22207982, 581.27450931])]  # 查询组1
] * 2
graph_0 = nx.Graph()
graph_0.add_edges_from([
    (0, 1, {'type':0}),
])
graph_1 = nx.Graph()
graph_1.add_edges_from([
    (0, 1, {'type':0}),
    (1, 2, {'type':0}),
])
losses = pd.Series([60.0],index=[0])
refs_queue = [
    SpectrumPatternWrapper(
        graphs=pd.Series([graph_0, graph_1]),
        losses=losses
    )
] * 2

# 初始化工具
peak_pattern_search_tool = PeakPatternSearch(
    loss_tolerance=0.1,
    mz_tolerance=3,
    mz_tolerance_type='ppm',
    chunk_size=5120,
    num_workers=None,
    work_device='auto',
    output_device='cpu'
)

# 处理单组搜索
results = peak_pattern_search_tool.run(query_mzs_queue[0], refs_queue[0])
print(results)
# 输出:
# [torch.Tensor([0,1])]

# 处理嵌套结构数据
nested_result = peak_pattern_search_tool.run_by_queue(query_mzs_queue, refs_queue)
print(nested_result)
# 输出：
# [[torch.Tensor([0,1])],[torch.Tensor([0,1])]]
```

### <a id="embedding-similarity-search-anchor"></a> EmbeddingSimilaritySearch
嵌入向量相似度搜索工具盒，输入嵌入向量并指定相似度算子，计算查询向量与参考向量之间的相似度矩阵。

#### 核心功能
- 封装嵌入向量相似度搜索流程
- 提供配置化的计算参数管理
- 支持设备自动分配策略
- 批处理并行计算
- 嵌套数据结构处理

#### 配置参数
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| sim_operator | Type[EmbbedingSimilarityOperator] | CosineOperator | 用于计算嵌入向量之间相似度的算子 |
| chunk_size | int | 5120 | 并行计算时每次处理的向量块大小 |
| top_k | Optional[int] | None | 每个查询向量返回前top_k个最相关的参考向量的索引和相似度分数 |
| batch_size | int | 128 | 批处理大小 |
| num_workers | int | 4 | 数据预处理并行度 |
| work_device | Union[str, torch.device, Literal['auto']] | 'auto' | 计算设备（自动推断策略：优先使用输入数据所在设备） |
| output_device | Union[str, torch.device, Literal['auto']] | 'auto' | 结果存储设备（默认与计算设备一致） |
| operator_kwargs | Optional[dict] | None | 相似度计算算子的额外参数 |

**相似度算子**

在masslib4search/snaps/MassSearchTools/utils/similarity/operators.py中，我们预定义了以下算子：
  - CosineOperator (默认使用)：余弦相似度
  - DotOperator：点积
  - JaccardOperator：杰卡德相似度
  - TanimotoOperator：谷本系数
  - PearsonOperator：皮尔逊相关系数

所有计算向量相似度的算子均是`EmbbedingSimilarityOperator`的子类，用户可以自定义算子实现自定义相似度计算。
`EmbbedingSimilarityOperator`类的详细接口见masslib4search/snaps/MassSearchTools/utils/similarity/operators/ABC_operator.py

#### 数据输入
`EmbeddingSimilaritySearch`工具支持两种输入模式：

1. **单层结构输入**:
   - 输入格式: `query: torch.Tensor`, `ref: torch.Tensor`
   - 形状: 
     - `query`: (n_q, dim)
     - `ref`: (n_r, dim)
   - 数据类型: `torch.float32`
   - 示例:
     ```python
     query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
     ref = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
     ```

2. **嵌套结构输入**:
   - 输入格式: `query_queue: List[torch.Tensor]`, `ref_queue: List[torch.Tensor]`
   - 形状: 
     - 每个`query_queue`元素: (n_q, dim)
     - 每个`ref_queue`元素: (n_r, dim)
   - 数据类型: `torch.float32`
   - 适用于处理多个样本或实验的批量相似度搜索操作
   - 示例:
     ```python
     query_queue = [
         torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 查询组1
         torch.tensor([[5.0, 6.0], [7.0, 8.0]])   # 查询组2
     ]
     ref_queue = [
         torch.tensor([[9.0, 10.0], [11.0, 12.0]]),  # 参考组1
         torch.tensor([[13.0, 14.0], [15.0, 16.0]])   # 参考组2
     ]
     ```

#### 数据输出
`EmbeddingSimilaritySearch`工具的输出格式根据调用方法不同而有所区别：

1. **run()方法输出** (单层结构处理):
   - 输出类型: `Tuple[torch.Tensor, torch.Tensor]`
   - 形状: `(n_q, n_r)` 对于索引矩阵，`(n_q, n_r)` 对于相似度分数矩阵
   - 数据类型: `torch.float32` 对于相似度分数矩阵，`torch.long` 对于索引矩阵
   - 设备位置: 默认与计算设备相同，可通过`output_device`参数指定
   - 示例:
     ```python
     # 输入2个查询向量和2个参考向量 → 相似度矩阵和索引矩阵 (2x2)
     indices, scores = embedding_similarity_search_tool.run(query, ref)
     indices.shape  # torch.Size([2, 2])
     scores.shape  # torch.Size([2, 2])
     ```

2. **run_by_queue()方法输出** (嵌套结构处理):
   - 输出类型: `List[Tuple[torch.Tensor, torch.Tensor]]`
   - 结构特点: 保持与输入相同的嵌套层级
   - 每个元素的形状: `(n_q, n_r)` 对于索引矩阵，`(n_q, n_r)` 对于相似度分数矩阵
   - 示例:
     ```python
     # 输入结构: [ [查询组1,查询组2], [查询组3] ]
     results = embedding_similarity_search_tool.run_by_queue(query_queue, ref_queue)
     results[0][0].shape  # 第一组结果的索引矩阵形状
     results[0][1].shape  # 第一组结果的相似度矩阵形状
     results[1][0].shape  # 第二组结果的索引矩阵形状
     results[1][1].shape  # 第二组结果的相似度矩阵形状
     ```

#### 使用示例
```python
from masslib4search.snaps.MassSearchTools.utils.toolbox import EmbeddingSimilaritySearch
from masslib4search.snaps.MassSearchTools.utils.similarity.operators import CosineOperator

# 初始化工具
embedding_similarity_search_tool = EmbeddingSimilaritySearch(
    sim_operator=CosineOperator,
    chunk_size=1024,
    top_k=2,
    batch_size=128,
    num_workers=4,
    work_device='auto',
    output_device='auto',
    operator_kwargs=None
)

# 处理单层结构数据
indices, scores = embedding_similarity_search_tool.run(query, ref)

# 处理嵌套结构数据
nested_results = embedding_similarity_search_tool.run_by_queue(query_queue, ref_queue)
```

