# MassLib4Search
用于快速构建可搜索质谱数据集的Python库，并提供常见的搜索任务与工具函数.

## 库结构
- `masslib4search.snaps.MassSearchTools`：定义了常见的搜索任务与工具函数。

## 工具函数
工具函数是MassLib4Search库对外暴露的最低级别的API，用户可以通过组合这些函数来构建自己的搜索业务逻辑。
以下是可用的函数列表：

- [`binning`](#binning-anchor)：质谱数据分箱处理工具，可将离散质谱峰（m/z）聚合为规整特征向量。

### <a id="binning-anchor"></a> binning 函数
质谱数据分箱处理工具，可将离散质谱峰（m/z）聚合为规整特征向量。

#### 功能特性

**智能设备调度**  
- 自动检测输入数据所在设备（CPU/CUDA）
- 支持手动指定计算设备（`work_device`）和输出设备（`output_device`）

**自定义分箱逻辑**  
- 自定义m/z范围 `(min_mz, max_mz)`
- 可调分箱精度 `bin_size`（单位：m/z）
- 三种强度聚合方式：`sum`（求和）、`max`（最大值）、`avg`（平均值）

**批量处理**  
- 内置并行计算（`num_workers`控制并发数，`batch_size`控制块大小）
- 自动分发并行实现（CPU使用Dask并行，CUDA使用Worker并行，每个Worker由4个CUDA流组成）

#### 参数说明
| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `mzs` | `List[torch.Tensor]` | 必填 | 质荷比列表（多批次输入） |
| `intensities` | `List[torch.Tensor]` | 必填 | 强度值列表（需与mzs维度一致） |
| `binning_window` | `Tuple[float,float,float]` | (50.0,1000.0,1.0) | (最小m/z, 最大m/z, 分箱步长) |
| `pool_method` | `Literal['sum','max','avg']` | "sum" | 强度聚合方法 |
| `batch_size` | `int` | 128 | 单次处理样本量 |
| `num_workers` | `int` | 4 | 并行使用的线程（CPU后端）/ CUDA流Worker（CUDA后端）数 |
| `work_device` | `Union[str, torch.device, 'auto']` | 'auto' | 计算设备（`auto` 将会自动使用输入数据所在设备） |
| `output_device` | `Union[str, torch.device, 'auto']` | 'auto' | 输出张量设备（'auto' 将会自动对齐到 `work_device`） |

#### 函数输出
- 输出为一个二维浮点型张量，形状：`(queue_lens, bins_num)`
  - `queue_lens`：输入队列长度（处理的质谱样本总数）
  - `bins_num`：分箱总数（根据`(max_mz - min_mz)/bin_size`计算）
- 设备位置自动对齐`output_device`参数配置

#### 使用示例
```python
from masslib4search.snaps.MassSearchTools.utils.embedding import binning

# 输入频谱队列
mzs_queue, intensities_queue = [...]  

# 执行分箱（GPU加速）
binned_spectrum = binning(
    mzs_queue,
    intensities_queue,
    binning_window=(50.0, 1000.0, 1.0),  # 1.0 Da分箱精度
    pool_method="max",                   # 使用最大池化处理bins
    work_device="cuda:0",               # 指定GPU计算（output_device默认与work_device一致）
)

print(binned_spectrum.shape)  # 输出形如：torch.Size([len(mzs_queue), 950])
```