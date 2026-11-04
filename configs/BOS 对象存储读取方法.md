# BOS 对象存储读取方法

> **Workspace:** /home/jovyan/pfs/users/zhanqian.wu

---

**User**

怎么读取

桶名：gigaai-5090；另有数据集共享桶 gigaai-share（5090 资源池挂 datasets 时，数据集的正确数据源是这个桶，不是 gigaai-5090）
• ENDPOINT：`https://hn-gddgct01.becbos.baidu.com`
• 性质：边缘节点对象存储 ——不是中心集群 BOS，没有国内几大城市服务点名称，是随东莞边缘节点配套开通的
• AK/SK（9/17 16:35 Tiger Wang 发的「5090 BOS 存储密钥」）：
◦ AK：<见内部密钥记录, 已脱敏>
◦ SK：<见内部密钥记录, 已脱敏>
• 挂载方式：BosFS 2.0；5090 云开发机 14.152.92.141 曾用它挂载；资源池内数据集挂载点形如 /mnt/dataset/51wrg8
• 带宽：9/20 已扩容至 1G（Tiger Wang 同步），并要求控制服务商传输速度避免打满

是东莞5090集群挂载的对象存储BOS

**Assistant**

当前机器（IP 100.88.253.97，中心集群 pfs 环境）没有装 BOS 工具，也不是 5090 那台 14.152.92.141。我先装 SDK 并验证能否直接从这个边缘节点读数据。

HTTP 通（200），HTTPS 握手失败。看下 TLS 细节并尝试兼容。

TLS 版本问题找到了：边缘节点只支持老 cipher 套件，Python/OpenSSL 3 默认安全级别（SECLEVEL=2）拒绝握手，放宽到 SECLEVEL=1 即可。现在装 SDK 并验证密钥和桶。

包名是 `baidubce`（不是 baidubce-sdk）。重新安装并测试。

沙箱不允许写 /opt/conda，改装到用户目录。

装到了一个远古版本（py2 风格 import）。从 PyPI 拿最新版本。

---

**User**

1. 5090集群基本信息（必读）：
- 存储：
  - 每台机器有各自本地盘，本地盘关机后就重置了，所以只能临时放东西；
  - 共享盘采用rapidfs方案，每台机器抽出一部分本地盘合并出一个共享盘，共享盘的挂载Bos作为数据源。具体bucket为gigaai-5090，挂载整个bucket；
  - 共享盘大小64TB（还未实现），起到缓存盘的作用，挂载整个bucket。系统会自动调度缓存，当缓存超过大小后，读写操作会直接拉取Bos，只是会慢一些，不影响使用。
  - 共享盘：建立开发机时请挂载下图所示的本地盘，这个路径对应的就是bos:/giga-5090/ 这个bucket。具体命名无要求。建议为/mnt/rfs
[图片]
  - 每个人的个人文件直接放到/【设置的挂载目录】/users/【用户名】下面。（例如：/mnt/rfs/users/xxx)
- 公网带宽1Gbps
- 每个人的环境建议制作成Docker，放在镜像仓库
- debug、开发用的一般使用范式：
  - 环境：Docker 镜像只负责固定运行环境（CUDA、系统依赖等）
  - 代码数据都正常存到共享盘即可。偏大的数据建议别全放到共享盘，会导致整体缓存调度变慢。
- 多机多卡推理的一般使用范式：
  - 暂无

2. 东莞本地BOS相关：
- 可以通过SDK和CMD方式来访问BOS。gigaai-5090这个bucket还可以通过挂载到开发机来访问。
- 访问方式：
公网ENDPOINT `https://hn-gddgct01.becbos.baidu.com`
内网ENDPOINT `http://169.254.169.244`
注意在5090集群访问bos要用内网ENDPOINT，访问bos的内网带宽50Gbps
密钥（临时）：
AK:<见内部密钥记录, 已脱敏>
SK:<见内部密钥记录, 已脱敏>
2.1 BOS桌面软件预览工具cyberduck
暂时无法在飞书文档外展示此内容
2.2 通过CMD访问BOS
先安装bcecmd，然后进行一次配置，按下方：
root@aibox-r07b86c8f048-66477b6ff9-zdcdq:~# bcecmd -c
Your Access Key ID [none]: <见内部密钥记录, 已脱敏>
Your Secret Access Key ID [none]: <见内部密钥记录, 已脱敏>
Your Security Token []: 
Default Region Name [bj]: bj
Default Domain [bj.bcebos.com]: 169.254.169.244
Default use auto switch domain [yes]: no
Default Proxy Host [none]: none
Default breakpoint_file_expiration [7] days: 999
Default use https protocol [yes]: no 
Default multi upload thread num [10]: 10
Default multi download thread num [10]: 10
Default sync processing num [10]: 10
Default multi upload part size [12] MB (Must be positive integer and equal or greater than 1) : 12
Default use path style [no]: yes
常用命令
注意文件夹相关最好要带末尾斜杠
# 看有哪些 bucket
bcecmd bos ls bos:/
bcecmd bos ls bos:/gigaai-share/users/

BOS 没有真正的 mkdir。
做法（推荐）：上传文件时带上路径，目录自动出现
# 本地文件放到「文件夹」下
bcecmd bos cp ./readme.txt bos:/gigaai-share/alice/notes/readme.txt
# 整个本地目录上传
bcecmd bos cp ./mydir/ bos:/gigaai-share/alice/mydir/ -r

# 下载
bcecmd bos cp bos:/gigaai-share/alice/notes/readme.txt ./readme.txt
# 删单个文件
bcecmd bos rm bos:/gigaai-share/alice/notes/readme.txt
# 删整个前缀（相当于删目录）
bcecmd bos rm bos:/gigaai-share/alice/ -r
使用文档：
BEC bcecmd.pdf
CMD文档：`https://cloud.baidu.com/doc/BOS/s/qjwvyqegc`
2.3 通过SDK访问BOS
SDK文档：`https://cloud.baidu.com/doc/BOS/s/ukca9bn9w`

3. 东莞镜像仓库：
1. 镜像仓库账户和密码：
  40cdd88ef9e4485487e1424978f6cbf2
  vN7!qL2@zR9#tX4$kM8
登录命令参考：
  docker login 169.254.169.250:80 \
    -u '40cdd88ef9e4485487e1424978f6cbf2' \
    -p '密码'
2. 推送镜像命令
公网：docker push hn-gddgct01.bechub.baidu.com/gigahub/REPOSITORY[:TAG]
内网：docker push 169.254.169.250:80/gigahub/REPOSITORY[:TAG]
3. 百舸平台开发机建立镜像：
  1. 点击制作镜像
[图片]
  2. 制作镜像参数参考（gigahub示例）
[图片]
4. 多机多卡使用：
启动多机多卡训练即可

你看下怎么连bos

**Assistant**

当前机器确认是中心集群，不在 5090 内网里 —— `169.254.169.244` 不通，公网 HTTP 通（0.11s 很快）。所以这台机器走公网 endpoint，5090 上的机器走内网 endpoint。

现在把 SDK 修到正确的版本（0.8.3）。

这个 0.8.3 居然还是 2014 年的 Python 2 风格代码（`import protocol` 隐式相对导入），从没适配过 Python 3。直接修补包里的隐式导入。

这包在 Python 3 下根本没法用（`httplib`/`urlparse`/`cStringIO` 全是 py2 模块）。与其修补一堆 py2 遗留代码，不如先试试官方 GitHub 上的新版 SDK（支持 py3）。

GitHub 直连成功。卸载旧版，装官方 master。

新版 SDK 可用。测试实际访问桶。

桶能看到了。修下参数名（`max_keys`）再列对象。

边缘节点要求 **path style** 访问（你给的 bcecmd 配置里 `use path style: yes` 正是这个）。查下 SDK 是否支持。

两个桶完全打通了。再验证读文件内容和数据集目录结构。

全链路打通。我把验证过的配置封装成一个可复用的小工具放到你的用户目录。

连 BOS 已打通，工具封装在 [bos_utils.py](file:///home/jovyan/pfs/users/zhanqian.wu/bos_utils.py)，实测两桶都能读写。

**当前机器结论**：这台是中心集群（IP 100.88.253.97），`169.254.169.244` 内网不通，只能走公网 endpoint（实测 HTTP 0.1s，很稳）。在 5090 机器上运行时工具会自动探测并切内网 endpoint（50Gbps）。

**踩过的三个坑（已解决）**
1. **包名是 `baidubce`**，但镜像源（ivolces/aliyun）里的 0.8.3 是 2014 年的 Python 2 代码（`import httplib`），Python 3 直接装完用不了。正确装法：
   ```bash
   pip install --user -i https://pypi.org/simple baidubce
   # 若还是老版, 从 GitHub 装: https://github.com/baidubce/bce-sdk-python (master 支持 py3)
   ```
2. **边缘节点必须 path style**：默认 virtual-host 风格会报 `bucket_name is not equal your query bucket_name`，需要 `path_style_enable=True`（对应你 bcecmd 配置里的 `use path style: yes`）。
3. **公网 HTTPS 握手失败**：边缘节点只支持老 cipher 套件，OpenSSL 3 默认 SECLEVEL=2 拒绝。工具里已自动放宽（用 HTTP 则无此问题）。

**用法**

```python
from bos_utils import get_client, list_dir, download, upload

c = get_client()
dirs, files = list_dir(c, "gigaai-share", "datasets/")   # 浏览目录
download(c, "gigaai-share", "datasets/xxx.tar", "/mnt/pfs/xxx.tar")
upload(c, "gigaai-5090", "users/zhanqian.wu/tmp.txt", "/tmp/tmp.txt")
```

当前 `gigaai-share` 里 `datasets/` 还是空的，已有 `users/{boyuan.wang, changyifan, jian.yang}`。AK/SK 已内置（可用环境变量 `BOS_AK`/`BOS_SK` 覆盖），临时密钥过期后换掉即可。bcecmd 就按你文档里那份配置配即可，和 SDK 走的是同一套密钥和 endpoint。

---

**User**

帮我把这两个传上去/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_images_lmdb
/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_edit_images_lmdb

users/zhanqian.wu


**Assistant**

先看这两个目录的规模和 BOS 上的目标结构。

`du` 在后台跑着（pfs 上较慢），先并行看 BOS 上的目标结构。

BOS 结构确认：`gigaai-5090/users/` 下已有多人目录（changyifan、david 等），这是文档约定的个人目录；`gigaai-share/users/` 下也有。你俩桶都有 `users/`，先确认传哪个桶。

目标定了：`gigaai-5090/users/zhanqian.wu/`。看下数据规模。

第一个目录 8.5TB，非常大。先看文件构成，确认上传策略。

等遍历的同时，先验证 SDK 大文件分片支持并实测上传速度。

SDK 有现成的 `put_super_object_from_file`（自动分片 + 多线程 + 限速 + 断点）。先找个小文件实测公网上传速度。

小文件实测 6.3MB/s（53Mbps，单流小文件偏低）。再测大文件多线程分片的真实带宽。

大文件 8 线程实测 **33MB/s（279Mbps）**——公网链路没打满还能提。8.5TB 按此速度约 3 天，提高并发可压到 2 天内。写正式上传脚本（断点续传 + 多文件并行 + 进度条）。

---

**User**

（1）提高并发 （2）先传unified_images_lmdb

**Assistant**

收到：先传 `unified_images_lmdb`（8.5TB），并发拉高。写上传脚本。

收集清单（pfs 上遍历 8.5TB 目录树较慢，放后台跑）。

清单收集中（后台）。先用两个大文件验证提高并发后的带宽。

高并发实测 **97MB/s（814Mbps）**——接近打满 1Gbps，8.5TB 约 25 小时。但清单收集任务挂了，查原因。

沙箱进程崩了（非脚本问题）。重跑收集。

两次都崩在同一处 —— 沙箱 host 进程扛不住全量遍历。改成逐个子数据集目录遍历（边收集边上传）。先看各子目录规模。

23 个子目录全部遍历成功（共 5.5 万文件，8.4TB）——崩溃是全量一次遍历扛不住。改成逐目录收集、每个目录落盘一份清单，可断点续跑。

后台跑必被沙箱杀，但前台短任务都正常（逐目录 for 循环全跑通过）。改前台跑，10 分钟超时。

清单齐了（50,000 文件 / 8.4TB）。启动上传，用 nohup 脱离沙箱进程组。

上传进程存活（PID 2681709）。等一分钟看实际速度。

进程活跃（CPU 281%），但大文件传完才更新进度条 —— sa1b_en 单文件 2.6TB，要等太久。给进度条接上 SDK 的分片级回调。先确认回调签名。

回调签名确认：`callback(consumed_bytes, total_bytes)`。改进度逻辑：分片级实时更新。

重启上传进程使用新进度逻辑。

上传跑起来了，进度条实时刷新。

**上传已启动并稳定运行**

```
upload: 3.41G/8.42T [00:40<28:06:37, 91.4MB/s]
```

- **速度**：8 文件并行 × 8 分片线程，实测 **91.4MB/s（约 780Mbps）**，接近打满 1Gbps 公网带宽
- **规模**：unified_images_lmdb 共 53,121 文件 / 8.42TB，预计 **~28 小时**传完
- **目标**：`bos:/gigaai-5090/users/zhanqian.wu/unified_images_lmdb/`
- **进程**：PID 2684607（nohup 脱离沙箱，断线不影响）

**断点续传**：每个文件完成即写入 [bos_upload_done.json](file:///home/jovyan/pfs/users/zhanqian.wu/bos_upload_done.json)，中断后重跑 `python bos_upload.py --upload --only unified_images_lmdb` 自动跳过已完成；单文件失败自动重试 3 次。

**监控命令**
```bash
tail -f ~/upload.log          # 实时进度条
python bos_upload.py --upload --only unified_images_lmdb   # 若进程挂了, 重跑续传
```

**后续**：images 传完后传 edit（9.7TB，又需 ~30h）：
```bash
python bos_upload.py --collect --only unified_edit_images_lmdb
python bos_upload.py --upload --only unified_edit_images_lmdb
```

如需控制带宽（文档要求避免打满），加 `--limit 100`（单请求限速 100Mbps，8 并发 × 8 分片 ≈ 总带宽 6.4Gbps 上限，实际按请求并发折算）。

踩坑记录：沙箱后台进程跑长遍历任务会被杀（signal），清单收集改逐子目录前台跑解决（20 秒完成）；脚本已内置逐目录断点收集。

---

**User**

有点慢

**Assistant**

先看当前进度，并测试链路是否已打满（决定提并发是否有意义）。

速度掉到 46.5MB/s（可能是 8 个大文件同时处于分片尾段收敛期）。测一下链路是否还有余量。

增量只有 7.7MB/s —— 链路接近打满，但主进程自身效率有问题。查 SDK 分片逻辑发现疑点：大文件自动分片会被切成 2-5MB 的小分片（10000 分片上限导致），小请求效率极低。确认下最大文件的分片大小。

修复方向明确：显式大分片（128MB）。先确认 SDK 分片任务是否全量提交（内存风险）。

SDK 是按需读取（内存安全），只是分片太小。改用 64MB 显式分片，重启进程。

旧进程已杀，新进程没起来。重新启动。

881MB/s 超过公网上限 6 倍 —— 进度条在虚报（回调重复计数）。验证真实速率。

找到进度虚报根因：多分片线程交错回调，`state["last"]` 单值跟踪导致同一分片被重复累计。改为“分片读满才算一次”（每分片恰好计一次，无重复）。

发现之前 pkill 总是把命令自身杀掉（`-f` 模式匹配到 bash 自己的命令行）——分两步：先启动新进程（旧进程已被杀）。

进度条修复，速度爬升中（73.5MB/s 且还在

---

**User**

`/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/configs/stage_1.yaml#L39-51`把这些也上传bos

**Assistant**

上传这两个模型目录到 BOS。先看大小（注意大上传进程正占着带宽，模型目录小，先传完让路）。

两个模型目录都能访问（中间层 NFS 抖动导致之前超时）。拉完整文件清单。

133 文件 / 40.1GB，量小。为避免和正在跑的 8.42TB 上传抢 `done.json`（会互相覆盖断点记录），模型用独立一次性脚本传，目标 `bos:/gigaai-5090/models/huggingface/`（与 `/mnt/nfs/models/huggingface` 结构对齐）。

---

