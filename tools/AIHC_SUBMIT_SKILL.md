# yisu 集群任务提交实战手册（AIHC edge）

> 从 yisu-submit skill（boyuan.wang）+ 2026-11-14 两次全量数据转换任务实战总结。
> 适用：向 yisu 集群（resourcePoolId `cce-g6g1x2s2`）提交 PyTorchJob 自定义任务。

## 1. 连接与凭证

| 项目 | 值 |
|---|---|
| 调试机 | `ssh yisu_dev`（= 106.74.40.215:30873，走 Giga2504 跳板，root） |
| Endpoint | `https://cce-egsz.su.baidubce.com` |
| region / resourcePoolId | `edge` / `cce-g6g1x2s2`（yisu） |
| queue | `default` |
| GPU 资源名 | `baidu.com/h800_80g_cgpu` |
| 工具 | `/home/jovyan/pfs/users/boyuan.wang/misc/aihc/yisu-submit/scripts/aihc.py` |

凭证文件（含 `accesskey`、`secretkey`，可选 `sessionToken`）放私有目录，**不要打印、不要进命令行参数或仓库**：

```bash
mkdir -p /root/.aihc && chmod 700 /root/.aihc
umask 077
cat > /root/.aihc/credentials.json <<EOF
{"accesskey": "ALTAK...", "secretkey": "..."}
EOF
```

所有 `aihc.py` 调用：`--credentials` 放在子命令前。

```bash
export SKILL=/home/jovyan/pfs/users/boyuan.wang/misc/aihc/yisu-submit
export AIHC_CREDENTIALS=/root/.aihc/credentials.json
PY=python3
```

依赖：`bce-python-sdk`（调试机已装，验证 `python3 -c "import baidubce.auth.bce_v1_signer"`）。

## 2. request.json 生成（推荐脚本化，别在 heredoc 里拼）

模板：`$SKILL/assets/request.json`（已验证：1 GPU / 2C / 16G / shm 10G，pfs hostpath）。

**教训：在 `ssh '...'` 单引号串里用未加引号 heredoc 拼 JSON，`$`/`\\` 转义极易翻车
（实测踩过：`datasources` KeyError、SyntaxError、env 没导出）。**
把生成逻辑存成脚本文件再执行，参数用 `os.environ` 传。

需修改的字段：

```python
import json, os, shlex
from pathlib import Path
r = json.loads(Path(f"{os.environ['SKILL']}/assets/request.json").read_text())
r["name"] = os.environ["JOB_NAME"]                       # 全局唯一，含时间戳
r["jobSpec"]["resources"][1]["quantity"] = 16            # cpu
r["jobSpec"]["resources"][2]["quantity"] = 64            # memory (GB)
# 需要 /mnt/nfs 时追加 hostpath（镜像不含共享盘，pfs 默认已挂）
r["datasources"].append({"type": "hostpath", "sourcePath": "/mnt/nfs",
                         "mountPath": "/mnt/nfs", "name": "nfs",
                         "options": {"readOnly": False}})
# command 用 bash -lc + shlex.quote 包多行脚本，不要写 imageConfig
r["jobSpec"]["command"] = "bash -lc " + shlex.quote(cmd)
```

任务 command 模板（日志/退出码落 PFS 独立目录，便于事后查）：

```bash
set -u
OUT=<PFS 上的任务目录>
exec > >(tee -a "$OUT/run.log") 2>&1
echo "JOB_START $(date -u +%FT%TZ) host=$(hostname)"
cd /home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools
# 依赖兜底
python3 -c "import pyarrow" 2>/dev/null || pip install pyarrow -q
python3 your_job.py ...
rc=$?
printf "%s\n" "$rc" > "$OUT/exit_code"
echo "JOB_END exit_code=$rc"
exit "$rc"
```

## 3. 查重 → 提交 → 观测

```bash
# 查重（必须，分页已处理；同名存在则不要重发）
$PY "$SKILL/scripts/aihc.py" --credentials "$AIHC_CREDENTIALS" find --name "$JOB_NAME"

# 提交（一次尝试目录只允许一次 POST；回执自动落 --output-dir）
$PY "$SKILL/scripts/aihc.py" --credentials "$AIHC_CREDENTIALS" submit \
    --request "$WORK/request.json" --output-dir "$WORK"
# 回执: submission-started.json / response.json / job.json / status.json

# 状态 / 事件
$PY "$SKILL/scripts/aihc.py" --credentials "$AIHC_CREDENTIALS" get --job-id "$JOB_ID"
$PY "$SKILL/scripts/aihc.py" --credentials "$AIHC_CREDENTIALS" events --job-id "$JOB_ID"

# 停止（按确切 job ID，停后必须核实终态 ManualTermination/Failed）
$PY "$SKILL/scripts/aihc.py" --credentials "$AIHC_CREDENTIALS" stop --job-id "$JOB_ID"
```

### 观测的坑（重要）

- **`aihc.py logs` 在本 endpoint 恒返回空**（连成功任务也是），别指望平台日志。
- 任务自己的 stdout 重定向到 PFS 的 run.log **有延迟**（Python 无 `-u` 时块缓冲 +
  PFS attr cache），`run.log` 迟迟不出现 ≠ 任务卡死。
- **判断任务是否真在干活，看产物**：PFS 的 `exit_code`/state 文件、输出目录里的
  shard/文件数（`find <dst> -name 'shard-*.tar' | wc -l`、`du -sh <dst>`）。
- POST 超时/5xx/回执缺 jobId = 结果不确定：先 `find` 精确回查再决定，别换名重发。

## 4. 疑似卡死时的 smoke 诊断法

先用最小任务（模板默认 2C/16G 即可）分别验证 PFS、NFS，全部带 `timeout`，
stdout 写本地盘 `/tmp`（避开被测对象）：

```bash
exec > >(tee -a /tmp/smoke.log) 2>&1
timeout 60 touch /home/jovyan/pfs/.../smoke_ok.txt; echo pfs_rc=$?
timeout 60 ls /mnt/nfs/data/zhanqian.wu/ >/dev/null 2>&1; echo nfs_rc=$?
```

结论看产物文件是否出现在共享盘（不是看 logs 接口）。实测 2026-11-13：
PFS/NFS 均正常，"卡死"实为日志缓冲误判，任务主体在正常跑。

## 5. 重试边界

- 明确参数拒绝且确认没创建任务 → 修复参数，保留旧回执，**新尝试目录**重试。
- 疑似卡死先 smoke、看产物，再决定 stop；stop 后核实终态才算停干净。
- 支持**断点续传**的任务（如 lmdb_to_tar.py）直接原样重提即可，从断点继续，
  已产出 shard 不重做；同目录并发跑两个实例会冲突，先停旧的再提新的。

## 6. 已验证的资源参数

| 场景 | replicas×GPU | cpu | memory | shm | 备注 |
|---|---|---|---|---|---|
| 单实例 smoke | 1×1 | 2 | 16 | 10 | endpoint 拒绝 shm<10G |
| CPU+IO 密集转换 | 1×1 | 16 | 64 | 10 | 16 workers，实测聚合 ~4GB/s（PFS→NFS） |
| 单实例测试 | - | - | - | - | `enableRDMA: false`、`hostNetwork: false`、`NCCL_IB_DISABLE=1` |

多节点训练才需要 RDMA/hostNetwork/NCCL 配置，参见原 SKILL.md。

## 7. 实例回执

- `unified-edit-lmdb2tar-20261113-190704`
  job: `pytorchjob-07fc231d-d587-4010-878f-8827af42cfca`
  日志: `pfs_data/lmdb2tar_runs/unified-edit-lmdb2tar-20261113-190704/`
- `unified-images-lmdb2tar-v2-193340`
  job: `pytorchjob-559a0dc2-1b25-4fe6-b787-418d036bfd17`
  转换脚本: `code/model_exp/tools/lmdb_to_tar.py`（tar 分片 + sidecar idx + parquet，
  断点续传，进度打日志）
