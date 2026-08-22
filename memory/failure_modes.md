# failure_modes — 实现失败模式与实践建议（l_t(h)）

> Coding Agent / 运维操作前必读。每次事故后由 Diagnostic 步骤追加。
> 格式：`- <失败模式>: <建议>（来源）`

## 环境与依赖

- conda 环境改名用 mv+sed 扫 bin 目录: 含路径字符串的 ELF 二进制被 sed 改短字节错位直接段错误。改名=新建环境重装。（setup）
- pip 装 torch 不锁版本: 本机驱动 CUDA 12.4，PyPI 最新 torch(2.13/cu130) 大版本不兼容。锁 `torch==2.10.0`（PyPI 默认即 cu128 构建，元数据 requires_dist 可验证），全走清华源。（setup）
- pip 命令漏 `-i 清华源`: nvidia 依赖会回落 pypi.org 直连，75KB/s 卡死。凡 pip 必带 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。（setup）
- 旧版 pip + pysocks 走代理报 urllib3 PoolKey TypeError: 别修 pip，改用 curl --socks5-hostname 直接下 wheel。（setup）

## 文件操作

- pkill/pgrep -f 模式匹配到自身命令行: 远程 shell 自杀、整条命令静默消失（两次"无输出"事故根源）。用 `[u]nzip` 方括号转义模式。（setup）
- 删除/移动前不检查占用: rm 掉了 pip 正在写入的 /tmp/pip-unpack-* 导致安装中断；mv 与后台解压竞争产生错乱布局。任何清理前先 pgrep 确认无活跃写入者。（setup）
- 长内联 SSH 命令中途断连则静默半执行: >30s 的远程操作一律 tmux + 完成标记文件（如 .EXTRACT_OK），验证以标记+计数为准，不以"没报错"为准。（setup）

## Git/账号

- GitHub push 被拒 email privacy: 提交邮箱必须用 `qkun-zh@users.noreply.github.com`。（setup）
