# failure_modes — 实现失败模式与实践建议（l_t(h)）

> Coding Agent 动手前必读。Diagnostic Agent 在每次失败/超时后追加。
> 格式：`- <失败模式>: <建议>（来源节点）`

- 环境改名用 mv+sed 扫 bin 目录: 会把含路径字符串的 ELF 二进制改出字节错位直接段错误。改 conda 环境名一律新建环境重装，禁止 mv。（setup, S0001 前）
- pip 装 torch 不锁版本: 本服务器驱动为 CUDA 12.4（12040），PyPI 最新 torch（2.13+cu130）大版本不兼容 → cuda.is_available()=False。必须锁 `torch==2.10.x+cu128`（小版本兼容性已验证），来源阿里云 pytorch-wheels 镜像或官方 cu128 index。（setup, S0001 前）
- GitHub push 被拒 email privacy: 账户开了邮箱隐私保护时，提交邮箱必须是 `qkun-zh@users.noreply.github.com`。（setup, 初始化时）
- conda defaults 通道触发 TOS 拦截: 建环境用 `--override-channels -c conda-forge`，包一律走 pip + 清华源。（setup, S0001 前）
- 前台 SSH 跑长任务被本地中断即全灭: >1min 任务必须 tmux（AGENTS 规则1 的血泪验证）。（setup, S0001 前）
