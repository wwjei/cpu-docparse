# GitHub CLI (gh) 使用指南

本文档记录 GitHub CLI 的安装、认证和常用操作，供 agent 或开发者在服务器/沙箱环境中非交互式操作 GitHub。

## 1. 安装

### Ubuntu / Debian (apt)

```bash
# 添加官方仓库
(type -p wget >/dev/null || sudo apt-get install wget -y) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt update \
  && sudo apt install gh -y
```

### 直接下载二进制（无 root 权限时）

```bash
# 以 v2.74.1 linux amd64 为例
cd /tmp
curl -sLO https://github.com/cli/cli/releases/download/v2.74.1/gh_2.74.1_linux_amd64.tar.gz
tar -xzf gh_2.74.1_linux_amd64.tar.gz
cp gh_2.74.1_linux_amd64/bin/gh /usr/local/bin/
gh --version
```

### macOS (Homebrew)

```bash
brew install gh
```

## 2. 认证（非交互式）

在 CI/沙箱/agent 环境中，无法打开浏览器，使用以下方式：

### 方式一：GH_TOKEN 环境变量（推荐）

```bash
export GH_TOKEN="github_pat_xxxxxxxxxxxx"

# 验证
gh auth status
```

设置后所有 `gh` 命令自动使用该 token，无需 `gh auth login`。

### 方式二：gh auth login --with-token

```bash
echo "github_pat_xxxxxxxxxxxx" | gh auth login --with-token
```

从 stdin 读取 token 并存储到 `~/.config/gh/hosts.yml`。

### Token 要求

| 操作 | 所需权限 (Fine-grained PAT) |
|------|---------------------------|
| git push/pull | Contents: Read and write |
| 创建/编辑 Issue | Issues: Read and write |
| 创建/编辑 PR | Pull requests: Read and write |
| 读取仓库信息 | Metadata: Read (自动包含) |

生成路径：GitHub → Settings → Developer settings → Fine-grained tokens → Generate new token

## 3. 常用操作

### 3.1 Git 推送（通过 gh 认证）

```bash
# 设置 gh 为 git 凭据助手
gh auth setup-git

# 之后正常 push 即可
git push origin main
```

或直接用 token URL：

```bash
git push "https://x-access-token:${GH_TOKEN}@github.com/OWNER/REPO.git" main
```

### 3.2 创建 Issue

```bash
gh issue create \
  --repo OWNER/REPO \
  --title "Issue 标题" \
  --body "Issue 正文（支持 Markdown）" \
  --label "enhancement"
```

从文件读取正文：

```bash
gh issue create \
  --repo OWNER/REPO \
  --title "标题" \
  --body-file ./issue_body.md
```

### 3.3 查看 Issue

```bash
gh issue list --repo OWNER/REPO
gh issue view 1 --repo OWNER/REPO
```

### 3.4 创建 PR

```bash
gh pr create \
  --repo OWNER/REPO \
  --title "PR 标题" \
  --body "描述" \
  --base main \
  --head feature-branch
```

### 3.5 调用任意 REST API

```bash
# GET
gh api /repos/OWNER/REPO/issues

# POST
gh api /repos/OWNER/REPO/issues \
  --method POST \
  --field title="标题" \
  --field body="正文"

# 带 JSON body
gh api /repos/OWNER/REPO/issues \
  --method POST \
  --input issue.json
```

### 3.6 仓库操作

```bash
gh repo view OWNER/REPO
gh repo clone OWNER/REPO
gh repo create my-repo --public
```

## 4. 本项目操作示例

```bash
export GH_TOKEN="your_token_here"

# 推送代码
cd /data
gh auth setup-git
git push origin main

# 创建架构验证 Issue
gh issue create \
  --repo wwjei/cpu-docparse \
  --title "[验证] AMD x86_64 架构性能验证" \
  --body-file docs/agent/issue_amd_x86_64.md

gh issue create \
  --repo wwjei/cpu-docparse \
  --title "[验证] ARM64 架构性能验证" \
  --body-file docs/agent/issue_arm64.md

gh issue create \
  --repo wwjei/cpu-docparse \
  --title "[验证] NVIDIA GPU 加速验证" \
  --body-file docs/agent/issue_nvidia_gpu.md
```

## 5. 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `The token in GH_TOKEN is invalid` | Token 过期/拼写错误/权限不足 | 重新生成 token |
| `Authentication failed` (git push) | Token 无 Contents write 权限 | 检查 token 的 Repository access 和 Permissions |
| `gh: command not found` | 未安装或不在 PATH | 检查安装路径 |
| TLS 连接中断 | 网络不稳定/代理问题 | 重试或配置 `git config http.proxy` |
| `dubious ownership` | 目录 owner 与当前用户不同 | `git config --global --add safe.directory /data` |

## 6. 参考

- 官方文档: https://cli.github.com/manual/
- gh auth login: https://cli.github.com/manual/gh_auth_login
- gh issue create: https://cli.github.com/manual/gh_issue_create
- GitHub CLI 快速入门: https://docs.github.com/zh/github-cli/github-cli/quickstart
