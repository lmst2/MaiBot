# A_Memorix 子模块维护说明（维护者内部文档）

> 本文档用于维护者，不面向普通用户。

## 1. 基本约束
- 子模块路径固定：`plugins/A_memorix`
- 子模块仓库固定：`https://github.com/A-Dawn/A_memorix.git`
- 子模块分支固定：`MaiBot_branch`
- 强约束：主仓内 `plugins/A_memorix` 指针必须等于远端 `origin/MaiBot_branch` 最新 HEAD

## 2. 首次拉取/恢复子模块
```bash
git submodule update --init --recursive
```

若目录为空或缺少 `_manifest.json`，先执行上面的命令再排查其他问题。

## 3. 维护者更新流程
1. 先在外部仓 `MaiBot_branch` 完成目标功能合入。
2. 在主仓执行：
```bash
git submodule update --remote --recursive plugins/A_memorix
git add plugins/A_memorix .gitmodules
git commit -m "chore(submodule): bump A_memorix"
```

## 4. CI 严格校验说明
- PR Precheck 会校验：
  - `.gitmodules` 的 path/url/branch 必须匹配固定值
  - 子模块指针必须等于远端 `MaiBot_branch` 最新 HEAD
- Docker 构建工作流在构建前也会执行同样的 fail-fast 对齐检查

## 5. 回滚策略
- 回滚主仓提交会同时回滚子模块指针。
- 但若回滚后的指针不再是远端 `MaiBot_branch` 最新 HEAD，CI 会阻断。
- 处理方式：
  - 先在外部仓移动/回滚 `MaiBot_branch` 到目标提交，再重跑；
  - 或按团队流程申请一次性 CI 豁免。
