# git 操作备忘

## 当前分支
- dev_csj_0511

## 创建分支(例)
```bash
git checkout -b dev_csj_0511
git push -u origin dev_csj_0511
```

## 日常提交
```bash
git status
git diff
git add -A
git commit -m "msg"
git push
```

## 同步主分支
```bash
git checkout main
git pull origin main
git checkout dev_csj_0511
git merge main
git push
```

或用 rebase：
```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

## 分支管理
```bash
git branch              # 本地
git branch -a           # 含远端
git checkout main       # 切回主
git branch -d  dev_csj_0511   # 已合并删
git branch -D  dev_csj_0511   # 强制删
git push origin --delete dev_csj_0511
```

## 合并回主
```bash
git checkout main
git pull origin main
git merge --no-ff dev_csj_0511
git push origin main
```

## 常用查看
```bash
git log --oneline -20
git log --oneline --graph --all
git diff HEAD~1
git stash         # 暂存改动
git stash pop     # 取回
```
