# GitHub使用说明

将`enose-drift-toolkit`目录解压到本地，从该目录执行完整性核验。仓库中不需要保存原始Dataset.zip、虚拟环境、缓存字节码或临时输出，`.gitignore`已配置对应规则。

安装GitHub CLI并登录后，可使用以下命令创建远端仓库；`--public`会让仓库公开，执行前可先检查本地文件与许可。这里提供命令，不自动执行上传。

```text
git init -b main
git add .
git commit -m "Initial standalone numerical release"
gh repo create enose-drift-toolkit --public --source=. --remote=origin --push
```

如果已有目标仓库，请使用其真实URL设置remote，不要直接重建同名仓库。GitHub普通Git单文件硬上限为100 MiB；原始归档应使用数据源下载入口，或另行配置LFS/Release附件。

可在GitHub上创建版本化Release，再通过Zenodo保存不可变版本并获得软件存档DOI。未创建的仓库地址、Release和DOI没有写成已存在资源。

参考：
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
