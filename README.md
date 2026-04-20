本仓库提供一份可直接在 Overleaf 编译的恒星大气物理复习笔记。

## 文件
- `main.tex`：主文档（每个复习点一个 `\section{}`，含关键推导与物理解释）

## Overleaf 编译方式
1. 在 Overleaf 新建项目并上传 `main.tex`
2. 将编译器设置为 **XeLaTeX**（或 LuaLaTeX 也可）
3. 直接 Recompile 即可生成 PDF

## 本地编译（可选）
如果你本地装了 TeX Live：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

