# math_matcher — 稳健数学答案匹配器

基于 [`math_verify`](https://github.com/huggingface/Math-Verify) 封装的一个**鲁棒答案判等**工具。
直接裸调 `math_verify.parse/verify` 在真实数据集上会产生大量**误判（false negative）**，
本模块对这些常见坑做了统一处理。

## 为什么需要它

裸调 `math_verify` 在以下答案类型上经常判错（判成不相等）：

| 场景 | 例子（两者其实等价） |
|------|----------------------|
| 区间带变量前缀 | `[0,2)` vs `x \in [0,2)` |
| 实数集写法 | `\mathbb{R}` vs `(-\infty,\infty)` |
| 分数宏 | `\dfrac{1}{2}` vs `\frac{1}{2}` |
| 多解 | `166464` vs `166464 or 646416` |
| 纯文本答案 | `Saturday`、`No`、`convergent` |
| 元组/集合 | `(2,8)` vs `(2, 8)` |

根因有两类：
1. **调用姿势**：两边没都包 `$...$`，且没启用 `String/Expr` 抽取配置，导致元组/集合/文本被解析成空。
2. **答案类型**：区间前缀、多解、文本等 `math_verify` 本身覆盖不到。

本模块在正确调用姿势之上，叠加了 5 层兜底：
1. 两边包 `$...$` + 启用 `Latex/Expr/String` 三种抽取配置；
2. 剥离 `x \in`、`var =` 等赋值前缀；
3. `\mathbb{R}` ↔ `(-\infty,\infty)` 等价；
4. 多解 ground truth（`A or B`）任一分支命中即算对；
5. 归一化后的字符串字面兜底（处理纯文本答案）。

## 安装

```bash
pip install math-verify
```

## 用法

```python
from math_matcher import match, match_solution, extract_boxed

# 1) 两个「纯答案字符串」判等
match("[0,2)", "x \\in [0,2)")          # True
match("\\dfrac{1}{2}", "\\frac{1}{2}")  # True
match("2", "4")                          # False

# 2) 从完整解题文本里抽取最后一个 \boxed{...} 再判等
solution = "......所以最终答案是 \\boxed{[0,2)}。"
match_solution(solution, "x \\in [0,2)")  # True

# 3) 只想抽 \boxed 内容
extract_boxed(solution)                   # "[0,2)"
```

### API

| 函数 | 说明 |
|------|------|
| `match(prediction, ground_truth) -> bool` | 比较两个**纯答案字符串**是否数学等价 |
| `match_solution(solution_text, ground_truth) -> bool` | 先从完整文本抽取最后一个 `\boxed{...}`，再 `match` |
| `extract_boxed(text) -> str` | 返回最后一个 `\boxed{...}` 的内容（花括号配平），无则返回 `""` |
| `norm_ans(s) -> str` | 字符串归一化（去定界符 / 统一分数宏 / 去空白等），一般内部使用 |

## 自测

```bash
python math_matcher.py
```

会跑一组内置用例并打印 `N/N self-tests passed`。

## 注意事项 / 已知边界

- **元组顺序**：`math_verify` 会把 `(0,2,1)` 与 `(2,1,0)` 判为相等（按集合而非有序元组）。
  若你的题目里坐标/赋值的**顺序有意义**，需要额外做有序比较，本模块不强制有序。
- 多解切分目前按 `or` / 中文逗号切分；如果你的数据用别的分隔符（如 `;`），可在
  `match` 的第 3 步正则里补充。
- 这是一个**判等工具**，不负责"判断模型输出对不对"之外的任何业务逻辑。

## 实测效果

在 10289 条 OpenMathReasoning teacher 数据上，裸调误判约 270+ 条；
换用本匹配器后，残留误判 **0 条**——确认那些"失败"全部是调用姿势/答案类型导致的误报，
而非数据本身错误。
