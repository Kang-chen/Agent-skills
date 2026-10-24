# Plain language

The reader is a competent colleague who has never heard of this project. They
should not have to re-read a sentence, look up an internal code, or open the
code base to follow the report.

## The recipe

1. **Section opens with the question, closes with the answer.** One sentence
   each. Everything between them is evidence.
2. **First use defines the term.** Every abbreviation, arm code, dataset name,
   pipeline stage, and custom metric gets, at first appearance in the body, its
   full name and one clause saying what it is for. The glossary table repeats
   the definitions; it does not replace them.
3. **Bare internal codes never stand alone.** Write "冻结表示加图像特征的融合臂
   （B5）", not "B5". The code is an index into the definition, never a
   substitute for it.
4. **Numbers carry their meaning.** metric name, value, uncertainty, unit of
   analysis, sample size, and what it is being compared with.
5. **Numbers carry their provenance.** A number recomputed from a run artifact
   and a number copied from a planning document are different kinds of claim.
   When two sources disagree about the same count, say so, name both, and say
   which one the report uses.
6. **One fact per sentence.** If a sentence needs a second reading, split it.
7. **Mechanism before name.** Say what the thing does in ordinary words, then
   give its technical name: "把每个患者的图像切块编码成一个向量，再取平均得到患者
   级表示（frozen encoder + mean pooling）".
8. **Say the direction.** Higher, lower, unchanged, and by how much, compared
   with what.

## Rewrites

| Hard to read | Readable |
|---|---|
| B5−B1 配对增量 +0.020，CI 跨 0。 | 把图像特征加进来以后（B5 臂），患者级 AUC 比只用细胞类型的原版 CellPA（B1 臂）高 0.020；这个差值的 95% 区间是 −0.091 到 +0.126，跨过 0，所以不能认为图像特征带来了增益。 |
| 冻结 VirTues-sp32 编码 256 crop/患者。 | 每位患者从多重免疫荧光图上取 256 个小窗口，用一个不再训练的图像编码器（VirTues-sp32）把每个窗口变成一个向量；这些向量平均成该患者的图像表示。 |
| 读出层用 8 维 PCA + L2 LR。 | 读出层先把表示压到 8 个主成分，再用带 L2 惩罚的逻辑回归给出预测概率；压维和回归都只在训练折上拟合。 |
| 结果不显著。 | 在这 52 位患者上，差值的区间跨过 0；本设计能稳定检出的最小差值约为 0.22 AUC，比观察到的差值大一个量级，所以这个实验既没有证明增益，也没有排除小幅增益。 |

## Register

- No marketing words. No "大幅提升", "显著优于", "完美复现" without the number
  that earns them.
- Hedging that carries information stays: "在这个队列上", "未验证", "探索性".
- Write in the report language for prose and in the native form for paths,
  identifiers, code names, and Figure text.
- A sentence whose subject is an internal file name is usually a sentence about
  a scientific object; rewrite it around the object.

## The zero-background pass

Before finishing, read the report from the top as if you had just joined the
team. Mark every place where you needed knowledge that is not in the document.
Each mark is either a missing definition, a missing Figure, or a missing
sentence of motivation. Fix all three kinds, then re-read.
