---
name: data_analysis
layer: analysis and diagnosis
description: "Summarize objective coverage, factor exploration, missingness, current best, trends, data quality, and priorities for continuing experiments."
when_to_use: "User asks about data quality, trends, statistics, current best results, or recommendations for next steps."
title_zh: "数据诊断、趋势解释与下一步建议"
title_en: "data diagnosis, trend interpretation, and next-step advice"
actions:
  - analyze_data
  - status
  - refresh_scores
  - inspect_workbook
  - export
---

# Data Diagnosis, Trend Interpretation, and Next-Step Advice

## Purpose

**zh:** 汇总目标覆盖、因素探索、缺失值、当前最优、趋势、数据质量和继续实验的优先级。
**en:** Summarize objective coverage, factor exploration, missingness, current best, trends, data quality, and priorities for continuing experiments.

## Triggers

**zh:** 分析, 统计, 诊断, 趋势, 当前最好, 数据质量, 下一步, 建议, 进展, 质量, 覆盖, 缺失, 异常, 异常值
**en:** analyze, statistics, diagnose, trend, current best, data quality, next step, recommendation, progress, quality, coverage, missing, anomaly, outlier

## Input contract

scope? in project|data_quality|optimization; include_recommendations?; include_quality?

## Output contract

analyze_data/status/refresh_scores/inspect_workbook/export JSON, followed by LLM narrative summary when analysis is requested.

## Quality checks

- Separate completed and pending rows
- Mention objective coverage and factor coverage
- Call out insufficient BO data explicitly
- Do not mutate records unless refresh_scores is requested
- The LLM must produce the narrative from actual data, not from templates

## Decision rule

**zh:** 用户问数据说明、质量、趋势或下一步时使用；不改变实验记录，除非用户明确要求刷新评分。
**en:** Use when the user asks for data explanation, quality, trends, or next steps; do not mutate records unless score refresh is explicitly requested.

## Failure policy

Retry the LLM up to 3 times; if the analysis narrator still fails, surface the exact LLM error and missing contract fields.
