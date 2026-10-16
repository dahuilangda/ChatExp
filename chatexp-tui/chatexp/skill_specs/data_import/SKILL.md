---
name: data_import
layer: data recognition
description: "Recognize experiment records, factor columns, response columns, units, header hierarchy, and grouped layouts from Excel or pasted text using the LLM."
when_to_use: "User provides external data (Excel, pasted text, copied tables) containing experimental records that need recognition and structuring."
title_zh: "LLM 实验数据识别与导入"
title_en: "LLM experimental data recognition and import"
actions:
  - import_experiment_sheet
  - append_results
---

# LLM Experimental Data Recognition and Import

## Purpose

**zh:** 用大语言模型识别 Excel 或粘贴文本中的实验记录、因素列、目标列、单位、表头层级和分组结构。
**en:** Use the LLM to recognize experiment records, factor columns, response columns, units, header hierarchy, and grouped layouts from Excel or pasted text.

## Triggers

**zh:** 导入, Excel, 表头, 历史数据, 初始数据, 粘贴, 复制, 数据表, 结果列, 整理, 实验记录, 数据, 表格, 帮我把
**en:** import, Excel, header, historical data, initial data, paste, copied, data sheet, response column, organize, record, data, table, help me

## Input contract

source_excel? OR rows[]; optional workbook_name; factors[]; objectives[]; objective_names[]; manual_column_map?

## Output contract

append_results/import_experiment_sheet JSON; pasted initial data must include rows[], factors[], objectives[], optional next_design_count.

## Quality checks

- One row per experiment/run/well/sample
- Factors and objectives are explicit LLM outputs
- Structurally invalid JSON triggers LLM retry up to 3 times with error feedback
- Preserve units and IDs when visible
- Factor types inferred from experiment semantics, not from domain keyword lists

## Decision rule

**zh:** 只要用户给出外部表格或长串粘贴数据，就先将数据语义整理成 rows/factors/objectives；不要退回规则识别。
**en:** Whenever the user provides an external sheet or pasted runs, first normalize the data semantics into rows/factors/objectives; do not fall back to deterministic parsing.

## Failure policy

Retry the LLM up to 3 times; if records/factors/objectives remain missing, surface the exact missing contract fields and the last LLM error.
