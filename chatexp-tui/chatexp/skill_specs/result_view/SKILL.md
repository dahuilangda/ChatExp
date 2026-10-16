---
name: result_view
layer: display control
description: "Control the terminal table and statistics panel by scope, sorting, batch, page, and best-record views."
when_to_use: "User wants to change the terminal display: filtering, sorting, pagination, or viewing statistics panels."
title_zh: "终端结果展示与视图控制"
title_en: "terminal result display and view control"
actions:
  - view_experiments
  - analyze_data
  - status
---

# Terminal Result Display and View Control

## Purpose

**zh:** 控制左侧终端表格或统计面板，切换范围、排序、批次、分页和最佳视图。
**en:** Control the left terminal table/statistics panel by scope, sorting, batch, page, and best-record views.

## Triggers

**zh:** 显示, 只看, 左侧, 最佳, 全部, 下一页, 上一页, 按评分, 统计面板, 查看, 翻页
**en:** show, only, left panel, best, all, next page, previous page, sort by score, statistics panel, view, page

## Input contract

scope? in recent|completed|pending|best|all; sort? in recent|score_desc|trial_asc; batch_id?; page?; page_delta?; clear_batch?

## Output contract

view_experiments JSON containing only changed display fields: scope, sort, batch_id, page, page_delta, clear_batch.

## Quality checks

- Display-only actions must not change workbook data
- Best/completed/pending filters preserve current project context
- Pagination requests change page only

## Decision rule

**zh:** 用户只是改变终端展示方式时使用；不涉及数据分析或实验记录修改。
**en:** Use when the user only changes terminal presentation; does not involve data analysis or record mutation.

## Failure policy

If the display intent is ambiguous, ask whether the user wants filtering, sorting, or statistical analysis. Retry LLM up to 3 times for structured output.
