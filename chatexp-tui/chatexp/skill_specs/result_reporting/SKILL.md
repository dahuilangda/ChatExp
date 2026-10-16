---
name: result_reporting
layer: state mutation
description: "Merge completed results, numbered experiments, modified conditions, skipped plans, and pasted initial data into canonical experiment records."
when_to_use: "User provides measurement results for numbered experiments, modifies conditions, or reports completed runs."
title_zh: "实验结果回填与计划对齐"
title_en: "result reporting and plan alignment"
actions:
  - append_results
  - update_plan_experiment
  - skip_pending_designs
---

# Result Reporting and Plan Alignment

## Purpose

**zh:** 把完成结果、编号实验、修改条件、跳过计划和粘贴初始数据合并为标准实验记录。
**en:** Merge completed results, numbered experiments, modified conditions, skipped plans, and pasted initial data into canonical experiment records.

## Triggers

**zh:** 做完, 结果, 回填, 编号, 实验, 跳过, 删除, 改成, 实际条件, 初始数据, 测完了, 完成了, 汇报
**en:** done, result, report, ID, experiment, skip, remove, change to, actual condition, initial data, finished, completed, reporting

## Input contract

rows[{candidate_id?,Trial?,experiment_id?,factor_values?,objective_values?,status?,COMMENT?}], candidate_ids?, batch_id?, updates?

## Output contract

append_results/update_plan_experiment/skip_pending_designs JSON with candidate_id or experiment_id whenever user references numbered plans.

## Quality checks

- Each reported experiment maps to one row
- Multi-objective measurements are merged into the same row
- Modified conditions are retained as reported, not overwritten by plans
- The LLM must extract candidate_id from formats like 'Experiment 6:', '6号', '#6', 'No.6'

## Decision rule

**zh:** 用户给了测量值或实验编号时使用；如果没有 project 但 rows/factors/objectives 齐全，则创建项目并导入初始记录。
**en:** Use when the user gives measurements or experiment IDs; if no project exists but rows/factors/objectives are complete, create the project and import initial records.

## Failure policy

Retry the LLM up to 3 times; if a numbered result still lacks a candidate_id, surface the issue and ask the user for the experiment number. Never assign candidate_id by deterministic rules.
