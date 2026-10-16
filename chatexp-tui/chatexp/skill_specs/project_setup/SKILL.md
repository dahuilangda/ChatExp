---
name: project_setup
layer: campaign framing
description: "Peel an informal experimental goal into domain, factors, ranges, categories, constraints, objectives, and weights."
when_to_use: "User describes an experiment goal but no project exists yet, or wants to modify factors/objectives of an existing project."
title_zh: "渐进式实验项目定义"
title_en: "progressive experiment project setup"
actions:
  - draft_setup
  - create_project
  - load_project
  - define_factors
  - define_objective
  - update_factor
  - update_objective
---

# Progressive Experiment Project Setup

## Purpose

**zh:** 把非结构化实验目标剥离成领域、因素、范围、类别、约束、目标和权重。
**en:** Peel an informal experimental goal into domain, factors, ranges, categories, constraints, objectives, and weights.

## Triggers

**zh:** 新建, 起步方案, 实验设计, 优化, 因素, 目标, 范围, 约束, 权重, 项目, 工作簿, 配比, 工艺, 配方, 参数
**en:** new project, starter plan, experimental design, optimize, factor, objective, range, constraint, weight, project, workbook, ratio, process, formulation, parameter

## Input contract

workbook_name?, domain?, factors[{name,kind,unit,min_value,max_value,step,categories}], objectives[{name,goal,target_value,weight,unit}], constraints?

## Output contract

draft_setup/create_project/load_project JSON with workbook_name when applicable, complete factors[], objectives[] for project definition; no prose-only plan.

## Quality checks

- Every factor has kind and range/categories
- Every objective has goal/weight/unit if available
- Do not create a project until the user confirms the draft
- Factor types should reflect experiment semantics: temperature/time/pH usually ordinal with steps; catalyst/solvent categorical; concentration/ratio continuous

## Decision rule

**zh:** 用户描述实验目的但尚未形成项目时使用；若用户同时粘贴了多条历史数据，则优先交给 data_import/result_reporting。
**en:** Use when the user describes an experiment goal but no project exists yet; if the user pasted multiple historical records, route to data_import/result_reporting first.

## Failure policy

If critical objective or controllable factors are missing, ask up to three concrete clarify questions. The LLM retries up to 3 times for structured output; after that the missing fields are surfaced to the user.
