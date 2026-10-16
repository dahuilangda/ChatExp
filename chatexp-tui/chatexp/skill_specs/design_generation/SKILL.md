---
name: design_generation
layer: design generation
description: "Generate initial designs or next-batch BO suggestions from current project state, historical results, and scoring functions."
when_to_use: "Project has factors and objectives defined, and user wants to generate initial or next-batch experiment designs."
title_zh: "初始化与下一批实验设计"
title_en: "initial and next-batch experiment design"
actions:
  - initialize_design
  - generate_next_design
---

# Initial and Next-Batch Experiment Design

## Purpose

**zh:** 基于当前项目状态、历史结果和评分函数生成初始化设计或下一批 BO 建议。
**en:** Generate initial designs or next-batch BO suggestions from current project state, historical results, and scoring functions.

## Triggers

**zh:** 初始化设计, 生成, 下一批, 推荐, 组合, 实验点, BO, 再来, 再来一批, 追加
**en:** initialize design, generate, next batch, recommend, combination, runs, BO, more, another batch, append

## Input contract

count?, mode? in initial|next; score_profile_confirmation?; constraints?

## Output contract

initialize_design/generate_next_design JSON with count, mode, optional constraints and score_profile_confirmation.

## Quality checks

- Project must already have factors and objectives
- If pasted data is present, import it before proposing new runs
- Confirm batch generation before writing designs
- Recommended batch size should be reasonable for the domain

## Decision rule

**zh:** 只有项目已有 factors/objectives 后使用；若用户同时贴了初始数据，先导入 rows，再准备下一批设计确认。
**en:** Use only after factors/objectives exist; if the user pasted initial data, import rows first and then prepare next-batch confirmation.

## Failure policy

If count or constraints are unclear, ask a targeted clarification; do not generate an unconfirmed batch. Retry LLM up to 3 times for structured output.
