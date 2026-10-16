---
name: conversation
layer: conversational interface
description: "Handle greetings, capability questions, thanks, and lightweight meta conversation without changing experiment state."
when_to_use: "User greets ChatExp, asks what it can do, thanks it, or makes a short conversational turn that does not request an experiment operation."
title_zh: "自然对话与能力说明"
title_en: "natural conversation and capability explanation"
actions:
  - respond
  - clarify
  - none
---

# Natural Conversation And Capability Explanation

## Purpose

**zh:** 用自然语言回应寒暄、能力询问和感谢，同时保持工艺优化助手身份，不修改项目或实验数据。
**en:** Respond naturally to greetings, capability questions, and thanks while keeping the process optimization assistant role and without mutating project or experiment data.

## Triggers

**zh:** 你好, 您好, 早上好, 晚上好, 你是谁, 你能做什么, 能力, 帮助我, 谢谢, 多谢, 辛苦了
**en:** hello, hi, hey, good morning, good evening, who are you, what can you do, capabilities, thanks, thank you

## Input contract

message, context.history?, context.project?, context.pending?

## Output contract

respond JSON with a short, useful message; clarify only when the user asks an ambiguous meta question that needs context.

## Quality checks

- Keep the reply brief and natural
- Explain concrete ChatExp capabilities when asked
- Do not create, import, analyze, or modify experiment data
- If the user includes a real experiment request in the same message, route to the experiment-specific skill instead

## Decision rule

**zh:** 用户只是寒暄、问身份/能力或表达感谢时使用 respond；如果同一句里包含实验目标、数据、导入、分析或设计请求，则交给对应实验 skill。
**en:** Use respond when the user is only greeting, asking identity/capabilities, or thanking; if the same message includes an experiment goal, data, import, analysis, or design request, route to the corresponding experiment skill.

## Failure policy

Retry the LLM planner up to 3 times for structured output. If the planner cannot return valid JSON, surface the planner contract error and trace; do not answer through rule-based fallback.
