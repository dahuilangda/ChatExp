from __future__ import annotations

from datetime import date, datetime, time
import re
from pathlib import Path
from typing import Any

import pandas as pd


class ExcelDataHandler:
    def normalize_header(self, value: object) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[\s\r\n\t]+", "", text)
        text = re.sub(r"[\(\[（【].*?[\)\]）】]", "", text)
        text = re.sub(r"[%/%℃°_\\-]+", "", text)
        return text

    def read_raw_sheet(self, source_excel: str | Path, *, sheet_name: str | int = 0) -> pd.DataFrame:
        return pd.read_excel(source_excel, sheet_name=sheet_name, header=None)

    def build_llm_sheet_payload(
        self,
        raw: pd.DataFrame,
        *,
        known_factor_names: list[str] | None = None,
        known_objective_names: list[str] | None = None,
        objective_names: list[str] | None = None,
        max_rows: int = 32,
        max_cols: int = 12,
    ) -> dict[str, Any]:
        selected_row_indices = self._select_row_indices_for_llm(raw, max_rows=max_rows)
        visible_rows = len(selected_row_indices)
        visible_cols = min(raw.shape[1], max_cols)
        window = raw.iloc[selected_row_indices, :visible_cols].copy()
        rows: list[dict[str, Any]] = []
        for row_offset, source_row_index in enumerate(selected_row_indices):
            values = [self._serialize_cell(window.iat[row_offset, col_index]) for col_index in range(visible_cols)]
            while values and values[-1] == "":
                values.pop()
            if any(value != "" for value in values):
                rows.append({"row_index": source_row_index + 1, "values": values})
        return {
            "sheet_shape": {"rows": int(raw.shape[0]), "columns": int(raw.shape[1])},
            "visible_window": {"rows": visible_rows, "columns": visible_cols},
            "llm_task": (
                "Infer schema from this sampled window only. Do not emit all workbook rows. "
                "Python will apply header_row, column_mapping, factors, and objectives to the full sheet."
            ),
            "known_factor_names": list(known_factor_names or []),
            "known_objective_names": list(known_objective_names or []),
            "objective_hints": list(objective_names or []),
            "rows": rows,
        }

    def _select_row_indices_for_llm(self, raw: pd.DataFrame, *, max_rows: int) -> list[int]:
        if raw.empty:
            return []
        total_rows = len(raw)
        if total_rows <= max_rows:
            return list(range(total_rows))
        row_scores: list[float] = []
        for row_index in range(total_rows):
            row = raw.iloc[row_index]
            non_empty = int(row.notna().sum())
            numeric = int(pd.to_numeric(row, errors="coerce").notna().sum())
            row_scores.append(non_empty + numeric * 0.5)
        best_start = 0
        best_score = float("-inf")
        for start in range(0, total_rows - max_rows + 1):
            score = sum(row_scores[start : start + max_rows])
            if score > best_score:
                best_score = score
                best_start = start
        context_start = max(best_start - 2, 0)
        context_end = min(context_start + max_rows, total_rows)
        if context_end - context_start < max_rows:
            context_start = max(context_end - max_rows, 0)
        return list(range(context_start, context_end))

    def _serialize_cell(self, value: object) -> Any:
        if pd.isna(value):
            return ""
        if isinstance(value, pd.Timestamp):
            return value.isoformat(sep=" ", timespec="seconds")
        if isinstance(value, datetime):
            return value.isoformat(sep=" ", timespec="seconds")
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, time):
            return value.isoformat(timespec="seconds")
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned = cleaned.where(pd.notnull(cleaned), None)
        cleaned.columns = [str(col).strip() for col in cleaned.columns]
        for col in cleaned.columns:
            if cleaned[col].dtype == object:
                cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        return cleaned

    def _coalesce_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for column in dict.fromkeys(df.columns):
            matching = df.loc[:, [name for name in df.columns if name == column]]
            if matching.shape[1] == 1:
                result[column] = matching.iloc[:, 0]
                continue
            result[column] = matching.bfill(axis=1).iloc[:, 0]
        return result

    def _header_cell_text(self, value: object, index: int) -> str:
        text = str(value or "").strip()
        return text if text and text.lower() != "nan" else f"column_{index}"
