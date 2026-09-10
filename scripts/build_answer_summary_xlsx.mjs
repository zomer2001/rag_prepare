#!/usr/bin/env node

import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [summaryPath, outputPath, previewPath] = process.argv.slice(2);
if (!summaryPath || !outputPath || !previewPath) {
  throw new Error("Usage: build_answer_summary_xlsx.mjs SUMMARY_JSON OUTPUT_XLSX PREVIEW_PNG");
}

const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));
const labels = {
  hybrid__raw: ["Hybrid", "Raw evidence"],
  evolink__raw: ["EvoLink", "Raw evidence"],
  d1_norerank__raw: ["EvoNeed-D1", "Raw evidence"],
  d1_norerank__graph: ["EvoNeed-D1", "Entity/relation/event tables"],
  d1_norerank__evolution: ["EvoNeed-D1", "Evolution structure"],
};
const rawEm = Number(summary.d1_norerank__raw?.exact_match ?? 0);
const typeEm = (values, key) => Number(values.by_question_type?.[key]?.exact_match ?? 0);
const rows = Object.entries(summary)
  .map(([runId, values]) => {
    const [method, context] = labels[runId] ?? [runId, "unknown"];
    return [
      method,
      context,
      Number(values.query_count ?? 0),
      Number(values.exact_match ?? 0),
      Number(values.token_f1 ?? 0),
      Number(values.exact_match ?? 0) - rawEm,
      typeEm(values, "comparison_query"),
      typeEm(values, "inference_query"),
      typeEm(values, "temporal_query"),
      Number(values.mean_citation_id_validity ?? 0),
      Number(values.mean_context_tokens ?? 0),
      Number(values.error_count ?? 0),
    ];
  })
  .sort((left, right) => right[3] - left[3] || String(left[0]).localeCompare(String(right[0])));

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Summary");
sheet.showGridLines = false;

sheet.getRange("A2").values = [["Gold500 answer evaluation pilot"]];
sheet.getRange("A3").values = [[
  "First 100 questions; Top-20 evidence; DeepSeek deepseek-v4-flash; temperature 0; thinking disabled; 16,000-token context limit",
]];
sheet.getRange("A5:L5").values = [[
  "Method",
  "Answer context",
  "N",
  "Exact Match",
  "Token F1",
  "Delta EM vs D1 raw",
  "Comparison EM",
  "Inference EM",
  "Temporal EM",
  "Citation ID validity",
  "Mean context tokens",
  "Errors",
]];
sheet.getRange(`A6:L${5 + rows.length}`).values = rows;

sheet.getRange("A1:L15").format.font = { name: "Arial", size: 10, color: "#1F2937" };
sheet.getRange("A2").format.font = { name: "Arial", size: 14, bold: true, color: "#172554" };
sheet.getRange("A3").format.font = { name: "Arial", size: 10, italic: true, color: "#475569" };
sheet.getRange("A5:L5").format = {
  fill: "#1E3A5F",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
};
sheet.getRange(`A6:L${5 + rows.length}`).format.verticalAlignment = "center";
sheet.getRange(`D6:J${5 + rows.length}`).format.numberFormat = "0.0%";
sheet.getRange(`C6:C${5 + rows.length}`).format.numberFormat = "0";
sheet.getRange(`K6:K${5 + rows.length}`).format.numberFormat = "#,##0";
sheet.getRange(`L6:L${5 + rows.length}`).format.numberFormat = "0";
sheet.getRange(`A6:L${5 + rows.length}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#E2E8F0" },
  bottom: { style: "thin", color: "#94A3B8" },
};

sheet.getRange("A13").values = [[
  "Current result: Hybrid raw and EvoNeed-D1 with generic graph tables tie at 82.0% Exact Match.",
]];
sheet.getRange("A14").values = [[
  "Latency omitted: the current batch included concurrency queue time, so it is not valid per-question latency.",
]];
sheet.getRange("A13:A14").format.font = { name: "Arial", size: 10, color: "#334155" };

sheet.getRange("A:A").format.columnWidth = 16;
sheet.getRange("B:B").format.columnWidth = 29;
sheet.getRange("C:C").format.columnWidth = 8;
sheet.getRange("D:J").format.columnWidth = 14;
sheet.getRange("K:K").format.columnWidth = 20;
sheet.getRange("L:L").format.columnWidth = 8;
sheet.getRange("2:2").format.rowHeight = 24;
sheet.getRange("3:3").format.rowHeight = 18;
sheet.getRange("5:5").format.rowHeight = 32;

workbook.recalculate();
const inspection = await workbook.inspect({
  kind: "table",
  range: `Summary!A2:L${5 + rows.length}`,
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 12,
});
console.log(inspection.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Summary",
  range: "A1:L15",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
