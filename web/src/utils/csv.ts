export type CsvOverview = {
  headers: string[];
  rowCount: number;
  missingCells: number;
  numericColumns: string[];
};

/**
 * 解析一行 CSV 字符串为字段数组。
 * @param line 单行 CSV 文本。
 * @returns 字段数组。
 */
function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
        continue;
      }
      inQuotes = !inQuotes;
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

/**
 * 解析 CSV 文件并输出概览信息。
 * @param file CSV 文件对象。
 * @returns CSV 概览信息。
 */
export async function parseCsvOverview(file: File): Promise<CsvOverview> {
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) {
    throw new Error("CSV 内容不足（至少包含表头与一行数据）");
  }

  const headers = splitCsvLine(lines[0]);
  if (headers.length < 2) {
    throw new Error("CSV 表头列数过少");
  }

  let missingCells = 0;
  const sampleLimit = Math.min(lines.length, 800);
  const numericHits = new Array(headers.length).fill(0);
  const totalHits = new Array(headers.length).fill(0);

  for (let i = 1; i < sampleLimit; i++) {
    const cells = splitCsvLine(lines[i]);
    for (let c = 0; c < headers.length; c++) {
      const v = cells[c] ?? "";
      if (v === "") {
        missingCells++;
        continue;
      }
      totalHits[c] += 1;
      const n = Number(v);
      if (!Number.isNaN(n) && Number.isFinite(n)) {
        numericHits[c] += 1;
      }
    }
  }

  const numericColumns = headers.filter((_, idx) => {
    if (totalHits[idx] === 0) return false;
    return numericHits[idx] / totalHits[idx] >= 0.9;
  });

  return {
    headers,
    rowCount: lines.length - 1,
    missingCells,
    numericColumns,
  };
}
