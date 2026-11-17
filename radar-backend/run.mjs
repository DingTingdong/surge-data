import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { appendFile, mkdtemp, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const SITE_ORIGIN = "https://surge-radar-iwjk7.w29623013.chatgpt.site";
export const ENGINE_VERSION = "surge-1.1.0";
const MARKETS = ["TWSE", "TPEX"];
const SHA = /^[a-f0-9]{64}$/;
const fail = code => { throw new Error(code); };
const ensure = (condition, code) => { if (!condition) fail(code); };
const timestamp = value => typeof value === "string" ? Date.parse(value) : NaN;
const isoDay = value => typeof value === "string" && /^\d{4}-\d\d-\d\d$/.test(value)
  && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
export const safeCode = error => error instanceof Error && /^[A-Z][A-Z0-9_]{2,79}$/.test(error.message)
  ? error.message : "BACKEND_UPDATE_FAILED";

const ARTIFACTS = {
  writer: { manifest: "/_ops/manifest.json", path: "/_ops/collector.mjs" },
  reader: { manifest: "/_ops/github-source-manifest.json", path: "/_ops/github-source-reader.mjs" },
};

export function validateManifest(manifest, kind = "writer") {
  ensure(manifest?.format === "node-esm" && manifest.version === 1 && manifest.nodeMajor === 24
    && ARTIFACTS[kind] && manifest.path === ARTIFACTS[kind].path && SHA.test(manifest.sha256)
    && Number.isInteger(manifest.bytes) && manifest.bytes >= 1000 && manifest.bytes <= 4 * 1024 * 1024,
  "COLLECTOR_MANIFEST_INVALID");
  return manifest;
}

async function request(token, route, { fetcher = fetch, limit = 12 * 1024 * 1024 } = {}) {
  ensure(typeof route === "string" && route.startsWith("/") && !route.startsWith("//")
    && !route.includes("\\") && !route.includes("#"), "SITE_ROUTE_INVALID");
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetcher(SITE_ORIGIN + route, {
        redirect: "manual", signal: AbortSignal.timeout(90_000),
        headers: { "OAI-Sites-Authorization": "Bearer " + token, Accept: "application/json" },
      });
      const transient = response.status === 429 || response.status >= 500;
      if (transient && attempt < 2) {
        await response.body?.cancel();
        await new Promise(resolve => setTimeout(resolve, 2000 * (attempt + 1)));
        continue;
      }
      ensure(response.status === 200, response.status === 401 || response.status === 403
        ? "SITE_AUTHORIZATION_FAILED" : "SITE_REQUEST_FAILED");
      const declared = Number(response.headers.get("content-length"));
      ensure(!Number.isFinite(declared) || declared <= limit, "SITE_RESPONSE_TOO_LARGE");
      const reader = response.body?.getReader();
      ensure(reader, "SITE_RESPONSE_EMPTY");
      const chunks = [];
      let size = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > limit) { await reader.cancel(); fail("SITE_RESPONSE_TOO_LARGE"); }
        chunks.push(Buffer.from(value));
      }
      return Buffer.concat(chunks);
    } catch (error) {
      if (safeCode(error) !== "BACKEND_UPDATE_FAILED" || attempt === 2) throw error;
      await new Promise(resolve => setTimeout(resolve, 2000 * (attempt + 1)));
    }
  }
  fail("SITE_REQUEST_FAILED");
}

async function jsonRequest(token, route, options) {
  const bytes = await request(token, route, options);
  try { return JSON.parse(bytes.toString("utf8")); } catch { fail("SITE_JSON_INVALID"); }
}

export function verifyUpdate(proof, readback, {
  startedAt, finishedAt, collectorExitCode, requireComplete = false, catchup = false,
}) {
  const started = timestamp(startedAt), finished = timestamp(finishedAt);
  ensure(Number.isFinite(started) && Number.isFinite(finished) && finished >= started, "RUN_TIME_INVALID");
  ensure(proof?.origin === SITE_ORIGIN && proof.mode === "automatic" && isoDay(proof.targetDate)
    && timestamp(proof.checkedAt) >= started - 5000 && timestamp(proof.checkedAt) <= finished + 5000,
  "COLLECTOR_PROOF_INVALID");
  ensure(typeof proof.ok === "boolean" && [0, 1].includes(collectorExitCode)
    && (proof.ok ? collectorExitCode === 0 : collectorExitCode === 1), "COLLECTOR_EXIT_MISMATCH");
  ensure(proof.priceOk === true && proof.actionReferencesOk === true
    && proof.referenceRecovery?.ok === true && proof.backfill?.ok === true
    && proof.evidence?.ok === true && proof.context?.ok === true && proof.contextConsistent === true
    && proof.regulatoryConsistent === true && proof.verification?.ok === true
    && Array.isArray(proof.verification.pending) && proof.verification.pending.length === 0,
  "CORE_DATA_NOT_VERIFIED");
  const { radar, chips, regulatory } = readback;
  const startDay = new Date(started + 8 * 60 * 60_000).toISOString().slice(0, 10);
  const finishDay = new Date(finished + 8 * 60 * 60_000).toISOString().slice(0, 10);
  ensure(radar?.meta?.version === ENGINE_VERSION && radar.meta.asOf === proof.targetDate,
    "RADAR_ENGINE_OR_DATE_MISMATCH");
  ensure(chips?.targetDate === proof.targetDate && Array.isArray(chips.feeds), "CHIPS_READBACK_INVALID");
  ensure(Array.isArray(proof.providers) && proof.providers.length === 2, "MARKETS_INCOMPLETE");
  const markets = MARKETS.map(market => {
    const rows = proof.providers.filter(row => row.market === market);
    ensure(rows.length === 1, "MARKETS_INCOMPLETE");
    const provider = rows[0], receipt = provider.receipt;
    const persisted = radar.meta.markets?.[market], snapshot = radar.journal?.markets?.[market];
    ensure(provider.ok === true && provider.date === proof.targetDate
      && receipt?.market === market && receipt.date === proof.targetDate
      && receipt.rawVerified === true && receipt.pointerVerified === true && receipt.journalVerified === true
      && Number.isInteger(receipt.rowCount) && receipt.rowCount >= 300 && SHA.test(provider.sha256)
      && receipt.sha256 === provider.sha256 && persisted?.date === proof.targetDate
      && persisted.count === receipt.rowCount && persisted.sha256 === provider.sha256
      && persisted.sourceUrl === provider.sourceUrl
      && timestamp(persisted.fetchedAt) >= timestamp(provider.fetchedAt)
      && snapshot?.date === proof.targetDate && Number.isFinite(timestamp(snapshot.capturedAt))
      && ((snapshot.status === "compared" && isoDay(snapshot.previousDate) && snapshot.previousDate < proof.targetDate)
        || (snapshot.status === "baseline" && snapshot.previousDate === null)),
    "MARKET_OR_SNAPSHOT_READBACK_FAILED");
    return { market, date: persisted.date, rows: persisted.count, fetchedAt: persisted.fetchedAt,
      sha256: persisted.sha256, rawVerified: true, pointerVerified: true, snapshotVerified: true };
  });
  ensure(proof.regulatory?.ok === true && proof.regulatory.readback?.ok === true
    && Array.isArray(proof.regulatory.feeds) && proof.regulatory.feeds.length === 4
    && isoDay(proof.regulatory.assessmentDate) && proof.regulatory.assessmentDate >= startDay
    && proof.regulatory.assessmentDate <= finishDay && proof.regulatory.assessmentDate >= proof.targetDate
    && Array.isArray(regulatory?.feeds) && regulatory.feeds.length === 4,
  "REGULATORY_NOT_VERIFIED");
  const regulatoryFeeds = [];
  for (const market of MARKETS) for (const kind of ["attention", "disposition"]) {
    const matches = proof.regulatory.feeds.filter(row => row.market === market && row.kind === kind);
    const stored = regulatory.feeds.filter(row => row.market === market && row.kind === kind);
    ensure(matches.length === 1 && stored.length === 1, "REGULATORY_NOT_VERIFIED");
    const feed = matches[0], current = stored[0];
    const expectedDate = kind === "attention" ? proof.targetDate : proof.regulatory.assessmentDate;
    ensure(feed.ok === true && feed.rawVerified === true && feed.pointerVerified === true
      && feed.recordsVerified === true && SHA.test(feed.sha256) && feed.targetDate === expectedDate
      && current.status === "ready" && current.referenceDate === feed.targetDate
      && current.sha256 === feed.sha256 && current.rowCount === feed.rowCount
      && timestamp(current.fetchedAt) >= timestamp(feed.fetchedAt)
      && timestamp(feed.fetchedAt) >= started - 5000, "REGULATORY_READBACK_FAILED");
    regulatoryFeeds.push({ market, kind, date: current.referenceDate, rows: current.rowCount,
      fetchedAt: current.fetchedAt, sha256: current.sha256 });
  }
  ensure(proof.chips?.targetDate === proof.targetDate && Array.isArray(proof.chips.feeds)
    && proof.chips.feeds.length === 47 && Array.isArray(proof.chips.history)
    && proof.chips.history.every(row => row.ok === true), "CHIPS_NOT_VERIFIED");
  for (const market of MARKETS) {
    const sessions = chips.sessions?.[market], ownershipDates = chips.ownershipDates?.[market];
    ensure(Array.isArray(sessions) && sessions.length === 20 && sessions[0] === proof.targetDate
      && new Set(sessions).size === 20 && sessions.every(isoDay)
      && Array.isArray(ownershipDates) && ownershipDates.length === 3
      && ownershipDates[0] === proof.targetDate && new Set(ownershipDates).size === 3
      && ownershipDates.every(isoDay), "CHIPS_SESSIONS_INVALID");
    for (const [kind, dates] of [["institutional", sessions], ["foreign-ownership", ownershipDates]]) {
      for (const date of dates) ensure(proof.chips.feeds.filter(row => row.market === market
        && row.kind === kind && row.date === date).length === 1, "CHIPS_FEED_MISSING");
    }
  }
  ensure(isoDay(chips.expectedHolderDate) && proof.chips.feeds.filter(row => row.market === "TDCC"
    && row.kind === "holders" && row.date === chips.expectedHolderDate && row.ok === true).length === 1,
  "HOLDER_WEEK_NOT_VERIFIED");
  const pending = [];
  for (const feed of proof.chips.feeds) {
    if (feed.ok !== true) {
      ensure(MARKETS.includes(feed.market) && feed.kind === "foreign-ownership"
        && feed.date === proof.targetDate && feed.code === "CHIPS_SOURCE_DATE_NOT_READY"
        && feed.failureRecorded === true, "UNEXPECTED_CHIPS_FAILURE");
      pending.push({ market: feed.market, kind: feed.kind, targetDate: proof.targetDate, code: feed.code });
      continue;
    }
    ensure(SHA.test(feed.sha256) && Number.isInteger(feed.rowCount) && feed.rowCount > 0
      && feed.rawVerified === true && feed.pointerVerified === true
      && chips.feeds.some(row => row.market === feed.market && row.kind === feed.kind
        && row.date === feed.date && row.sha256 === feed.sha256 && row.rowCount === feed.rowCount
        && row.rawVerified === true && row.pointerVerified === true), "CHIPS_READBACK_FAILED");
  }
  const foreignOwnership = MARKETS.map(market => {
    const current = proof.chips.feeds.filter(row => row.market === market
      && row.kind === "foreign-ownership" && row.date === proof.targetDate);
    ensure(current.length === 1, "FOREIGN_OWNERSHIP_INCOMPLETE");
    const feed = current[0];
    return feed.ok === true
      ? { market, date: feed.date, status: "ready", rows: feed.rowCount, sha256: feed.sha256,
        fetchedAt: feed.fetchedAt }
      : { market, date: null, status: "awaiting-source", code: feed.code };
  });
  const complete = pending.length === 0;
  ensure(complete ? proof.ok === true && proof.chips.ok === true
    && proof.chips.readback?.rawAndPointersVerified === true
    : proof.ok === false && proof.chips.ok === false, "DATA_COMPLETENESS_MISMATCH");
  return {
    schemaVersion: 1, status: complete ? "complete" : "awaiting-source", dataComplete: complete,
    requiresAttention: !complete && (requireComplete || catchup), targetDate: proof.targetDate,
    engineVersion: ENGINE_VERSION, startedAt, checkedAt: finishedAt,
    markets, regulatoryFeeds, foreignOwnership, pending,
  };
}

function collectorEnvironment() {
  const env = {};
  for (const key of ["PATH", "Path", "SystemRoot", "SYSTEMROOT", "TEMP", "TMP",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE"]) if (process.env[key]) env[key] = process.env[key];
  return env;
}

export async function executeCollector(script, cwd, input) {
  return await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["--use-env-proxy", script], {
      cwd, env: collectorEnvironment(), stdio: ["pipe", "pipe", "pipe"], windowsHide: true,
    });
    let bytes = 0, timedOut = false, oversized = false;
    const timer = setTimeout(() => { timedOut = true; child.kill("SIGTERM"); }, 20 * 60_000);
    const hardStop = setTimeout(() => child.kill("SIGKILL"), 20 * 60_000 + 5000);
    const consume = chunk => {
      bytes += chunk.length;
      if (bytes > 8 * 1024 * 1024) { oversized = true; child.kill("SIGTERM"); }
    };
    child.stdout.on("data", consume);
    child.stderr.on("data", consume);
    child.stdin.on("error", () => {});
    child.once("error", () => {
      clearTimeout(timer); clearTimeout(hardStop); reject(new Error("COLLECTOR_START_FAILED"));
    });
    child.once("close", code => {
      clearTimeout(timer); clearTimeout(hardStop);
      if (timedOut) reject(new Error("COLLECTOR_TIMEOUT"));
      else if (oversized) reject(new Error("COLLECTOR_OUTPUT_LIMIT"));
      else if (![0, 1].includes(code)) reject(new Error("COLLECTOR_PROCESS_FAILED"));
      else resolve(code);
    });
    child.stdin.end(JSON.stringify(input) + "\n");
  });
}

function verifyReaderShape(receipt) {
  ensure(receipt?.ok === true && isoDay(receipt.targetDate)
    && receipt.source?.repository === "DingTingdong/surge-data"
    && /^[a-f0-9]{40}$/.test(receipt.source.commitSha) && /^[a-f0-9]{40}$/.test(receipt.source.blobSha)
    && SHA.test(receipt.source.sha256) && Number.isInteger(receipt.source.bytes)
    && receipt.source.bytes > 0 && Array.isArray(receipt.markets) && receipt.markets.length === 2,
  "GITHUB_RECEIPT_INVALID");
}

export function prepareCanonicalInputs(receipt, captures, now = new Date()) {
  verifyReaderShape(receipt);
  ensure(Array.isArray(captures) && captures.length === 2, "GITHUB_CAPTURE_INTEGRITY_INVALID");
  return MARKETS.map(market => {
    const rows = captures.filter(row => row?.market === market);
    const verified = receipt.markets.filter(row => row.market === market);
    ensure(rows.length === 1 && verified.length === 1, "GITHUB_CAPTURE_INTEGRITY_INVALID");
    const c = rows[0];
    ensure(c.date === receipt.targetDate && c.sha256 === verified[0].canonicalSha256
      && SHA.test(c.sha256) && typeof c.rawText === "string" && c.rawText.length > 0
      && Buffer.byteLength(c.rawText) <= 10 * 1024 * 1024
      && createHash("sha256").update(c.rawText).digest("hex") === c.sha256
      && timestamp(c.fetchedAt) <= now.getTime() + 5000
      && timestamp(c.fetchedAt) >= now.getTime() - 15 * 60_000, "GITHUB_CAPTURE_INTEGRITY_INVALID");
    let url;
    try { url = new URL(c.url); } catch { fail("GITHUB_CAPTURE_URL_INVALID"); }
    ensure(url.protocol === "https:" && !url.username && !url.password && !url.hash && !url.port,
      "GITHUB_CAPTURE_URL_INVALID");
    ensure(market === "TWSE"
      ? url.origin === "https://www.twse.com.tw" && url.pathname === "/exchangeReport/MI_INDEX"
        && url.searchParams.get("response") === "json" && url.searchParams.get("type") === "ALLBUT0999"
        && url.searchParams.get("date") === c.date.replaceAll("-", "") && [...url.searchParams].length === 3
      : url.origin === "https://www.tpex.org.tw" && url.pathname === "/www/zh-tw/afterTrading/dailyQuotes"
        && url.searchParams.get("response") === "json"
        && url.searchParams.get("date") === c.date.replaceAll("-", "/") && [...url.searchParams].length === 2,
    "GITHUB_CAPTURE_URL_INVALID");
    return { market, date: c.date, fetchedAt: c.fetchedAt, sha256: c.sha256, url: c.url, rawText: c.rawText };
  });
}

export function verifyGitHubReceipt(receipt, proof) {
  verifyReaderShape(receipt);
  ensure(receipt.targetDate === proof.targetDate, "GITHUB_RECEIPT_INVALID");
  for (const market of MARKETS) {
    const matches = receipt.markets.filter(row => row.market === market);
    const provider = proof.providers?.find(row => row.market === market);
    ensure(matches.length === 1 && matches[0].date === proof.targetDate
      && matches[0].priceMatched === true && matches[0].ordinaryRows === matches[0].officialRows
      && Number.isInteger(matches[0].ordinaryRows) && matches[0].ordinaryRows >= 300
      && matches[0].ordinaryRows === provider?.receipt?.rowCount
      && Number.isInteger(matches[0].csvRows) && matches[0].csvRows >= matches[0].ordinaryRows
      && Number.isInteger(matches[0].volumeDifferenceCount) && matches[0].volumeDifferenceCount >= 0
      && Number.isInteger(matches[0].amountDifferenceCount) && matches[0].amountDifferenceCount >= 0
      && (market === "TWSE"
        ? matches[0].volumePolicy === "exact" && matches[0].volumeDifferenceCount === 0 && matches[0].amountDifferenceCount === 0
        : matches[0].volumePolicy === "ordinary-session-compared-to-broad-canonical-retained")
      && matches[0].canonicalSha256 === provider?.sha256, "GITHUB_CANONICAL_READBACK_MISMATCH");
  }
  return {
    repository: receipt.source.repository, commitSha: receipt.source.commitSha,
    csvSha256: receipt.source.sha256, targetDate: receipt.targetDate,
    markets: receipt.markets.map(row => ({
      market: row.market, csvRows: row.csvRows, ordinaryRows: row.ordinaryRows, priceMatched: row.priceMatched,
      volumeDifferenceCount: row.volumeDifferenceCount, amountDifferenceCount: row.amountDifferenceCount,
      volumePolicy: row.volumePolicy,
    })),
  };
}

export function summaryMarkdown(summary) {
  if (summary.status === "failed") return "## 雷達後台更新失敗\n\n錯誤代碼：" + summary.code
    + "\n\n整體驗收未通過，部分來源可能已更新；請依網站各來源日期核對。\n";
  if (summary.status === "awaiting-github") return "## GitHub 原始行情尚未就緒\n\n"
    + "- 需要的交易日：" + summary.targetDate
    + "\n- 完整更新：**否**。原始行情未通過核對，保留既有資料與原日期。"
    + "\n- 由後續背景執行重試，不需要開電腦。\n";
  const lines = [
    "## " + (summary.dataComplete ? "雷達資料已完整更新" : "等待官方資料：本輪尚未齊全"),
    "", "- 資料日期：" + summary.targetDate, "- 執行方式：GitHub 雲端後台",
    "- 選股引擎：" + summary.engineVersion, "- 完成核對：" + summary.checkedAt,
    "", "| 市場 | 行情筆數 | 原件與快照 |", "| --- | ---: | --- |",
    ...summary.markets.map(row => "| " + row.market + " | " + row.rows + " | 已核對 |"),
    "", "注意／處置四個來源：均已核對入庫。", "",
    "| 市場 | 外資持股日期 | 筆數／狀態 |", "| --- | --- | --- |",
    ...summary.foreignOwnership.map(row => "| " + row.market + " | " + (row.date ?? "尚待官方公布")
      + " | " + (row.rows ?? "未完成") + " |"),
  ];
  if (!summary.dataComplete) lines.push("", "完整更新：**否**。保留各來源原日期，由後續背景執行重試。");
  if (summary.requiresAttention) lines.push("", "**此輪必須完整更新，現仍有缺口，已標示失敗。**");
  if (summary.githubSource) lines.push("", "原始行情來源：DingTingdong/surge-data",
    "- 來源 commit：" + summary.githubSource.commitSha,
    "- CSV SHA-256：" + summary.githubSource.csvSha256,
    "- 上櫃量額保留原官方綜合口徑；已核對 GitHub 普通交易口徑差異。");
  return lines.join("\n") + "\n";
}

export async function runBackend({
  token, githubReadToken, requireComplete = false, now = () => new Date(), fetcher = fetch,
  execute = executeCollector, temporaryRoot = tmpdir(),
} = {}) {
  ensure(typeof token === "string" && token.length >= 20 && !/[\r\n]/.test(token), "SITE_SECRET_MISSING");
  ensure(Number(process.versions.node.split(".")[0]) >= 24, "NODE_24_REQUIRED");
  const startedAt = now().toISOString(), temporaryParent = await realpath(temporaryRoot);
  const temporary = await mkdtemp(join(temporaryParent, "surge-radar-"));
  try {
    const localDay = new Date(Date.parse(startedAt) + 8 * 60 * 60_000).toISOString().slice(0, 10);
    const download = async kind => {
      const manifest = validateManifest(await jsonRequest(token, ARTIFACTS[kind].manifest,
        { fetcher, limit: 32768 }), kind);
      const bytes = await request(token, manifest.path, { fetcher, limit: 4 * 1024 * 1024 });
      ensure(bytes.length === manifest.bytes
        && createHash("sha256").update(bytes).digest("hex") === manifest.sha256, "COLLECTOR_HASH_MISMATCH");
      const path = join(temporary, kind + ".mjs");
      await writeFile(path, bytes, { flag: "wx", mode: 0o600 });
      return { path, manifest };
    };
    const reader = await download("reader");
    const readerExit = await execute(reader.path, temporary,
      githubReadToken ? { githubToken: githubReadToken } : {});
    let githubReceipt;
    try { githubReceipt = JSON.parse(await readFile(join(temporary, "outputs", "github-source-verification.json"), "utf8")); }
    catch { fail("GITHUB_RECEIPT_MISSING"); }
    if (readerExit === 1 && githubReceipt.ok === false && githubReceipt.code === "GITHUB_CSV_SOURCE_NOT_READY") {
      ensure(isoDay(githubReceipt.targetDate) && githubReceipt.targetDate <= localDay, "GITHUB_RECEIPT_INVALID");
      return { schemaVersion: 1, status: "awaiting-github", dataComplete: false,
        requiresAttention: requireComplete || githubReceipt.targetDate < localDay,
        targetDate: githubReceipt.targetDate, checkedAt: now().toISOString() };
    }
    ensure(readerExit === 0 && githubReceipt.ok === true,
      /^[A-Z][A-Z0-9_]{2,79}$/.test(githubReceipt.code ?? "") ? githubReceipt.code : "GITHUB_READER_FAILED");
    const capturePath = join(temporary, "outputs", "github-canonical-captures.json");
    let captures;
    try {
      ensure((await stat(capturePath)).size <= 24 * 1024 * 1024, "GITHUB_CAPTURE_INTEGRITY_INVALID");
      captures = JSON.parse(await readFile(capturePath, "utf8"));
    } catch { fail("GITHUB_CAPTURE_INTEGRITY_INVALID"); }
    const canonicalCaptures = prepareCanonicalInputs(githubReceipt, captures, now());
    const writer = await download("writer");
    const exitCode = await execute(writer.path, temporary, { origin: SITE_ORIGIN, token, mode: "automatic",
      expectedTargetDate: githubReceipt.targetDate,
      expectedCanonicalSha256: Object.fromEntries(githubReceipt.markets.map(row => [row.market, row.canonicalSha256])),
      canonicalCaptures });
    let proof;
    try { proof = JSON.parse(await readFile(join(temporary, "outputs", "automatic-update-verification.json"), "utf8")); }
    catch { fail("COLLECTOR_PROOF_MISSING"); }
    const results = await Promise.allSettled([
      jsonRequest(token, "/api/radar?fresh=1", { fetcher }),
      jsonRequest(token, "/api/chips?view=coverage", { fetcher }),
      jsonRequest(token, "/api/regulatory", { fetcher }),
    ]);
    for (const result of results) if (result.status === "rejected") throw result.reason;
    const [radar, chips, regulatory] = results.map(result => result.value);
    const githubSource = verifyGitHubReceipt(githubReceipt, proof);
    const summary = verifyUpdate(proof, { radar, chips, regulatory }, {
      startedAt, finishedAt: now().toISOString(), collectorExitCode: exitCode, requireComplete,
      catchup: isoDay(proof.targetDate) && proof.targetDate < localDay,
    });
    return { ...summary, githubSource, collectorSha256: writer.manifest.sha256,
      readerSha256: reader.manifest.sha256 };
  } finally {
    const cleanup = await realpath(temporary);
    ensure(dirname(cleanup) === temporaryParent && basename(cleanup).startsWith("surge-radar-"),
      "TEMPORARY_PATH_INVALID");
    await rm(cleanup, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  let result;
  try {
    result = await runBackend({
      token: process.env.SURGE_RADAR_SITE_TOKEN,
      githubReadToken: process.env.GITHUB_READ_TOKEN,
      requireComplete: process.env.SURGE_REQUIRE_COMPLETE === "true",
    });
  } catch (error) {
    result = { schemaVersion: 1, status: "failed", dataComplete: false, code: safeCode(error) };
  }
  const message = summaryMarkdown(result);
  if (process.env.GITHUB_STEP_SUMMARY) await appendFile(process.env.GITHUB_STEP_SUMMARY, message);
  console.log(JSON.stringify(result));
  if (result.status === "failed" || result.requiresAttention) process.exitCode = 1;
}
