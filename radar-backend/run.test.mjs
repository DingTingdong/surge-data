import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  ENGINE_VERSION, SITE_ORIGIN, prepareCanonicalInputs, runBackend, safeCode, summaryMarkdown,
  validateManifest, verifyGitHubReceipt, verifyUpdate,
} from "./run.mjs";

const now = "2026-11-16T14:00:00.000Z", target = "2026-11-16";
const rawText = JSON.stringify({ fixture: true });
const digest = createHash("sha256").update(rawText).digest("hex");
const options = { startedAt: now, finishedAt: now, collectorExitCode: 0 };
function fixture() {
  const sessions = ["2026-11-16", "2026-11-13", "2026-11-12", "2026-11-11", "2026-11-10",
    "2026-11-09", "2026-11-06", "2026-11-05", "2026-11-04", "2026-11-03",
    "2026-11-02", "2026-10-30", "2026-10-29", "2026-10-28", "2026-10-27",
    "2026-10-23", "2026-10-22", "2026-10-21", "2026-10-20", "2026-10-19"];
  const ownership = [target, "2026-11-09", "2026-10-16"];
  const provider = market => ({
    market, ok: true, date: target, fetchedAt: now, sha256: digest,
    sourceUrl: market === "TWSE"
      ? "https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date=20261116&type=ALLBUT0999"
      : "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes?response=json&date=2026%2F11%2F16",
    receipt: { market, date: target, sha256: digest, rowCount: 900,
      rawVerified: true, pointerVerified: true, journalVerified: true },
  });
  const feed = (market, kind, date) => ({
    market, kind, date, ok: true, sha256: digest, rowCount: 900, fetchedAt: now,
    rawVerified: true, pointerVerified: true,
  });
  const chipFeeds = [feed("TDCC", "holders", "2026-11-13")];
  const regulatoryFeeds = [];
  for (const market of ["TWSE", "TPEX"]) {
    chipFeeds.push(...sessions.map(date => feed(market, "institutional", date)));
    chipFeeds.push(...ownership.map(date => feed(market, "foreign-ownership", date)));
    for (const kind of ["attention", "disposition"]) regulatoryFeeds.push({
      market, kind, ok: true, targetDate: target, fetchedAt: now,
      rowCount: 4, sha256: digest, rawVerified: true, pointerVerified: true, recordsVerified: true,
    });
  }
  const proof = {
    origin: SITE_ORIGIN, mode: "automatic", targetDate: target, checkedAt: now, ok: true,
    priceOk: true, actionReferencesOk: true, referenceRecovery: { ok: true }, backfill: { ok: true },
    evidence: { ok: true }, context: { ok: true }, contextConsistent: true, regulatoryConsistent: true,
    verification: { ok: true, pending: [] }, providers: [provider("TWSE"), provider("TPEX")],
    regulatory: { ok: true, assessmentDate: target, feeds: regulatoryFeeds, readback: { ok: true } },
    chips: { ok: true, targetDate: target, feeds: chipFeeds, history: [{ ok: true }],
      readback: { rawAndPointersVerified: true } },
  };
  const readback = {
    radar: { meta: { version: ENGINE_VERSION, asOf: target, markets: {} }, journal: { markets: {} } },
    chips: { targetDate: target, feeds: structuredClone(chipFeeds), expectedHolderDate: "2026-11-13",
      sessions: { TWSE: sessions, TPEX: [...sessions] },
      ownershipDates: { TWSE: ownership, TPEX: [...ownership] } },
    regulatory: { feeds: regulatoryFeeds.map(row => ({ ...row, status: "ready", referenceDate: target })) },
  };
  for (const p of proof.providers) {
    readback.radar.meta.markets[p.market] = {
      date: target, count: 900, sha256: digest, sourceUrl: p.sourceUrl, fetchedAt: now,
    };
    readback.radar.journal.markets[p.market] = {
      date: target, capturedAt: now, previousDate: "2026-11-13", status: "compared",
    };
  }
  const githubReceipt = {
    ok: true, targetDate: target,
    source: { repository: "DingTingdong/surge-data", commitSha: "c".repeat(40),
      blobSha: "d".repeat(40), sha256: "e".repeat(64), bytes: 50000 },
    markets: ["TWSE", "TPEX"].map(market => ({
      market, date: target, priceMatched: true, csvRows: 900, ordinaryRows: 900,
      officialRows: 900, canonicalSha256: digest, volumeDifferenceCount: market === "TWSE" ? 0 : 100,
      amountDifferenceCount: market === "TWSE" ? 0 : 100,
      volumePolicy: market === "TWSE" ? "exact" : "ordinary-session-compared-to-broad-canonical-retained",
    })),
  };
  const githubCaptures = proof.providers.map(row => ({ market: row.market, date: target,
    fetchedAt: now, url: row.sourceUrl, sha256: digest, rawText }));
  return { proof, readback, githubReceipt, githubCaptures };
}
function pendingFixture() {
  const f = fixture();
  f.proof.ok = false; f.proof.chips.ok = false; f.proof.chips.readback.rawAndPointersVerified = false;
  const p = f.proof.chips.feeds.find(row => row.market === "TWSE"
    && row.kind === "foreign-ownership" && row.date === target);
  Object.assign(p, { ok: false, code: "CHIPS_SOURCE_DATE_NOT_READY", failureRecorded: true });
  f.readback.chips.feeds = f.readback.chips.feeds.filter(row => !(row.market === "TWSE"
    && row.kind === "foreign-ownership" && row.date === target));
  return f;
}

test("a complete run requires persisted markets, original objects, all chip sessions and four regulatory feeds", () => {
  const f = fixture(), result = verifyUpdate(f.proof, f.readback, options);
  assert.equal(result.status, "complete");
  assert.equal(result.dataComplete, true);
  assert.equal(result.markets.length, 2);
  assert.equal(result.foreignOwnership[0].date, target);
  assert.equal(result.requiresAttention, false);
});

for (const [name, mutate, expected] of [
  ["exit zero alone", f => { f.proof.ok = false; }, "COLLECTOR_EXIT_MISMATCH"],
  ["old proof", f => { f.proof.checkedAt = "2026-11-13T14:00:00.000Z"; }, "COLLECTOR_PROOF_INVALID"],
  ["changed formula version", f => { f.readback.radar.meta.version = "surge-2.0.0"; }, "RADAR_ENGINE_OR_DATE_MISMATCH"],
  ["one market missing", f => { f.proof.providers.pop(); }, "MARKETS_INCOMPLETE"],
  ["stale market day", f => { f.readback.radar.meta.markets.TPEX.date = "2026-11-13"; }, "MARKET_OR_SNAPSHOT_READBACK_FAILED"],
  ["unverified raw object", f => { f.proof.providers[0].receipt.rawVerified = false; }, "MARKET_OR_SNAPSHOT_READBACK_FAILED"],
  ["snapshot absent", f => { delete f.readback.radar.journal.markets.TPEX; }, "MARKET_OR_SNAPSHOT_READBACK_FAILED"],
  ["snapshot status absent", f => { delete f.readback.radar.journal.markets.TWSE.status; }, "MARKET_OR_SNAPSHOT_READBACK_FAILED"],
  ["snapshot failed", f => { f.readback.radar.journal.markets.TWSE.status = "failed"; }, "MARKET_OR_SNAPSHOT_READBACK_FAILED"],
  ["raw hash mismatch", f => { f.readback.radar.meta.markets.TWSE.sha256 = "b".repeat(64); }, "MARKET_OR_SNAPSHOT_READBACK_FAILED"],
  ["regulatory stale attempt", f => { f.proof.regulatory.feeds[0].fetchedAt = "2026-11-13T14:00:00Z"; }, "REGULATORY_READBACK_FAILED"],
  ["regulatory duplicate", f => { f.proof.regulatory.feeds[0] = f.proof.regulatory.feeds[1]; }, "REGULATORY_NOT_VERIFIED"],
  ["matching but wrong announcement day", f => {
    f.proof.regulatory.feeds[0].targetDate = "2000-01-01";
    f.readback.regulatory.feeds[0].referenceDate = "2000-01-01";
  }, "REGULATORY_READBACK_FAILED"],
  ["chip session missing", f => { f.proof.chips.feeds[1] = f.proof.chips.feeds[2]; }, "CHIPS_FEED_MISSING"],
  ["chip raw integrity failed", f => { f.readback.chips.feeds[0].rawVerified = false; }, "CHIPS_READBACK_FAILED"],
  ["wrong holder week", f => { f.readback.chips.expectedHolderDate = target; }, "HOLDER_WEEK_NOT_VERIFIED"],
  ["history failed", f => { f.proof.chips.history[0].ok = false; }, "CHIPS_NOT_VERIFIED"],
]) test("rejects " + name, () => {
  const f = fixture(); mutate(f);
  assert.throws(() => verifyUpdate(f.proof, f.readback, options), { message: expected });
});

test("known unpublished foreign source remains explicitly incomplete without inventing today's date", () => {
  const f = pendingFixture();
  const result = verifyUpdate(f.proof, f.readback, { ...options, collectorExitCode: 1 });
  assert.equal(result.dataComplete, false);
  assert.equal(result.status, "awaiting-source");
  assert.equal(result.foreignOwnership[0].date, null);
  assert.equal(result.requiresAttention, false);
  assert.match(summaryMarkdown(result), /完整更新：\*\*否\*\*/);
  assert.doesNotMatch(summaryMarkdown(result), /資料已完整更新/);
});

test("immediate acceptance and following-morning catchup require every source to be complete", () => {
  const f = pendingFixture();
  for (const extra of [{ requireComplete: true }, { catchup: true }]) {
    const r = verifyUpdate(f.proof, f.readback, { ...options, collectorExitCode: 1, ...extra });
    assert.equal(r.requiresAttention, true);
  }
});

test("schema failures cannot be disguised as unpublished data", () => {
  const f = pendingFixture();
  f.proof.chips.feeds.find(row => row.ok === false).code = "CHIPS_FIELDS_INVALID";
  assert.throws(() => verifyUpdate(f.proof, f.readback, { ...options, collectorExitCode: 1 }),
    { message: "UNEXPECTED_CHIPS_FAILURE" });
});

test("a valid manifest cannot redirect the credential or substitute another executable", () => {
  const m = { format: "node-esm", version: 1, nodeMajor: 24,
    path: "/_ops/collector.mjs", bytes: 1200, sha256: digest };
  assert.equal(validateManifest(m), m);
  for (const patch of [{ path: "https://evil.test/script.mjs" }, { path: "/_ops/other.mjs" },
    { bytes: 0 }, { bytes: 9_000_000 }, { sha256: "bad" }, { nodeMajor: 22 }]) {
    assert.throws(() => validateManifest({ ...m, ...patch }), { message: "COLLECTOR_MANIFEST_INVALID" });
  }
});

test("raw exceptions and private information never appear in public failure summaries", () => {
  assert.equal(safeCode(new Error("token=private-token-value")), "BACKEND_UPDATE_FAILED");
  const text = summaryMarkdown({ status: "failed", code: safeCode(new Error("secret")) });
  assert.doesNotMatch(text, /secret|token=/);
  assert.match(text, /整體驗收未通過/);
});

test("bootstrap verifies the private artifact, executes it, reads all persisted results and cleans only its own temporary directory", async () => {
  const f = fixture(), script = Buffer.from("// trusted test artifact\n".repeat(60));
  const manifest = { format: "node-esm", version: 1, nodeMajor: 24, path: "/_ops/collector.mjs",
    bytes: script.length, sha256: createHash("sha256").update(script).digest("hex") };
  let directory, executed = 0;
  const requests = [];
  const token = "unit-test-credential-only";
  const result = await runBackend({
    token, now: () => new Date(now),
    fetcher: async (url, init) => {
      requests.push(url);
      assert.equal(init.redirect, "manual");
      assert.equal(init.headers["OAI-Sites-Authorization"], "Bearer " + token);
      assert.ok(url.startsWith(SITE_ORIGIN + "/"));
      const value = url.endsWith("/github-source-manifest.json") ? { ...manifest, path: "/_ops/github-source-reader.mjs" }
        : url.endsWith("/manifest.json") ? manifest
          : url.endsWith(".mjs") ? script
          : url.includes("/api/radar") ? f.readback.radar
            : url.includes("/api/chips") ? f.readback.chips : f.readback.regulatory;
      return new Response(Buffer.isBuffer(value) ? value : JSON.stringify(value));
    },
    execute: async (path, cwd, input) => {
      executed++; directory = cwd;
      assert.deepEqual(await readFile(path), script);
      await mkdir(join(cwd, "outputs"), { recursive: true });
      if (path.endsWith("reader.mjs")) {
        assert.equal(input.targetDate, undefined, "target comes from fresh official calendar, not yesterday's stored benchmarks");
        assert.equal(input.token, undefined);
        await writeFile(join(cwd, "outputs", "github-source-verification.json"), JSON.stringify(f.githubReceipt));
        await writeFile(join(cwd, "outputs", "github-canonical-captures.json"), JSON.stringify(f.githubCaptures));
      } else {
        assert.equal(input.token, token);
        assert.equal(input.githubToken, undefined);
        assert.equal(input.expectedTargetDate, target);
        assert.deepEqual(input.canonicalCaptures, f.githubCaptures);
        await writeFile(join(cwd, "outputs", "automatic-update-verification.json"), JSON.stringify(f.proof));
      }
      return 0;
    },
  });
  assert.equal(executed, 2);
  assert.equal(result.dataComplete, true);
  assert.equal(requests.length, 7);
  assert.equal(result.githubSource.commitSha, f.githubReceipt.source.commitSha);
  assert.doesNotMatch(JSON.stringify(result), new RegExp(token));
  await assert.rejects(readdir(directory), { code: "ENOENT" });
});

test("both captured raw documents must match the approved reader hashes before the writer starts", () => {
  const f = fixture();
  assert.deepEqual(prepareCanonicalInputs(f.githubReceipt, f.githubCaptures, new Date(now)), f.githubCaptures);
  f.githubCaptures[1].rawText += " ";
  assert.throws(() => prepareCanonicalInputs(f.githubReceipt, f.githubCaptures, new Date(now)),
    { message: "GITHUB_CAPTURE_INTEGRITY_INVALID" });
});

test("overnight checks keep the price trading day separate from the disposition assessment day", () => {
  const f = fixture(), morning = "2026-11-17T00:23:00.000Z";
  f.proof.checkedAt = morning;
  f.proof.regulatory.assessmentDate = "2026-11-17";
  f.proof.regulatory.feeds.forEach(row => {
    row.fetchedAt = morning;
    if (row.kind === "disposition") row.targetDate = "2026-11-17";
  });
  f.readback.regulatory.feeds.forEach(row => {
    row.fetchedAt = morning;
    if (row.kind === "disposition") row.referenceDate = "2026-11-17";
  });
  assert.equal(verifyUpdate(f.proof, f.readback,
    { ...options, startedAt: morning, finishedAt: morning }).dataComplete, true);
});

test("a modified artifact is never executed", async () => {
  let executed = false;
  await assert.rejects(runBackend({
    token: "unit-test-credential-only", now: () => new Date(now),
    fetcher: async url => new Response(url.includes("/api/chips") ? JSON.stringify(fixture().readback.chips)
      : url.endsWith(".json")
      ? JSON.stringify({ format: "node-esm", version: 1, nodeMajor: 24,
        path: "/_ops/github-source-reader.mjs", bytes: 1200, sha256: digest }) : "x".repeat(1200)),
    execute: async () => { executed = true; return 0; },
  }), { message: "COLLECTOR_HASH_MISMATCH" });
  assert.equal(executed, false);
});

test("GitHub's ordinary-session turnover cannot silently replace the canonical strategy input", () => {
  const f = fixture();
  const summary = verifyGitHubReceipt(f.githubReceipt, f.proof);
  assert.equal(summary.markets[1].volumeDifferenceCount, 100);
  f.githubReceipt.markets[1].volumePolicy = "replaced-with-github";
  assert.throws(() => verifyGitHubReceipt(f.githubReceipt, f.proof),
    { message: "GITHUB_CANONICAL_READBACK_MISMATCH" });
});

test("GitHub input must match the raw object actually stored by this writer run", () => {
  const f = fixture();
  f.githubReceipt.markets[0].canonicalSha256 = "b".repeat(64);
  assert.throws(() => verifyGitHubReceipt(f.githubReceipt, f.proof),
    { message: "GITHUB_CANONICAL_READBACK_MISMATCH" });
});

test("missing credentials fail before any network operation", async () => {
  let requested = false;
  await assert.rejects(runBackend({ fetcher: async () => { requested = true; } }),
    { message: "SITE_SECRET_MISSING" });
  assert.equal(requested, false);
});

test("unpublished GitHub data does not run the writer or become a completed update", async () => {
  const f = fixture(), bytes = Buffer.from("// reader fixture\n".repeat(70));
  const manifest = { format: "node-esm", version: 1, nodeMajor: 24,
    path: "/_ops/github-source-reader.mjs", bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex") };
  let executions = 0;
  const result = await runBackend({
    token: "unit-test-credential-only", now: () => new Date(now), requireComplete: true,
    fetcher: async url => {
      assert.ok(!url.endsWith("/_ops/manifest.json"), "writer must not be downloaded");
      return new Response(url.includes("/api/chips") ? JSON.stringify(f.readback.chips)
        : url.endsWith(".json") ? JSON.stringify(manifest) : bytes);
    },
    execute: async (path, cwd, input) => {
      executions++;
      assert.ok(path.endsWith("reader.mjs"));
      assert.equal(input.token, undefined);
      await mkdir(join(cwd, "outputs"), { recursive: true });
      await writeFile(join(cwd, "outputs", "github-source-verification.json"),
        JSON.stringify({ ok: false, targetDate: target, code: "GITHUB_CSV_SOURCE_NOT_READY" }));
      return 1;
    },
  });
  assert.equal(executions, 1);
  assert.equal(result.dataComplete, false);
  assert.equal(result.status, "awaiting-github");
  assert.equal(result.requiresAttention, true);
  assert.match(summaryMarkdown(result), /完整更新：\*\*否\*\*/);
});
