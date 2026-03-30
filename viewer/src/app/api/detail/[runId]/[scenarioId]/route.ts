import { readFile, access } from "fs/promises";
import { execSync } from "child_process";
import { join } from "path";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string; scenarioId: string }> }
) {
  const { runId, scenarioId } = await params;
  const detailPath = join(
    process.cwd(),
    "public",
    "data",
    "runs",
    runId,
    "details",
    `${scenarioId}.json`
  );

  // Try reading directly first
  try {
    const data = await readFile(detailPath, "utf-8");
    return new Response(data, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    // File doesn't exist locally — try lazy fetch via sync script
  }

  // Lazy fetch: call viewer_sync's fetch_single_detail
  try {
    const scriptPath = join(process.cwd(), "..", "scripts", "viewer_sync.py");
    await access(scriptPath);
    execSync(
      `python3 ${scriptPath} --fetch-detail ${runId} ${scenarioId}`,
      { timeout: 30000, stdio: "pipe" }
    );
    const data = await readFile(detailPath, "utf-8");
    return new Response(data, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json({ error: "Not found" }, { status: 404 });
  }
}
