import { readFile } from "fs/promises";
import { join } from "path";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string; scenarioId: string }> }
) {
  const { runId, scenarioId } = await params;
  try {
    const path = join(
      process.cwd(),
      "public",
      "data",
      "runs",
      runId,
      "details",
      `${scenarioId}.json`
    );
    const data = await readFile(path, "utf-8");
    return new Response(data, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json({ error: "Not found" }, { status: 404 });
  }
}
