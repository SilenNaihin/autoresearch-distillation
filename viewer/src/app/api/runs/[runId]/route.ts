import { rm, readFile, writeFile } from "fs/promises";
import { join } from "path";

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ runId: string }> }
) {
  const { runId } = await params;
  const dataDir = join(process.cwd(), "public", "data");
  const runDir = join(dataDir, "runs", runId);

  // Delete the run directory
  try {
    await rm(runDir, { recursive: true, force: true });
  } catch {
    return Response.json({ error: "Failed to delete run" }, { status: 500 });
  }

  // Update index.json to remove this run
  try {
    const indexPath = join(dataDir, "index.json");
    const idx = JSON.parse(await readFile(indexPath, "utf-8"));
    idx.runs = (idx.runs ?? []).filter(
      (r: { run_id: string }) => r.run_id !== runId
    );
    await writeFile(indexPath, JSON.stringify(idx, null, 2));
  } catch {
    // Index update failed but run is already deleted
  }

  return Response.json({ ok: true });
}
