import { execSync } from "child_process";
import { join } from "path";
import { readFile } from "fs/promises";

export const dynamic = "force-dynamic";

export async function GET() {
  const scriptPath = join(process.cwd(), "..", "scripts", "viewer_sync.py");

  // Run sync script in training-only mode to refresh index.json
  try {
    execSync(`python3 ${scriptPath} --training-only`, {
      timeout: 30000,
      stdio: "pipe",
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
  } catch (e) {
    console.error("[/api/training] sync failed:", (e as { stderr?: Buffer })?.stderr?.toString() ?? e);
  }

  // Read the updated index and return just training_runs
  try {
    const indexPath = join(process.cwd(), "public", "data", "index.json");
    const idx = JSON.parse(await readFile(indexPath, "utf-8"));
    return Response.json(idx.training_runs ?? []);
  } catch {
    return Response.json([]);
  }
}
