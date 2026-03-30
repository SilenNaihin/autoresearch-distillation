import { readFile } from "fs/promises";
import { join } from "path";

export async function GET() {
  try {
    const path = join(process.cwd(), "public", "data", "benchmark.json");
    const data = await readFile(path, "utf-8");
    return new Response(data, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json({ stats: {}, scenarios: [] });
  }
}
