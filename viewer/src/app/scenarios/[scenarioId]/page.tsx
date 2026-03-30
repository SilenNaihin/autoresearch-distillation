import { ScenarioDetailClient } from "./detail-client";

export default async function Page({
  params,
  searchParams,
}: {
  params: Promise<{ scenarioId: string }>;
  searchParams: Promise<{ run?: string }>;
}) {
  const { scenarioId } = await params;
  const { run } = await searchParams;
  return <ScenarioDetailClient scenarioId={scenarioId} runId={run ?? ""} />;
}
