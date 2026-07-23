"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity } from "lucide-react";

import { getHealth } from "@/lib/api/openapi-client";
import { queryKeys } from "@/lib/api/query-keys";

import { StatusBadge } from "../ui/status-badge";

export function HealthIndicator() {
  const health = useQuery({
    queryKey: queryKeys.health,
    queryFn: () => getHealth().then((response) => response.data),
    refetchInterval: 15_000
  });

  const status = health.isError
    ? "unavailable"
    : health.data?.service.status || (health.isPending ? "processing" : "unavailable");

  return (
    <div className="health-indicator" title="系统健康状态">
      <Activity size={16} aria-hidden="true" />
      <StatusBadge value={status} />
    </div>
  );
}
