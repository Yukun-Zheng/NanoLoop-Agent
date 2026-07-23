import type { HealthData } from "@/lib/api/types";

export function coreMutationBlocker(
  health: HealthData | null | undefined,
  state: { failed?: boolean; pending?: boolean } = {}
): string | null {
  if (state.failed) return "无法确认核心服务健康，写入与运行操作已暂停。";
  if (!health) {
    return state.pending
      ? "正在确认核心服务健康，写入与运行操作暂不可用。"
      : "尚未取得核心服务健康状态，写入与运行操作已暂停。";
  }
  if (health.service.status !== "healthy") {
    return `服务状态为 ${health.service.status}，写入与运行操作已暂停。`;
  }
  if (health.database.status !== "healthy") {
    return `数据库状态为 ${health.database.status}，写入与运行操作已暂停。`;
  }
  return null;
}
