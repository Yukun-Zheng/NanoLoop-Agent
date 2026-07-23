import type { LucideIcon } from "lucide-react";

export function EmptyState({
  icon: Icon,
  title,
  detail,
  action
}: {
  icon: LucideIcon;
  title: string;
  detail: string;
  action?: React.ReactNode;
}) {
  return (
    <section className="empty-state">
      <span className="empty-icon">
        <Icon size={20} aria-hidden="true" />
      </span>
      <h3>{title}</h3>
      <p>{detail}</p>
      {action}
    </section>
  );
}
