export default function Loading() {
  return (
    <main className="centered-state" aria-live="polite">
      <span className="status-spinner" aria-hidden="true" />
      <p>正在载入科研任务…</p>
    </main>
  );
}
