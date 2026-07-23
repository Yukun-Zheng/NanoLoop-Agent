import Link from "next/link";

export default function NotFound() {
  return (
    <main className="centered-state">
      <div className="empty-symbol">404</div>
      <h1>没有找到这个科研任务</h1>
      <p>任务可能不存在，或当前身份没有读取权限。</p>
      <Link className="button button-primary" href="/">
        返回任务首页
      </Link>
    </main>
  );
}
