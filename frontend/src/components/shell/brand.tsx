import Link from "next/link";

export function Brand() {
  return (
    <Link className="brand" href="/" aria-label="NanoLoop Agent 首页">
      <span className="brand-mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span>
        <strong>NanoLoop</strong>
        <small>Scientific Agent</small>
      </span>
    </Link>
  );
}
