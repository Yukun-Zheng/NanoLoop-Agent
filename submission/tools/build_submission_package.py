#!/usr/bin/env python3
"""Build the private AI4S submission ZIP from a clean Git revision.

The registration form is supplied explicitly at build time and is never copied
into the public repository.  The source tree is produced with ``git archive``,
so ignored machine-local files cannot leak into the judging package.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission"
GENERATED = SUBMISSION / "generated"
PACKAGE_TEMPLATES = SUBMISSION / "package"
OUTPUT_DIR = ROOT / "dist" / "competition"
PACKAGE_NAME = "杨雨宁+纳米颗粒图像识别工具开发小组+参赛作品.zip"
VIDEO_NAME = "NanoLoop三分钟演示视频_纳米颗粒图像识别工具开发小组.mp4"


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"缺少交付文件：{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def export_source(destination: Path) -> None:
    archive = destination.parent / "nanoloop-source.tar"
    with archive.open("wb") as handle:
        subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"],
            cwd=ROOT,
            stdout=handle,
            check=True,
        )
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source_tar:
        source_tar.extractall(destination, filter="data")
    archive.unlink()


def install_root_launchers(source_root: Path) -> None:
    launcher_dir = SUBMISSION / "docker"
    for launcher in sorted(launcher_dir.iterdir()):
        if launcher.is_file():
            shutil.copy2(launcher, source_root / launcher.name)


def write_example_readme(destination: Path) -> None:
    destination.write_text(
        "示例图 nanoloop_ui_acceptance_fixture.png 为公开验收夹具，"
        "仅用于检查上传、ROI、运行和结果页面是否正常，不代表模型科学精度。\n"
        "第一次验收可直接选择该图；如需展示真实模型效果，请使用团队已授权的原始 SEM 图像。\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(package_root: Path) -> None:
    rows: list[str] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            relative = path.relative_to(package_root).as_posix()
            rows.append(f"{sha256(path)}  {relative}")
    (package_root / "SHA256SUMS.txt").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def zip_tree(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as output:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(source).as_posix())


def build(
    registration_form: Path,
    video: Path | None,
    offline_image: Path | None,
) -> tuple[Path, Path | None]:
    if not registration_form.is_file():
        raise FileNotFoundError(f"找不到报名表：{registration_form}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_zip = OUTPUT_DIR / PACKAGE_NAME
    separate_offline: Path | None = None

    with tempfile.TemporaryDirectory(prefix="nanoloop-submission-") as temporary:
        package_root = Path(temporary) / "NanoLoop参赛作品"

        copy_required(
            PACKAGE_TEMPLATES / "README_先看这里.txt",
            package_root / "README_先看这里.txt",
        )
        for stem in ("NanoLoop智能体设计文档",):
            for suffix in (".pdf", ".docx"):
                copy_required(
                    GENERATED / f"{stem}{suffix}",
                    package_root / "01_智能体设计文档" / f"{stem}{suffix}",
                )

        docker_dir = package_root / "02_Docker部署包"
        source_root = docker_dir / "NanoLoop-Agent"
        export_source(source_root)
        install_root_launchers(source_root)
        for suffix in (".pdf", ".docx"):
            copy_required(
                GENERATED / f"NanoLoop_Docker部署与使用手册{suffix}",
                docker_dir / f"NanoLoop_Docker部署与使用手册{suffix}",
            )
        example_dir = docker_dir / "示例输入"
        copy_required(
            ROOT / "demo_data" / "acceptance" / "nanoloop_ui_acceptance_fixture.png",
            example_dir / "nanoloop_ui_acceptance_fixture.png",
        )
        write_example_readme(example_dir / "README_示例图说明.txt")

        video_dir = package_root / "03_三分钟演示视频"
        for suffix in (".pdf", ".docx"):
            copy_required(
                GENERATED / f"NanoLoop三分钟演示录制手册{suffix}",
                video_dir / f"NanoLoop三分钟演示录制手册{suffix}",
            )
        copy_required(
            SUBMISSION / "assets" / "diagrams" / "video-title.png",
            video_dir / "片头.png",
        )
        copy_required(
            SUBMISSION / "assets" / "diagrams" / "video-end.png",
            video_dir / "片尾.png",
        )
        if video is not None:
            copy_required(video, video_dir / VIDEO_NAME)
        else:
            copy_required(
                PACKAGE_TEMPLATES / "视频文件请放这里.txt",
                video_dir / "视频文件请放这里.txt",
            )

        private_dir = package_root / "04_报名与提交信息"
        copy_required(
            registration_form,
            private_dir / "纳米颗粒图像识别工具开发小组_报名表.pdf",
        )
        for template in ("邮件主题与正文.txt", "提交前检查清单.txt"):
            copy_required(PACKAGE_TEMPLATES / template, private_dir / template)

        write_checksums(package_root)
        zip_tree(package_root, final_zip)

    if offline_image is not None:
        if not offline_image.is_file():
            raise FileNotFoundError(f"找不到离线镜像：{offline_image}")
        separate_offline = OUTPUT_DIR / "NanoLoop-Docker-linux-amd64.tar.gz"
        shutil.copy2(offline_image, separate_offline)

    summary_files = [final_zip]
    if separate_offline is not None:
        summary_files.append(separate_offline)
    (OUTPUT_DIR / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in summary_files),
        encoding="utf-8",
    )
    return final_zip, separate_offline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration-form",
        required=True,
        type=Path,
        help="报名表 PDF 的本机路径；只复制进最终私有 ZIP。",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="最终 MP4；未提供时保留录制占位说明。",
    )
    parser.add_argument(
        "--offline-image",
        type=Path,
        help="可选 linux/amd64 离线镜像；作为 ZIP 旁的独立文件交付。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final_zip, offline = build(
        registration_form=args.registration_form.expanduser().resolve(),
        video=args.video.expanduser().resolve() if args.video else None,
        offline_image=args.offline_image.expanduser().resolve()
        if args.offline_image
        else None,
    )
    print(f"主提交包：{final_zip} ({final_zip.stat().st_size:,} bytes)")
    if offline is not None:
        print(f"离线镜像：{offline} ({offline.stat().st_size:,} bytes)")
    print(f"外层校验：{OUTPUT_DIR / 'SHA256SUMS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
