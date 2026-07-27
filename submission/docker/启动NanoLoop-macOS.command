#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"${script_dir}/nanoloop-control.sh" start
printf '\n可关闭此窗口；NanoLoop 会继续在 Docker 中运行。\n'
