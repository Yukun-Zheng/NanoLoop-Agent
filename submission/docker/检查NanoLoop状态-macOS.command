#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"${script_dir}/nanoloop-control.sh" status
printf '\n按回车键关闭窗口。'
read answer
