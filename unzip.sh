#!/bin/bash

# 脚本：解压目录下所有 .tar 文件并删除原文件（支持并行）
# 用法: ./unzip.sh [目录路径] [并发数]
# - 目录路径：不提供则为当前目录
# - 并发数：可选，默认使用 CPU 核心数（无法获取时默认 4）

# 获取目标目录，默认为当前目录
TARGET_DIR="${1:-.}"

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录 '$TARGET_DIR' 不存在"
    exit 1
fi

# 切换到目标目录
cd "$TARGET_DIR" || exit 1

echo "开始处理目录: $(pwd)"
echo "查找 .tar 文件..."

# 统计文件数量
TAR_COUNT=$(find . -maxdepth 1 -type f -name "*.tar" | wc -l)

if [ "$TAR_COUNT" -eq 0 ]; then
    echo "未找到 .tar 文件"
    exit 0
fi

echo "找到 $TAR_COUNT 个 .tar 文件"
echo "准备并行解压..."

# 并发数设置
if command -v nproc >/dev/null 2>&1; then
    DEFAULT_JOBS=$(nproc)
else
    DEFAULT_JOBS=4
fi
PARALLEL_JOBS="${2:-$DEFAULT_JOBS}"
if ! [[ "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$PARALLEL_JOBS" -le 0 ]; then
    echo "无效并发数 '$PARALLEL_JOBS'，使用默认 $DEFAULT_JOBS"
    PARALLEL_JOBS="$DEFAULT_JOBS"
fi
echo "并发数: $PARALLEL_JOBS"

# 结果统计临时文件
RESULT_FILE="$(mktemp -t unzip_tar_result.XXXXXX)"

echo "开始解压（并行执行）..."

# 使用 xargs 并行执行：解压成功则删除原文件，并输出结果到 RESULT_FILE
find . -maxdepth 1 -type f -name "*.tar" -print0 | \
    xargs -0 -I{} -P "$PARALLEL_JOBS" bash -c '
        set -euo pipefail
        f="{}"
        # 去掉前导的 ./ 以便输出更简洁
        clean_f="${f#./}"
        echo "正在解压: ${clean_f}"
        if tar -xf "$f"; then
            if rm -f "$f"; then
                echo "SUCCESS ${clean_f}"
            else
                echo "FAIL_DEL ${clean_f}"
            fi
        else
            echo "FAIL_EXT ${clean_f}"
        fi
    ' >>"$RESULT_FILE" 2>/dev/null

# 汇总统计
SUCCESS_COUNT=$(grep -c "^SUCCESS " "$RESULT_FILE" 2>/dev/null || true)
FAIL_EXT_COUNT=$(grep -c "^FAIL_EXT " "$RESULT_FILE" 2>/dev/null || true)
FAIL_DEL_COUNT=$(grep -c "^FAIL_DEL " "$RESULT_FILE" 2>/dev/null || true)
FAIL_COUNT=$((FAIL_EXT_COUNT + FAIL_DEL_COUNT))

echo ""
echo "处理完成!"
echo "成功: $SUCCESS_COUNT 个文件"
echo "失败: $FAIL_COUNT 个文件"
if [ "$FAIL_EXT_COUNT" -gt 0 ]; then
    echo "  其中解压失败: $FAIL_EXT_COUNT"
fi
if [ "$FAIL_DEL_COUNT" -gt 0 ]; then
    echo "  其中删除失败: $FAIL_DEL_COUNT"
fi

# 清理
rm -f "$RESULT_FILE"

