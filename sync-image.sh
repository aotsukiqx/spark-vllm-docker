#!/bin/bash
set -e

# =====================================================
# Docker Image Sync Script
# 将镜像同步到集群所有主机
# =====================================================

# 默认值
IMAGE_TAG="vllm-node"
SSH_USER="$USER"
COPY_HOSTS=()
PARALLEL_COPY=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -t, --tag <tag>          : Docker 镜像标签 (default: 'vllm-node')"
    echo "  -c, --copy-to <hosts>   : 目标主机列表 (空格或逗号分隔)"
    echo "      --copy-parallel     : 并行复制到所有主机"
    echo "  -u, --user <user>       : SSH 用户名 (default: \$USER)"
    echo "  -h, --help              : 显示帮助"
    exit 1
}

add_copy_hosts() {
    local token part
    for token in "$@"; do
        IFS=',' read -ra PARTS <<< "$token"
        for part in "${PARTS[@]}"; do
            part="${part//[[:space:]]/}"
            if [ -n "$part" ]; then
                COPY_HOSTS+=("$part")
            fi
        done
    done
}

copy_to_host() {
    local host="$1"
    echo "Loading image into ${SSH_USER}@${host}..."
    local host_copy_start host_copy_end host_copy_time
    host_copy_start=$(date +%s)
    if cat "$TMP_IMAGE" | ssh "${SSH_USER}@${host}" "docker load"; then
        host_copy_end=$(date +%s)
        host_copy_time=$((host_copy_end - host_copy_start))
        printf "Copy to %s completed in %02d:%02d:%02d\n" "$host" $((host_copy_time/3600)) $((host_copy_time%3600/60)) $((host_copy_time%60))
    else
        echo "Copy to $host failed."
        return 1
    fi
}

# 参数解析
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--tag) IMAGE_TAG="$2"; shift ;;
        -c|--copy-to|--copy-to-host)
            shift
            while [[ "$#" -gt 0 && "$1" != -* ]]; do
                add_copy_hosts "$1"
                shift
            done
            continue
            ;;
        --copy-parallel) PARALLEL_COPY=true ;;
        -u|--user) SSH_USER="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# 验证
if [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
    echo "Error: 必须指定目标主机 - 使用 -c <hosts>"
    usage
fi

echo "同步镜像 '$IMAGE_TAG' 到 ${#COPY_HOSTS[@]} 台主机: ${COPY_HOSTS[*]}"

# 保存镜像到临时文件
TMP_IMAGE=$(mktemp -t vllm_image.XXXXXX)
echo "保存镜像到本地: $TMP_IMAGE"
docker save -o "$TMP_IMAGE" "$IMAGE_TAG"

COPY_START=$(date +%s)

# 复制到各主机
if [ "$PARALLEL_COPY" = true ]; then
    echo "并行复制模式"
    PIDS=()
    for host in "${COPY_HOSTS[@]}"; do
        copy_to_host "$host" &
        PIDS+=($!)
    done
    COPY_FAILURE=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            COPY_FAILURE=1
        fi
    done
    if [ "$COPY_FAILURE" -ne 0 ]; then
        echo "部分主机复制失败"
        rm -f "$TMP_IMAGE"
        exit 1
    fi
else
    echo "串行复制模式"
    for host in "${COPY_HOSTS[@]}"; do
        copy_to_host "$host"
    done
fi

COPY_END=$(date +%s)
COPY_TIME=$((COPY_END - COPY_START))

# 清理临时文件
rm -f "$TMP_IMAGE"

echo ""
echo "========================================="
echo "镜像同步完成! 耗时: $(printf '%02d:%02d:%02d' $((COPY_TIME/3600)) $((COPY_TIME%3600/60)) $((COPY_TIME%60)))"
echo "========================================="
