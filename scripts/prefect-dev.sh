#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${PREFECT_DEV_STATE_DIR:-${ROOT_DIR}/.tmp/prefect-dev}"
SERVER_PID_FILE="${STATE_DIR}/server.pid"
SERVER_LOG_FILE="${STATE_DIR}/server.log"

PREFECT_API_URL="${PREFECT_API_URL:-http://127.0.0.1:4200/api}"
CORTEX_ENV="${CORTEX_ENV:-local}"

mkdir -p "${STATE_DIR}"

is_running() {
    local pid_file="$1"
    [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null
}

server_is_healthy() {
    curl -fsS "${PREFECT_API_URL}/health" >/dev/null 2>&1
}

start_server() {
    if server_is_healthy; then
        echo "Prefect server already reachable at ${PREFECT_API_URL}"
        return
    fi

    if is_running "${SERVER_PID_FILE}"; then
        echo "Prefect server already running: pid $(cat "${SERVER_PID_FILE}")"
        return
    fi

    echo "Starting Prefect server..."
    (
        cd "${ROOT_DIR}"
        prefect server start
    ) >"${SERVER_LOG_FILE}" 2>&1 &
    echo "$!" >"${SERVER_PID_FILE}"
    echo "Prefect server pid: $(cat "${SERVER_PID_FILE}")"
    echo "Prefect server log: ${SERVER_LOG_FILE}"
}

stop_process() {
    local name="$1"
    local pid_file="$2"

    if ! is_running "${pid_file}"; then
        echo "${name} is not running"
        rm -f "${pid_file}"
        return
    fi

    local pid
    pid="$(cat "${pid_file}")"
    echo "Stopping ${name}: pid ${pid}"
    kill "${pid}" 2>/dev/null || true

    for _ in {1..20}; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${pid_file}"
            echo "${name} stopped"
            return
        fi
        sleep 0.25
    done

    echo "${name} did not stop cleanly; sending SIGKILL"
    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${pid_file}"
}

wait_for_server() {
    echo "Waiting for Prefect server at ${PREFECT_API_URL}..."
    for _ in {1..60}; do
        if server_is_healthy; then
            echo "Prefect server is ready"
            return
        fi
        sleep 1
    done

    echo "Prefect server did not become ready. See log: ${SERVER_LOG_FILE}" >&2
    exit 1
}

deploy_models() {
    echo "Creating/updating Prefect deployments..."
    (
        cd "${ROOT_DIR}"
        export PREFECT_API_URL
        export CORTEX_ENV
        python -m app.flows.deployments
    )
}

start_all() {
    echo "Using PREFECT_API_URL=${PREFECT_API_URL}"
    echo "Using CORTEX_ENV=${CORTEX_ENV}"
    start_server
    wait_for_server
    deploy_models
    echo "Prefect server is ready. Start the Prefect worker from VS Code debug."
}

stop_all() {
    stop_process "Prefect server" "${SERVER_PID_FILE}"
}

status() {
    if is_running "${SERVER_PID_FILE}"; then
        echo "Prefect server running: pid $(cat "${SERVER_PID_FILE}")"
    else
        echo "Prefect server stopped"
    fi

    echo "Prefect worker is not managed by this script. Start it from VS Code debug."
}

case "${1:-start}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        start_all
        ;;
    deploy)
        wait_for_server
        deploy_models
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|deploy|status}"
        exit 2
        ;;
esac
