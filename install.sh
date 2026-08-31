#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly RELEASE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
readonly RELEASE_VERSION="$(tr -d '\r\n' < "$RELEASE_DIR/VERSION")"
readonly SERVICE_NAME="phoena-translator.service"
readonly SERVICE_USER="phoena-translator"
readonly SERVICE_GROUP="phoena-translator"
readonly INSTALL_ROOT="/opt/phoena-translator"
readonly RELEASES_ROOT="$INSTALL_ROOT/releases"
readonly CURRENT_LINK="$INSTALL_ROOT/current"
readonly DATA_ROOT="/var/lib/phoena-translator"
readonly LOG_ROOT="/var/log/phoena-translator"
readonly STATE_ROOT="/var/lib/phoena-translator-installer"
readonly STATE_FILE="$STATE_ROOT/transaction.state"
readonly CONFIG_ROOT="/etc/phoena-translator"
readonly ENV_FILE="$CONFIG_ROOT/translator.env"
readonly UNIT_FILE="/etc/systemd/system/$SERVICE_NAME"
readonly NGINX_AVAILABLE="/etc/nginx/sites-available/phoena-translator.conf"
readonly NGINX_ENABLED="/etc/nginx/sites-enabled/phoena-translator.conf"
readonly NGINX_DEFAULT="/etc/nginx/sites-enabled/default"

MODE="install"
COMMITTED=0
STAGING_ROOT=""
NEW_RELEASE=""
ROLLBACK_ROOT=""
OLD_CURRENT_TARGET="NONE"
CURRENT_EXISTED=0
ENV_EXISTED=0
UNIT_EXISTED=0
NGINX_AVAILABLE_EXISTED=0
NGINX_ENABLED_EXISTED=0
NGINX_DEFAULT_EXISTED=0
SERVICE_WAS_ACTIVE=0
SERVICE_WAS_ENABLED=0
NGINX_WAS_ACTIVE=0
NGINX_WAS_ENABLED=0
BACKUPS_READY=0
SWITCH_STARTED=0
API_KEY=""
API_BASE_URL=""
API_MODEL=""
APT_UPDATED=0
PROBE_PID=""
readonly INSTALLER_PID="$BASHPID"
STATE_COMMITTING=0

log() {
    printf '[PhoenaTranslator] %s\n' "$*"
}

warn() {
    printf '[PhoenaTranslator] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[PhoenaTranslator] ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    printf '%s\n' \
        'Usage:' \
        '  ./install.sh                 Verify and install PhoenaTranslator' \
        '  ./install.sh --verify-only   Verify this release without changing the host' \
        '  ./install.sh --help          Show this help'
}

sync_path() {
    sync -f -- "$1"
}

managed_path_matches() {
    local path="$1"
    local parent="$2"
    local name_regex="$3"
    [[ "$(dirname -- "$path")" == "$parent" ]] || return 1
    [[ "$(basename -- "$path")" =~ $name_regex ]]
}

is_staging_path() {
    managed_path_matches "$1" "$INSTALL_ROOT" '^\.staging\.[A-Za-z0-9]+$'
}

is_release_path() {
    managed_path_matches "$1" "$RELEASES_ROOT" '^release-[A-Za-z0-9._-]+$'
}

is_rollback_path() {
    managed_path_matches "$1" "$STATE_ROOT" '^rollback\.[A-Za-z0-9]+$'
}

safe_remove_tree() {
    local target="$1"
    if ! is_staging_path "$target" \
        && ! is_release_path "$target" \
        && ! is_rollback_path "$target"; then
        warn "refusing unsafe cleanup target: $target"
        return 1
    fi
    [[ "$target" != "$OLD_CURRENT_TARGET" ]] || {
        warn "refusing to remove the previous active release: $target"
        return 1
    }
    rm -rf --one-file-system -- "$target"
}

atomic_install_file() {
    local source="$1"
    local destination="$2"
    local mode="$3"
    local parent temp
    parent="$(dirname -- "$destination")"
    temp="$(mktemp -p "$parent" ".$(basename -- "$destination").new.XXXXXXXX")"
    if ! install -o root -g root -m "$mode" "$source" "$temp"; then
        rm -f -- "$temp"
        return 1
    fi
    sync_path "$temp"
    mv -fT -- "$temp" "$destination"
    sync_path "$parent"
}

atomic_symlink() {
    local target="$1"
    local destination="$2"
    local parent temp
    parent="$(dirname -- "$destination")"
    temp="$parent/.$(basename -- "$destination").link.$$.$RANDOM"
    rm -f -- "$temp"
    ln -s -- "$target" "$temp"
    mv -fT -- "$temp" "$destination"
    sync_path "$parent"
}

remove_managed_file() {
    local destination="$1"
    rm -f -- "$destination"
    sync_path "$(dirname -- "$destination")"
}

backup_managed_path() {
    local source="$1"
    local label="$2"
    local destination="$ROLLBACK_ROOT/$label"
    local temp="$ROLLBACK_ROOT/.${label}.new.$$"
    rm -f -- "$temp" "$destination"
    cp -a -- "$source" "$temp"
    if [[ ! -L "$temp" ]]; then sync_path "$temp"; fi
    mv -fT -- "$temp" "$destination"
    sync_path "$ROLLBACK_ROOT"
}

restore_managed_path() {
    local destination="$1"
    local backup="$2"
    local existed="$3"
    local mode target
    if [[ "$existed" -eq 0 ]]; then
        remove_managed_file "$destination"
        return
    fi
    if [[ -L "$backup" ]]; then
        target="$(readlink -- "$backup")"
        atomic_symlink "$target" "$destination"
    elif [[ -f "$backup" ]]; then
        mode="$(stat -c '%a' -- "$backup")"
        atomic_install_file "$backup" "$destination" "$mode"
    else
        warn "required rollback copy is missing: $backup"
        return 1
    fi
}

validate_flag() {
    [[ "$1" == "0" || "$1" == "1" ]]
}

validate_state_paths() {
    is_staging_path "$REC_STAGING_ROOT" || return 1
    is_release_path "$REC_NEW_RELEASE" || return 1
    is_rollback_path "$REC_ROLLBACK_ROOT" || return 1
    [[ "$REC_OLD_CURRENT_TARGET" == "NONE" ]] \
        || is_release_path "$REC_OLD_CURRENT_TARGET" || return 1
    local flag
    for flag in \
        "$REC_CURRENT_EXISTED" "$REC_ENV_EXISTED" "$REC_UNIT_EXISTED" \
        "$REC_NGINX_AVAILABLE_EXISTED" "$REC_NGINX_ENABLED_EXISTED" \
        "$REC_NGINX_DEFAULT_EXISTED" "$REC_SERVICE_WAS_ACTIVE" \
        "$REC_SERVICE_WAS_ENABLED" "$REC_NGINX_WAS_ACTIVE" \
        "$REC_NGINX_WAS_ENABLED" "$REC_BACKUPS_READY" "$REC_SWITCH_STARTED"; do
        validate_flag "$flag" || return 1
    done
    if [[ "$REC_CURRENT_EXISTED" -eq 1 ]]; then
        [[ "$REC_OLD_CURRENT_TARGET" != "NONE" ]] || return 1
    else
        [[ "$REC_OLD_CURRENT_TARGET" == "NONE" ]] || return 1
    fi
    [[ "$REC_SWITCH_STARTED" -eq 0 || "$REC_BACKUPS_READY" -eq 1 ]] || return 1
    [[ "$REC_NEW_RELEASE" != "$REC_OLD_CURRENT_TARGET" ]] || return 1
}

write_state() {
    local temp
    install -d -o root -g root -m 0700 "$STATE_ROOT"
    temp="$(mktemp -p "$STATE_ROOT" .transaction.state.XXXXXXXX)"
    chmod 0600 "$temp"
    {
        printf 'STATE_VERSION=1\n'
        printf 'STAGING_ROOT=%s\n' "$STAGING_ROOT"
        printf 'NEW_RELEASE=%s\n' "$NEW_RELEASE"
        printf 'ROLLBACK_ROOT=%s\n' "$ROLLBACK_ROOT"
        printf 'OLD_CURRENT_TARGET=%s\n' "$OLD_CURRENT_TARGET"
        printf 'CURRENT_EXISTED=%s\n' "$CURRENT_EXISTED"
        printf 'ENV_EXISTED=%s\n' "$ENV_EXISTED"
        printf 'UNIT_EXISTED=%s\n' "$UNIT_EXISTED"
        printf 'NGINX_AVAILABLE_EXISTED=%s\n' "$NGINX_AVAILABLE_EXISTED"
        printf 'NGINX_ENABLED_EXISTED=%s\n' "$NGINX_ENABLED_EXISTED"
        printf 'NGINX_DEFAULT_EXISTED=%s\n' "$NGINX_DEFAULT_EXISTED"
        printf 'SERVICE_WAS_ACTIVE=%s\n' "$SERVICE_WAS_ACTIVE"
        printf 'SERVICE_WAS_ENABLED=%s\n' "$SERVICE_WAS_ENABLED"
        printf 'NGINX_WAS_ACTIVE=%s\n' "$NGINX_WAS_ACTIVE"
        printf 'NGINX_WAS_ENABLED=%s\n' "$NGINX_WAS_ENABLED"
        printf 'BACKUPS_READY=%s\n' "$BACKUPS_READY"
        printf 'SWITCH_STARTED=%s\n' "$SWITCH_STARTED"
    } > "$temp"
    sync_path "$temp"
    mv -fT -- "$temp" "$STATE_FILE"
    sync_path "$STATE_ROOT"
}

load_state() {
    REC_STATE_VERSION=""
    REC_STAGING_ROOT=""
    REC_NEW_RELEASE=""
    REC_ROLLBACK_ROOT=""
    REC_OLD_CURRENT_TARGET=""
    REC_CURRENT_EXISTED=""
    REC_ENV_EXISTED=""
    REC_UNIT_EXISTED=""
    REC_NGINX_AVAILABLE_EXISTED=""
    REC_NGINX_ENABLED_EXISTED=""
    REC_NGINX_DEFAULT_EXISTED=""
    REC_SERVICE_WAS_ACTIVE=""
    REC_SERVICE_WAS_ENABLED=""
    REC_NGINX_WAS_ACTIVE=""
    REC_NGINX_WAS_ENABLED=""
    REC_BACKUPS_READY=""
    REC_SWITCH_STARTED=""
    local key value
    while IFS='=' read -r key value; do
        case "$key" in
            STATE_VERSION) REC_STATE_VERSION="$value" ;;
            STAGING_ROOT) REC_STAGING_ROOT="$value" ;;
            NEW_RELEASE) REC_NEW_RELEASE="$value" ;;
            ROLLBACK_ROOT) REC_ROLLBACK_ROOT="$value" ;;
            OLD_CURRENT_TARGET) REC_OLD_CURRENT_TARGET="$value" ;;
            CURRENT_EXISTED) REC_CURRENT_EXISTED="$value" ;;
            ENV_EXISTED) REC_ENV_EXISTED="$value" ;;
            UNIT_EXISTED) REC_UNIT_EXISTED="$value" ;;
            NGINX_AVAILABLE_EXISTED) REC_NGINX_AVAILABLE_EXISTED="$value" ;;
            NGINX_ENABLED_EXISTED) REC_NGINX_ENABLED_EXISTED="$value" ;;
            NGINX_DEFAULT_EXISTED) REC_NGINX_DEFAULT_EXISTED="$value" ;;
            SERVICE_WAS_ACTIVE) REC_SERVICE_WAS_ACTIVE="$value" ;;
            SERVICE_WAS_ENABLED) REC_SERVICE_WAS_ENABLED="$value" ;;
            NGINX_WAS_ACTIVE) REC_NGINX_WAS_ACTIVE="$value" ;;
            NGINX_WAS_ENABLED) REC_NGINX_WAS_ENABLED="$value" ;;
            BACKUPS_READY) REC_BACKUPS_READY="$value" ;;
            SWITCH_STARTED) REC_SWITCH_STARTED="$value" ;;
            *) warn "unknown transaction-state key: $key"; return 1 ;;
        esac
    done < "$STATE_FILE"
    [[ "$REC_STATE_VERSION" == "1" ]] || return 1
    validate_state_paths
}

clear_state() {
    rm -f -- "$STATE_FILE"
    sync_path "$STATE_ROOT"
}

ensure_unit_stopped() {
    local unit="$1"
    local _attempt load_state active_state
    systemctl stop "$unit" >/dev/null 2>&1 || true
    for _attempt in $(seq 1 20); do
        load_state="$(read_unit_property "$unit" LoadState)" || return 1
        [[ "$load_state" == "not-found" ]] && return 0
        active_state="$(read_unit_property "$unit" ActiveState)" || return 1
        case "$active_state" in
            inactive|failed) return 0 ;;
            deactivating) sleep 0.1 ;;
            *) return 1 ;;
        esac
    done
    return 1
}

read_unit_property() {
    local unit="$1"
    local property="$2"
    local value status
    if value="$(systemctl show "$unit" --property="$property" --value 2>/dev/null)"; then
        status=0
    else
        status=$?
    fi
    [[ -n "$value" ]] || return 1
    if [[ "$status" -ne 0 ]]; then
        [[ "$property" == "LoadState" && "$value" == "not-found" ]] || return 1
    fi
    printf '%s' "$value"
}

read_unit_enabled_state() {
    local unit="$1"
    local value
    value="$(systemctl is-enabled "$unit" 2>/dev/null || true)"
    [[ -n "$value" ]] || return 1
    printf '%s' "$value"
}

enabled_state_matches() {
    local actual="$1"
    local wanted="$2"
    if [[ "$wanted" -eq 1 ]]; then
        case "$actual" in
            enabled|enabled-runtime|linked|linked-runtime|alias|generated|transient) return 0 ;;
            *) return 1 ;;
        esac
    fi
    case "$actual" in
        disabled|static|indirect|masked|masked-runtime|not-found) return 0 ;;
        *) return 1 ;;
    esac
}

capture_unit_active_flag() {
    local unit="$1"
    local load_state active_state
    load_state="$(read_unit_property "$unit" LoadState)" \
        || die "could not query systemd load state for $unit"
    [[ "$load_state" == "not-found" ]] && { printf '0'; return; }
    active_state="$(read_unit_property "$unit" ActiveState)" \
        || die "could not query systemd active state for $unit"
    case "$active_state" in
        active) printf '1' ;;
        inactive|failed) printf '0' ;;
        *) die "$unit is in an unstable systemd state: $active_state" ;;
    esac
}

capture_unit_enabled_flag() {
    local unit="$1"
    local enabled_state
    enabled_state="$(read_unit_enabled_state "$unit")" \
        || die "could not query systemd enable state for $unit"
    if enabled_state_matches "$enabled_state" 1; then
        printf '1'
    elif enabled_state_matches "$enabled_state" 0; then
        printf '0'
    else
        die "$unit has an unsupported enable state: $enabled_state"
    fi
}

restore_enable_state() {
    local unit="$1"
    local was_enabled="$2"
    if [[ "$was_enabled" -eq 1 ]]; then
        systemctl enable "$unit" >/dev/null 2>&1 || true
    else
        systemctl disable "$unit" >/dev/null 2>&1 || true
    fi
    local actual
    actual="$(read_unit_enabled_state "$unit")" || return 1
    enabled_state_matches "$actual" "$was_enabled"
}

restore_active_state() {
    local unit="$1"
    local was_active="$2"
    local force_restart="${3:-0}"
    if [[ "$was_active" -eq 1 ]]; then
        local _attempt load_state active_state
        load_state="$(read_unit_property "$unit" LoadState)" || return 1
        [[ "$load_state" != "not-found" ]] || return 1
        active_state="$(read_unit_property "$unit" ActiveState)" || return 1
        if [[ "$active_state" == "active" && "$force_restart" -eq 0 ]]; then
            return 0
        fi
        systemctl restart "$unit" >/dev/null 2>&1 || true
        for _attempt in $(seq 1 20); do
            load_state="$(read_unit_property "$unit" LoadState)" || return 1
            [[ "$load_state" != "not-found" ]] || return 1
            active_state="$(read_unit_property "$unit" ActiveState)" || return 1
            case "$active_state" in
                active) return 0 ;;
                activating) sleep 0.1 ;;
                *) return 1 ;;
            esac
        done
        return 1
    else
        ensure_unit_stopped "$unit"
    fi
}

ensure_service_account_quiescent() {
    local service_uid
    ensure_unit_stopped "$SERVICE_NAME" || return 1
    service_uid="$(id -u "$SERVICE_USER" 2>/dev/null)" || {
        warn "service account is missing during transaction recovery: $SERVICE_USER"
        return 1
    }
    python3 - "$service_uid" <<'PY'
import pathlib
import sys

expected_uid = int(sys.argv[1])
live = []
for status_path in pathlib.Path("/proc").glob("[0-9]*/status"):
    try:
        fields = {}
        for line in status_path.read_text(errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        real_uid = int(fields.get("Uid", "-1").split()[0])
        state = fields.get("State", "?").split()[0]
        if real_uid == expected_uid and state not in {"Z", "X"}:
            live.append(status_path.parent.name)
    except (OSError, ValueError, IndexError):
        continue
if live:
    print("live service-account processes remain: " + ",".join(sorted(live)), file=sys.stderr)
    raise SystemExit(1)
PY
}

stop_precommit_probe() {
    [[ -n "$PROBE_PID" ]] || return 0
    if kill -0 "$PROBE_PID" >/dev/null 2>&1; then
        kill "$PROBE_PID" >/dev/null 2>&1 || true
        for _probe_stop_attempt in $(seq 1 20); do
            kill -0 "$PROBE_PID" >/dev/null 2>&1 || break
            sleep 0.1
        done
        if kill -0 "$PROBE_PID" >/dev/null 2>&1; then
            kill -KILL "$PROBE_PID" >/dev/null 2>&1 || true
        fi
    fi
    wait "$PROBE_PID" >/dev/null 2>&1 || true
    PROBE_PID=""
}

run_precommit_probe() {
    local probe_log="$ROLLBACK_ROOT/precommit-probe.log"
    local service_uid service_gid healthy=0
    service_uid="$(id -u "$SERVICE_USER")"
    service_gid="$(id -g "$SERVICE_USER")"
    install -o root -g root -m 0600 /dev/null "$probe_log"

    # This isolated WSGI probe listens only on an alternate loopback port.  It
    # is not connected to nginx and cannot receive public translation tasks.
    # setpriv drops privileges and asks the kernel to kill the single process
    # if the installer dies unexpectedly.
    (
        cd "$CURRENT_LINK/app"
        exec setpriv \
            --reuid "$service_uid" --regid "$service_gid" --init-groups \
            --nnp --inh-caps=-all --ambient-caps=-all --bounding-set=-all \
            --pdeathsig KILL -- \
            env -i \
                HOME="$DATA_ROOT" \
                PATH="$CURRENT_LINK/venv/bin:/usr/bin:/bin" \
                PYTHONDONTWRITEBYTECODE=1 \
                PYTHONUNBUFFERED=1 \
                PHOENA_INSTALLER_PID="$INSTALLER_PID" \
                DEEPSEEK_API_KEY="" \
                DEEPSEEK_BASE_URL="$API_BASE_URL" \
                DEEPSEEK_MODEL="$API_MODEL" \
                TRANSLATOR_DISABLE_AUTO_RESUME=1 \
                TRANSLATOR_UPLOAD_DIR="$DATA_ROOT/uploads" \
                TRANSLATOR_OUTPUT_DIR="$DATA_ROOT/outputs" \
                TRANSLATOR_PROGRESS_DIR="$DATA_ROOT/progress" \
                TRANSLATOR_LOG_FILE=/dev/null \
                "$CURRENT_LINK/venv/bin/python" -c \
                'import ctypes, os, signal, sys; expected = int(os.environ.pop("PHOENA_INSTALLER_PID")); ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) == 0 or sys.exit(71); os.getppid() == expected or sys.exit(70); from wsgiref.simple_server import make_server; from phoena_translator.application import app; make_server("127.0.0.1", 18501, app).serve_forever()'
    ) >>"$probe_log" 2>&1 &
    PROBE_PID=$!

    for _probe_attempt in $(seq 1 30); do
        if kill -0 "$PROBE_PID" >/dev/null 2>&1 \
            && curl --noproxy '*' --fail --silent --show-error --max-time 2 \
                http://127.0.0.1:18501/api/health \
                | grep -Eq '"ok"[[:space:]]*:[[:space:]]*true'; then
            healthy=1
            break
        fi
        kill -0 "$PROBE_PID" >/dev/null 2>&1 || break
        sleep 1
    done
    stop_precommit_probe
    [[ "$healthy" -eq 1 ]] || {
        warn "isolated pre-commit runtime probe failed; details: $probe_log"
        return 1
    }
}

rollback_after_activation_failure() {
    local reason="$1"
    ensure_unit_stopped "$SERVICE_NAME" \
        || warn "could not confirm $SERVICE_NAME is stopped after activation failure"
    ensure_unit_stopped nginx.service \
        || warn "could not confirm nginx.service is stopped after activation failure"
    if write_state; then
        COMMITTED=0
        die "$reason; starting automatic rollback"
    fi
    die "$reason; the switched files are internally consistent, and the root-only rollback copy was retained at $ROLLBACK_ROOT"
}

recover_transaction() {
    (
    set +e
    [[ -f "$STATE_FILE" ]] || return 0
    log "Recovering an interrupted installer transaction..."
    if ! load_state; then
        warn "transaction state is invalid; refusing automatic recovery: $STATE_FILE"
        return 1
    fi

    if ! ensure_service_account_quiescent; then
        ensure_unit_stopped nginx.service \
            || warn "could not confirm nginx.service is stopped"
        warn "service-account quiescence is unproven; transaction state retained at $STATE_FILE"
        return 1
    fi

    local failures=0
    local activation_failures=0
    local saved_old="$OLD_CURRENT_TARGET"
    OLD_CURRENT_TARGET="$REC_OLD_CURRENT_TARGET"
    if [[ "$REC_SWITCH_STARTED" -eq 1 ]]; then
        if [[ "$REC_CURRENT_EXISTED" -eq 1 ]]; then
            if [[ -d "$REC_OLD_CURRENT_TARGET" ]]; then
                atomic_symlink "$REC_OLD_CURRENT_TARGET" "$CURRENT_LINK" || failures=$((failures + 1))
            else
                warn "previous release is missing: $REC_OLD_CURRENT_TARGET"
                failures=$((failures + 1))
            fi
        else
            remove_managed_file "$CURRENT_LINK" || failures=$((failures + 1))
        fi

        if [[ "$REC_BACKUPS_READY" -eq 1 ]]; then
            restore_managed_path "$ENV_FILE" "$REC_ROLLBACK_ROOT/translator.env" "$REC_ENV_EXISTED" || failures=$((failures + 1))
            restore_managed_path "$NGINX_AVAILABLE" "$REC_ROLLBACK_ROOT/phoena-translator.nginx.conf" "$REC_NGINX_AVAILABLE_EXISTED" || failures=$((failures + 1))
            restore_managed_path "$NGINX_ENABLED" "$REC_ROLLBACK_ROOT/phoena-translator.nginx.enabled" "$REC_NGINX_ENABLED_EXISTED" || failures=$((failures + 1))
            restore_managed_path "$NGINX_DEFAULT" "$REC_ROLLBACK_ROOT/nginx.default" "$REC_NGINX_DEFAULT_EXISTED" || failures=$((failures + 1))
            # Restore a possibly legacy unit without the marker condition only
            # after every other managed path and current have been restored.
            if [[ "$failures" -eq 0 ]]; then
                restore_managed_path "$UNIT_FILE" "$REC_ROLLBACK_ROOT/$SERVICE_NAME" "$REC_UNIT_EXISTED" \
                    || failures=$((failures + 1))
            fi
        fi

        if [[ "$failures" -eq 0 ]]; then
            systemctl daemon-reload >/dev/null 2>&1 || failures=$((failures + 1))
        fi
        if [[ "$failures" -eq 0 ]]; then
            restore_enable_state "$SERVICE_NAME" "$REC_SERVICE_WAS_ENABLED" \
                || failures=$((failures + 1))
        fi
        if [[ "$failures" -eq 0 && -x "$(command -v nginx 2>/dev/null || true)" ]]; then
            nginx -t >/dev/null 2>&1 || failures=$((failures + 1))
        fi
        if [[ "$failures" -eq 0 ]]; then
            restore_enable_state nginx.service "$REC_NGINX_WAS_ENABLED" \
                || failures=$((failures + 1))
        fi

        # Do not start either service while transaction.state exists.  The
        # unit-level marker gate also enforces this across a machine reboot.
    else
        # Preparation may have installed nginx or a marker-gated existing
        # translator may have failed to restart.  Restore the exact recorded
        # enable semantics before clearing the marker even though no program
        # or configuration switch occurred.
        restore_enable_state "$SERVICE_NAME" "$REC_SERVICE_WAS_ENABLED" \
            || failures=$((failures + 1))
        if [[ "$failures" -eq 0 ]]; then
            restore_enable_state nginx.service "$REC_NGINX_WAS_ENABLED" \
                || failures=$((failures + 1))
        fi
    fi

    if [[ "$failures" -eq 0 ]]; then
        [[ ! -e "$REC_STAGING_ROOT" ]] || safe_remove_tree "$REC_STAGING_ROOT" || failures=$((failures + 1))
        [[ ! -e "$REC_NEW_RELEASE" ]] || safe_remove_tree "$REC_NEW_RELEASE" || failures=$((failures + 1))
    fi
    if [[ "$failures" -eq 0 ]]; then
        clear_state || failures=$((failures + 1))
    fi
    OLD_CURRENT_TARGET="$saved_old"

    if [[ "$failures" -ne 0 ]]; then
        ensure_unit_stopped "$SERVICE_NAME" \
            || warn "could not confirm $SERVICE_NAME is stopped"
        ensure_unit_stopped nginx.service \
            || warn "could not confirm nginx.service is stopped"
        warn "automatic recovery is incomplete; state retained at $STATE_FILE"
        return 1
    fi

    # File/config recovery is now durably committed and the marker gate is
    # open.  Restore runtime state only after this point.  A start failure can
    # no longer expose a mixed version; both services are stopped and the
    # already-restored files remain authoritative.
    restore_active_state nginx.service "$REC_NGINX_WAS_ACTIVE" "$REC_SWITCH_STARTED" \
        || activation_failures=$((activation_failures + 1))
    if [[ "$activation_failures" -eq 0 ]]; then
        restore_active_state "$SERVICE_NAME" "$REC_SERVICE_WAS_ACTIVE" "$REC_SWITCH_STARTED" \
            || activation_failures=$((activation_failures + 1))
    fi
    if [[ "$activation_failures" -ne 0 ]]; then
        ensure_unit_stopped "$SERVICE_NAME" \
            || warn "could not confirm $SERVICE_NAME is stopped"
        ensure_unit_stopped nginx.service \
            || warn "could not confirm nginx.service is stopped"
        warn "files were safely recovered, but service activation failed; rollback copy retained at $REC_ROLLBACK_ROOT"
        return 1
    fi

    # The rollback copy is the recovery authority, so keep it until after the
    # durable state marker has been removed.  Cleanup after that commit point
    # is best-effort; a later installer run also removes orphan rollback dirs.
    if [[ -e "$REC_ROLLBACK_ROOT" ]]; then
        safe_remove_tree "$REC_ROLLBACK_ROOT" \
            || warn "recovery succeeded, but root-only rollback cleanup remains: $REC_ROLLBACK_ROOT"
    fi
    log "Interrupted transaction recovered."
    )
}

cleanup_on_exit() {
    local status=$?
    trap - EXIT INT TERM
    stop_precommit_probe || true
    unset API_KEY
    if [[ "$COMMITTED" -eq 0 && -f "$STATE_FILE" ]]; then
        recover_transaction || true
    elif [[ "$COMMITTED" -eq 0 && "$STATE_COMMITTING" -eq 1 ]]; then
        warn "the durable marker was removed at the commit point; leaving the consistent release and root-only rollback copy in place"
    elif [[ "$COMMITTED" -eq 0 ]]; then
        if [[ -n "$STAGING_ROOT" && -e "$STAGING_ROOT" ]]; then safe_remove_tree "$STAGING_ROOT" || true; fi
        if [[ -n "$ROLLBACK_ROOT" && -e "$ROLLBACK_ROOT" ]]; then safe_remove_tree "$ROLLBACK_ROOT" || true; fi
    fi
    exit "$status"
}

verify_checksums() {
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
    log "Verifying package checksums..."
    (cd "$RELEASE_DIR" && sha256sum --quiet -c MANIFEST.sha256) \
        || die "package checksum verification failed"
}

verify_release_topology() {
    command -v python3 >/dev/null 2>&1 || die "python3 is required to verify release topology"
    log "Verifying package topology, exclusions and Python syntax..."
    PYTHONDONTWRITEBYTECODE=1 python3 "$RELEASE_DIR/verify_release.py" "$RELEASE_DIR"
}

case "${1:-}" in
    "") ;;
    --verify-only) MODE="verify" ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }
[[ "$RELEASE_VERSION" =~ ^[0-9A-Za-z._-]+$ ]] || die "VERSION contains unsafe characters"

verify_checksums
if [[ "$MODE" == "verify" ]]; then
    verify_release_topology
    log "Verification completed; no host changes were made."
    exit 0
fi

[[ ${EUID:-$(id -u)} -eq 0 ]] || die "run installation as root: sudo ./install.sh"
[[ -r /dev/tty && -w /dev/tty ]] || die "installation needs an interactive controlling terminal"
[[ "$(uname -m)" == "x86_64" ]] || die "only x86_64 is supported by this release"
[[ -r /etc/os-release ]] || die "/etc/os-release is missing"

# shellcheck disable=SC1091
source /etc/os-release
case "${ID:-}:${VERSION_ID:-}" in
    debian:12|debian:13|ubuntu:24.04) ;;
    *) die "supported targets are Debian 12/13 and Ubuntu 24.04; found ${ID:-unknown} ${VERSION_ID:-unknown}" ;;
esac

[[ ! -L "$STATE_ROOT" ]] || die "refusing installer-state symlink: $STATE_ROOT"
[[ ! -e "$STATE_ROOT" || -d "$STATE_ROOT" ]] \
    || die "installer state path is not a directory: $STATE_ROOT"
if [[ -d "$STATE_ROOT" ]]; then
    [[ "$(stat -c '%u:%g' -- "$STATE_ROOT")" == "0:0" ]] \
        || die "installer state directory must be root-owned: $STATE_ROOT"
    [[ $((8#$(stat -c '%a' -- "$STATE_ROOT") & 8#022)) -eq 0 ]] \
        || die "installer state directory must not be group/world-writable: $STATE_ROOT"
fi
install -d -o root -g root -m 0700 "$STATE_ROOT"
# Lock the already validated root-only directory itself.  This avoids opening
# a predictable pathname in a shared sticky directory and its symlink race.
exec 9<"$STATE_ROOT"
flock -n 9 || die "another PhoenaTranslator installer is already running"

trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

install -d -o root -g root -m 0700 "$STATE_ROOT"
recover_transaction || die "previous interrupted transaction requires manual recovery: $STATE_FILE"
while IFS= read -r -d '' orphan_rollback; do
    safe_remove_tree "$orphan_rollback" || die "could not remove stale transaction directory: $orphan_rollback"
done < <(find "$STATE_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'rollback.*' -print0)

if ! command -v python3 >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    log "Bootstrapping Python for full package verification..."
    apt-get update
    APT_UPDATED=1
    apt-get install -y --no-install-recommends python3
    unset DEBIAN_FRONTEND
fi
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
    || die "Python 3.11 or newer is required"
verify_release_topology

SERVICE_WAS_ACTIVE="$(capture_unit_active_flag "$SERVICE_NAME")"
SERVICE_WAS_ENABLED="$(capture_unit_enabled_flag "$SERVICE_NAME")"
NGINX_WAS_ACTIVE="$(capture_unit_active_flag nginx.service)"
NGINX_WAS_ENABLED="$(capture_unit_enabled_flag nginx.service)"

export DEBIAN_FRONTEND=noninteractive
log "Installing operating-system dependencies..."
if [[ "$APT_UPDATED" -eq 0 ]]; then apt-get update; fi
apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    fonts-noto-cjk \
    libgcc-s1 \
    libstdc++6 \
    nginx \
    python3 \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    tesseract-ocr-eng \
    util-linux
unset DEBIAN_FRONTEND

require_directory_or_absent() {
    local path="$1"
    [[ ! -L "$path" ]] || die "refusing directory symlink: $path"
    [[ ! -e "$path" || -d "$path" ]] || die "expected a directory or absent path: $path"
}

require_regular_or_absent() {
    local path="$1"
    [[ ! -L "$path" ]] || die "refusing file symlink: $path"
    [[ ! -e "$path" || -f "$path" ]] || die "expected a regular file or absent path: $path"
    if [[ -e "$path" ]]; then
        [[ "$(stat -c '%u:%g' -- "$path")" == "0:0" ]] || die "managed file must be root-owned: $path"
        [[ $((8#$(stat -c '%a' -- "$path") & 8#022)) -eq 0 ]] || die "managed file must not be group/world-writable: $path"
    fi
}

for directory in \
    "$INSTALL_ROOT" "$RELEASES_ROOT" "$DATA_ROOT" "$DATA_ROOT/uploads" \
    "$DATA_ROOT/outputs" "$DATA_ROOT/progress" "$LOG_ROOT" "$STATE_ROOT" "$CONFIG_ROOT"; do
    require_directory_or_absent "$directory"
done
for legacy_path in "$INSTALL_ROOT/app" "$INSTALL_ROOT/venv" "$INSTALL_ROOT/www"; do
    [[ ! -e "$legacy_path" && ! -L "$legacy_path" ]] \
        || die "unsupported legacy install path exists: $legacy_path"
done
for managed_file in "$ENV_FILE" "$UNIT_FILE" "$NGINX_AVAILABLE"; do
    require_regular_or_absent "$managed_file"
done
for managed_link in "$NGINX_ENABLED" "$NGINX_DEFAULT"; do
    [[ ! -e "$managed_link" && ! -L "$managed_link" ]] || {
        [[ -f "$managed_link" || -L "$managed_link" ]] \
            || die "expected a regular file/symlink: $managed_link"
        [[ "$(stat -c '%u:%g' -- "$managed_link")" == "0:0" ]] \
            || die "managed nginx entry must be root-owned: $managed_link"
    }
done

if [[ -e "$NGINX_DEFAULT" || -L "$NGINX_DEFAULT" ]]; then
    default_target="$(readlink -f -- "$NGINX_DEFAULT" 2>/dev/null || true)"
    [[ "$default_target" == "/etc/nginx/sites-available/default" ]] \
        || die "refusing to replace a non-standard nginx default site: $NGINX_DEFAULT"
fi

if [[ -e "$CURRENT_LINK" || -L "$CURRENT_LINK" ]]; then
    [[ -L "$CURRENT_LINK" ]] || die "$CURRENT_LINK must be a managed symlink"
    [[ "$(stat -c '%u:%g' -- "$CURRENT_LINK")" == "0:0" ]] \
        || die "$CURRENT_LINK must be root-owned"
    OLD_CURRENT_TARGET="$(readlink -f -- "$CURRENT_LINK" 2>/dev/null || true)"
    is_release_path "$OLD_CURRENT_TARGET" && [[ -d "$OLD_CURRENT_TARGET" ]] \
        || die "$CURRENT_LINK points outside the managed release directory"
    CURRENT_EXISTED=1
fi

read_secret() {
    printf 'Translation LLM API Key: ' >/dev/tty
    IFS= read -r -s API_KEY </dev/tty || die "could not read API Key"
    printf '\n' >/dev/tty
}

read_value() {
    local prompt="$1"
    local default_value="$2"
    local result
    printf '%s [%s]: ' "$prompt" "$default_value" >/dev/tty
    IFS= read -r result </dev/tty || die "could not read configuration"
    [[ -n "$result" ]] || result="$default_value"
    printf '%s' "$result"
}

safe_env_value() {
    local value="$1"
    [[ -n "$value" ]] || return 1
    [[ "$value" != *[[:space:]]* ]] || return 1
    [[ "$value" != *"'"* ]] || return 1
    [[ "$value" != *'"'* ]] || return 1
    [[ "$value" != *\\* ]] || return 1
    [[ "$value" != *$'\r'* && "$value" != *$'\n'* ]] || return 1
}

read_secret
API_BASE_URL="$(read_value "OpenAI-compatible API Base URL" "https://api.deepseek.com")"
API_MODEL="$(read_value "Model name" "deepseek-v4-flash")"
safe_env_value "$API_KEY" || die "API Key must be a non-empty single token without quotes or backslashes"
safe_env_value "$API_BASE_URL" || die "API Base URL contains unsafe whitespace or quoting"
safe_env_value "$API_MODEL" || die "model name contains unsafe whitespace or quoting"
[[ "$API_BASE_URL" =~ ^https?://[^[:space:]]+$ ]] || die "API Base URL must begin with http:// or https://"
log "Configuration accepted: API Base URL and model '$API_MODEL' (API Key hidden)"

if getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    service_gid="$(getent group "$SERVICE_GROUP" | awk -F: '{print $3}')"
    [[ "$service_gid" =~ ^[0-9]+$ && "$service_gid" -gt 0 && "$service_gid" -lt 1000 ]] \
        || die "existing $SERVICE_GROUP group is not an unprivileged system group"
else
    groupadd --system "$SERVICE_GROUP"
fi
if id "$SERVICE_USER" >/dev/null 2>&1; then
    service_uid="$(id -u "$SERVICE_USER")"
    [[ "$service_uid" -gt 0 && "$service_uid" -lt 1000 ]] \
        || die "existing $SERVICE_USER account is not an unprivileged system account"
    [[ "$(id -gn "$SERVICE_USER")" == "$SERVICE_GROUP" ]] \
        || die "existing $SERVICE_USER has an unexpected primary group"
    account_record="$(getent passwd "$SERVICE_USER")"
    [[ "$(awk -F: '{print $6}' <<< "$account_record")" == "$DATA_ROOT" ]] \
        || die "existing $SERVICE_USER has an unexpected home directory"
    account_shell="$(awk -F: '{print $7}' <<< "$account_record")"
    [[ "$account_shell" == "/usr/sbin/nologin" || "$account_shell" == "/usr/bin/nologin" ]] \
        || die "existing $SERVICE_USER has an unexpected login shell"
    [[ "$(id -nG "$SERVICE_USER" | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/ $//')" == "$SERVICE_GROUP" ]] \
        || die "existing $SERVICE_USER has unexpected supplementary groups"
else
    useradd --system --gid "$SERVICE_GROUP" --home-dir "$DATA_ROOT" \
        --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

# Establish durable recovery before stopping the old service or touching data
# directory ownership.  A failure from this point uses preparing recovery to
# restore the recorded enable/active states.
install -d -o root -g root -m 0755 "$INSTALL_ROOT" "$RELEASES_ROOT"
install -d -o root -g root -m 0750 "$CONFIG_ROOT"
install -d -o root -g root -m 0700 "$STATE_ROOT"
STAGING_ROOT="$(mktemp -d -p "$INSTALL_ROOT" .staging.XXXXXXXX)"
ROLLBACK_ROOT="$(mktemp -d -p "$STATE_ROOT" rollback.XXXXXXXX)"
NEW_RELEASE="$RELEASES_ROOT/release-${RELEASE_VERSION}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
[[ ! -e "$NEW_RELEASE" ]] || die "release destination already exists: $NEW_RELEASE"
chmod 0700 "$STAGING_ROOT" "$ROLLBACK_ROOT"
[[ -e "$ENV_FILE" ]] && ENV_EXISTED=1
[[ -e "$UNIT_FILE" ]] && UNIT_EXISTED=1
[[ -e "$NGINX_AVAILABLE" ]] && NGINX_AVAILABLE_EXISTED=1
[[ -e "$NGINX_ENABLED" || -L "$NGINX_ENABLED" ]] && NGINX_ENABLED_EXISTED=1
[[ -e "$NGINX_DEFAULT" || -L "$NGINX_DEFAULT" ]] && NGINX_DEFAULT_EXISTED=1
write_state

# Stop the only process authorized to write task directories before revoking
# its write access to their parent.
ensure_service_account_quiescent \
    || die "service-account quiescence is required before securing data paths"

secure_managed_directory() {
    local path="$1"
    local owner="$2"
    local group="$3"
    local mode="$4"
    local expected_uid expected_gid
    expected_uid="$(id -u "$owner")"
    expected_gid="$(getent group "$group" | awk -F: '{print $3}')"
    python3 - "$path" "$expected_uid" "$expected_gid" "$mode" <<'PY'
import os
import stat
import sys

path, uid_text, gid_text, mode_text = sys.argv[1:]
uid = int(uid_text)
gid = int(gid_text)
mode = int(mode_text, 8)
parent, name = os.path.split(path.rstrip("/"))
if not parent or not name or not os.path.isabs(path):
    raise SystemExit(f"unsafe managed-directory path: {path}")

directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
parent_fd = os.open(parent, directory_flags)
try:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    fd = os.open(name, directory_flags | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISDIR(before.st_mode):
            raise SystemExit(f"managed path is not a directory: {path}")
        os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
        after = os.fstat(fd)
        if (
            not stat.S_ISDIR(after.st_mode)
            or after.st_uid != uid
            or after.st_gid != gid
            or stat.S_IMODE(after.st_mode) != mode
        ):
            raise SystemExit(f"directory ownership/mode verification failed: {path}")
    finally:
        os.close(fd)
finally:
    os.close(parent_fd)
PY
}

# DATA_ROOT itself is never writable by the service account.  Each child is
# then opened relative to that parent with O_NOFOLLOW and repaired via its fd.
secure_managed_directory "$DATA_ROOT" root "$SERVICE_GROUP" 0750
for writable_dir in "$DATA_ROOT/uploads" "$DATA_ROOT/outputs" "$DATA_ROOT/progress"; do
    secure_managed_directory "$writable_dir" "$SERVICE_USER" "$SERVICE_GROUP" 0750
done
secure_managed_directory "$LOG_ROOT" "$SERVICE_USER" "$SERVICE_GROUP" 0750

mkdir -p "$STAGING_ROOT/app" "$STAGING_ROOT/www"
cp -a -- "$RELEASE_DIR/payload/." "$STAGING_ROOT/app/"
install -o root -g root -m 0644 "$RELEASE_DIR/payload/index.html" "$STAGING_ROOT/www/index.html"
chown -R root:root "$STAGING_ROOT/app" "$STAGING_ROOT/www"
find "$STAGING_ROOT/app" "$STAGING_ROOT/www" -type d -exec chmod 0755 {} +
find "$STAGING_ROOT/app" "$STAGING_ROOT/www" -type f -exec chmod 0644 {} +

log "Creating a fresh Python virtual environment..."
python3 -m venv "$STAGING_ROOT/venv"
"$STAGING_ROOT/venv/bin/python" -m pip install \
    --disable-pip-version-check --no-input --no-cache-dir \
    --constraint "$STAGING_ROOT/app/constraints.txt" \
    --requirement "$STAGING_ROOT/app/requirements.txt"
"$STAGING_ROOT/venv/bin/python" -m pip check
find "$STAGING_ROOT/venv" -type d -exec chmod 0755 {} +
find "$STAGING_ROOT/venv" -type f -perm /111 -exec chmod 0755 {} +
find "$STAGING_ROOT/venv" -type f ! -perm /111 -exec chmod 0644 {} +
chmod 0755 "$STAGING_ROOT" "$STAGING_ROOT/venv"

for writable_dir in "$DATA_ROOT/uploads" "$DATA_ROOT/outputs" "$DATA_ROOT/progress" "$LOG_ROOT"; do
    runuser -u "$SERVICE_USER" -- test -w "$writable_dir" \
        || die "service account cannot write required directory: $writable_dir"
done

log "Running staged runtime smoke test as $SERVICE_USER..."
(
    cd "$STAGING_ROOT/app"
    runuser -u "$SERVICE_USER" -- env -i \
        HOME="$DATA_ROOT" \
        PATH="$STAGING_ROOT/venv/bin:/usr/bin:/bin" \
        PYTHONDONTWRITEBYTECODE=1 \
        DEEPSEEK_API_KEY="" \
        TRANSLATOR_DISABLE_AUTO_RESUME=1 \
        TRANSLATOR_UPLOAD_DIR="$DATA_ROOT/uploads" \
        TRANSLATOR_OUTPUT_DIR="$DATA_ROOT/outputs" \
        TRANSLATOR_PROGRESS_DIR="$DATA_ROOT/progress" \
        TRANSLATOR_LOG_FILE="$LOG_ROOT/translator.log" \
        "$STAGING_ROOT/venv/bin/python" -c \
        'from phoena_translator.application import app; response = app.test_client().get("/api/health"); raise SystemExit(0 if response.status_code == 200 and response.get_json() == {"ok": True} else 1)'
)
# GNU sync -f issues syncfs for the staging filesystem, making all payload and
# virtualenv contents durable before the later directory rename/current swap.
sync_path "$STAGING_ROOT"

ENV_NEW="$ROLLBACK_ROOT/new-translator.env"
{
    printf 'DEEPSEEK_API_KEY=%s\n' "$API_KEY"
    printf 'DEEPSEEK_BASE_URL=%s\n' "$API_BASE_URL"
    printf 'DEEPSEEK_MODEL=%s\n' "$API_MODEL"
    printf 'TRANSLATOR_UPLOAD_DIR=%s\n' "$DATA_ROOT/uploads"
    printf 'TRANSLATOR_OUTPUT_DIR=%s\n' "$DATA_ROOT/outputs"
    printf 'TRANSLATOR_PROGRESS_DIR=%s\n' "$DATA_ROOT/progress"
    printf 'TRANSLATOR_LOG_FILE=%s\n' "$LOG_ROOT/translator.log"
} > "$ENV_NEW"
chmod 0600 "$ENV_NEW"
sync_path "$ENV_NEW"

[[ "$ENV_EXISTED" -eq 0 ]] || backup_managed_path "$ENV_FILE" translator.env
[[ "$UNIT_EXISTED" -eq 0 ]] || backup_managed_path "$UNIT_FILE" "$SERVICE_NAME"
[[ "$NGINX_AVAILABLE_EXISTED" -eq 0 ]] || backup_managed_path "$NGINX_AVAILABLE" phoena-translator.nginx.conf
[[ "$NGINX_ENABLED_EXISTED" -eq 0 ]] || backup_managed_path "$NGINX_ENABLED" phoena-translator.nginx.enabled
[[ "$NGINX_DEFAULT_EXISTED" -eq 0 ]] || backup_managed_path "$NGINX_DEFAULT" nginx.default
BACKUPS_READY=1
write_state

SWITCH_STARTED=1
write_state
atomic_install_file "$RELEASE_DIR/packaging/phoena-translator.service" "$UNIT_FILE" 0644
systemctl daemon-reload
ensure_unit_stopped "$SERVICE_NAME" || die "could not confirm $SERVICE_NAME is stopped before switching"

mv -- "$STAGING_ROOT" "$NEW_RELEASE"
sync_path "$RELEASES_ROOT"
# venv console scripts (gunicorn, pip, ...) embed the staging interpreter
# path in their shebang; repoint them at the final release path after the
# atomic rename, or systemd fails at EXEC with "No such file or directory".
while IFS= read -r -d "" venv_script; do
    sed -i "1s|^#!${STAGING_ROOT}/venv/bin/|#!${NEW_RELEASE}/venv/bin/|" "$venv_script"
done < <(find "$NEW_RELEASE/venv/bin" -maxdepth 1 -type f -print0)
sync_path "$NEW_RELEASE/venv/bin"
atomic_symlink "$NEW_RELEASE" "$CURRENT_LINK"
atomic_install_file "$ENV_NEW" "$ENV_FILE" 0600
atomic_install_file "$RELEASE_DIR/packaging/phoena-translator.nginx.conf" "$NGINX_AVAILABLE" 0644
atomic_symlink "$NGINX_AVAILABLE" "$NGINX_ENABLED"
if [[ -e "$NGINX_DEFAULT" || -L "$NGINX_DEFAULT" ]]; then
    remove_managed_file "$NGINX_DEFAULT"
fi

systemctl daemon-reload
nginx -t
restore_enable_state nginx.service 1 || die "could not confirm nginx.service is enabled"
restore_enable_state "$SERVICE_NAME" 1 || die "could not confirm $SERVICE_NAME is enabled"
ensure_unit_stopped nginx.service || die "could not confirm nginx.service is stopped before the commit check"

log "Running isolated pre-commit runtime and frontend checks..."
run_precommit_probe || die "isolated pre-commit runtime check failed"
grep -F '智译' "$CURRENT_LINK/www/index.html" >/dev/null \
    || die "staged translator frontend is invalid"

# Everything on disk and the isolated runtime are now verified.  Removing the
# durable marker is the single commit point; only after it is synced may the
# production systemd service start and become reachable through nginx.
STATE_COMMITTING=1
clear_state
COMMITTED=1
STATE_COMMITTING=0
if ! restore_active_state "$SERVICE_NAME" 1 1; then
    rollback_after_activation_failure "production service did not start"
fi

log "Waiting for the committed loopback service health check..."
healthy=0
for _attempt in $(seq 1 30); do
    if curl --noproxy '*' --fail --silent --show-error --max-time 3 \
        http://127.0.0.1:8501/api/health \
        | grep -Eq '"ok"[[:space:]]*:[[:space:]]*true'; then
        healthy=1
        break
    fi
    sleep 1
done
[[ "$healthy" -eq 1 ]] \
    || rollback_after_activation_failure "committed health endpoint did not become ready"

restore_active_state nginx.service 1 1 \
    || rollback_after_activation_failure "nginx did not start with the committed configuration"
curl --noproxy '*' --fail --silent --show-error --max-time 5 \
    http://127.0.0.1/api/health \
    | grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' \
    || rollback_after_activation_failure "nginx did not proxy the committed health endpoint"
curl --noproxy '*' --fail --silent --show-error --max-time 5 http://127.0.0.1/ \
    | grep -F '智译' >/dev/null \
    || rollback_after_activation_failure "nginx did not serve the committed translator frontend"
trap - EXIT INT TERM
unset API_KEY
safe_remove_tree "$ROLLBACK_ROOT" || warn "root-only transaction backup remains at $ROLLBACK_ROOT"

log "Installation completed successfully."
log "Open http://<server-ip>/ in a browser."
log "Health: http://127.0.0.1/api/health"
log "Active release: $NEW_RELEASE"
if [[ "$CURRENT_EXISTED" -eq 1 ]]; then
    log "Previous release retained for manual recovery: $OLD_CURRENT_TARGET"
fi
