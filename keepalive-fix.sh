#!/usr/bin/env bash
# keepalive-fix.sh v2 —— 让 UOS 服务器长期无人值守运行，不再自动挂起/掉线
# 用法：bash keepalive-fix.sh
PW="${SUDO_PASS:-<YOUR_SUDO_PASSWORD>}"
S() { echo "$PW" | sudo -S "$@" 2>/dev/null; }
U=<SSH_USER>
UIDS=$(id -u $U)
DBUS="unix:path=/run/user/$UIDS/bus"
RT="/run/user/$UIDS"
GS() { sudo -u $U env DBUS_SESSION_BUS_ADDRESS="$DBUS" XDG_RUNTIME_DIR="$RT" gsettings "$@" 2>/dev/null; }

echo "=================== 1) 屏蔽 systemd 休眠/挂起目标（最关键） ==================="
S systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
for t in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
  printf "  %-18s %s\n" "$t" "$(systemctl is-enabled $t 2>/dev/null)"
done

echo
echo "=================== 2) systemd-logind 忽略空闲 ==================="
S cp -n /etc/systemd/logind.conf /etc/systemd/logind.conf.bak.$(date +%F) 2>/dev/null
for kv in "IdleAction=ignore" "IdleActionSec=0" "HandleLidSwitch=ignore" "HandleLidSwitchDocked=ignore" "HandleSuspendKey=ignore" "HandleHibernateKey=ignore"; do
  k="${kv%%=*}"
  if S grep -qE "^#?${k}=" /etc/systemd/logind.conf; then
    S sed -i "s|^#\?${k}=.*|${kv}|" /etc/systemd/logind.conf
  else
    echo "$kv" | S tee -a /etc/systemd/logind.conf >/dev/null
  fi
done
echo "  当前值："
grep -E "^(IdleAction|IdleActionSec|Handle)" /etc/systemd/logind.conf | sed 's/^/    /'
S systemctl restart systemd-logind
echo "  systemd-logind 已重启"

echo
echo "=================== 3) DDE 桌面电源管理 ==================="
echo "  --- 修改前 ---"
GS list-recursively com.deepin.dde.power 2>/dev/null | grep -iE "suspend|sleep|lid" | sed 's/^/    /' || echo "    (读不到)"
for k in line-power-suspend-delay battery-suspend-delay line-power-screen-off-delay battery-screen-off-delay; do
  GS set com.deepin.dde.power "$k" 0 && echo "    已设 $k = 0(从不)"
done
GS set com.deepin.dde.power line-power-lid-is-closed-suspend false
GS set com.deepin.dde.power battery-lid-is-closed-suspend false
echo "  --- 修改后 ---"
GS list-recursively com.deepin.dde.power 2>/dev/null | grep -iE "suspend|sleep|lid" | sed 's/^/    /' || true
# gnome 兼容层（装了才有）
GS set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
GS set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'

echo
echo "=================== 4) 关闭 WiFi 省电（防大流量掉线） ==================="
S iw dev wlp3s0 set power_save off && echo "  本次已关闭" || echo "  (iw 失败)"
S mkdir -p /etc/NetworkManager/conf.d
echo "[connection]
wifi.powersave = 2" | S tee /etc/NetworkManager/conf.d/99-wifi-powersave-off.conf >/dev/null
echo "  已写入 /etc/NetworkManager/conf.d/99-wifi-powersave-off.conf"
S systemctl restart NetworkManager
sleep 6
echo -n "  WiFi powersave 现在: "; iw dev wlp3s0 get power_save 2>/dev/null | tail -1
echo -n "  wlp3s0 状态: "; ip -br addr show wlp3s0 2>/dev/null

echo
echo "=================== 5) 关键服务开机自启 ==================="
S systemctl enable tailscaled docker containerd smbd 2>/dev/null
for s in tailscaled docker containerd; do printf "  %-14s %s\n" "$s" "$(systemctl is-enabled $s 2>/dev/null)"; done

echo
echo "=================== 6) 心跳日志（事后可查掉线时间点） ==================="
mkdir -p $HOME/rag-stack/data/state
CRON="*/5 * * * * date '+\%F \%T' >> \$HOME/rag-stack/data/state/alive.log"
( crontab -l 2>/dev/null | grep -v "alive.log" ; echo "$CRON" ) | crontab - 2>/dev/null
echo "  已加 cron 心跳"

echo
echo "=================== 7) 验证 ==================="
echo -n "  运行时长       : "; uptime -p
echo -n "  sleep.target   : "; systemctl is-enabled sleep.target 2>/dev/null
echo -n "  suspend.target : "; systemctl is-enabled suspend.target 2>/dev/null
echo -n "  IdleAction     : "; grep -E "^IdleAction=" /etc/systemd/logind.conf | cut -d= -f2
echo -n "  tailscaled     : "; systemctl is-active tailscaled
echo -n "  docker         : "; systemctl is-active docker
echo -n "  WiFi           : "; iw dev wlp3s0 get power_save 2>/dev/null | tail -1
echo
echo "  >>> 现在即使 DDE 发起挂起请求，systemd 也会拒绝执行。"