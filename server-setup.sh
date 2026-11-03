#!/usr/bin/env bash
# server-setup.sh —— Samba 共享 + cron 自动索引
# 用法：bash server-setup.sh
PW="${SUDO_PASS:-<YOUR_SUDO_PASSWORD>}"
S() { echo "$PW" | sudo -S "$@" 2>&1 | grep -viE "验证成功|请输入密码" || true; }
SMB_USER=<SSH_USER>
SMB_PASS="$PW"
STACK_DIR="$HOME/rag-stack"
SHARE_DIR="$STACK_DIR/docs"

echo "=================== 1) 安装 Samba ==================="
if ! command -v smbd >/dev/null 2>&1; then
  echo "$PW" | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y -qq samba 2>&1 | grep -viE "验证成功|请输入密码" | tail -5
else
  echo "  samba 已安装"
fi
command -v smbd smbpasswd

echo
echo "=================== 2) 写共享配置（临时文件 + sudo cp）==================="
mkdir -p "$SHARE_DIR"
if grep -q "^\[ragdocs\]" /etc/samba/smb.conf 2>/dev/null; then
  echo "  [ragdocs] 已存在，跳过"
else
  # 先备份
  S cp /etc/samba/smb.conf /etc/samba/smb.conf.bak.$(date +%F-%H%M)
  # 生成追加片段到临时文件
  cat > /tmp/smb-append.conf <<EOF

[ragdocs]
   comment = 知识库文档目录（拖文件到这里即可自动入库）
   path = $SHARE_DIR
   browseable = yes
   read only = no
   valid users = $SMB_USER
   create mask = 0644
   directory mask = 0755
   server min protocol = SMB2
   veto files = /._*/.DS_Store/Thumbs.db/
   delete veto files = yes
EOF
  # 合并
  cat /etc/samba/smb.conf /tmp/smb-append.conf > /tmp/smb-new.conf
  S cp /tmp/smb-new.conf /etc/samba/smb.conf
  rm -f /tmp/smb-append.conf /tmp/smb-new.conf
  echo "  已写入 [ragdocs]"
fi
echo "  --- smb.conf 中的共享段 ---"
sed -n '/^\[ragdocs\]/,$p' /etc/samba/smb.conf

echo
echo "=================== 3) 设置 Samba 用户密码 ==================="
printf '%s\n%s\n' "$SMB_PASS" "$SMB_PASS" > /tmp/.smbpw
S smbpasswd -a -s "$SMB_USER" < /tmp/.smbpw
rm -f /tmp/.smbpw
S smbpasswd -e "$SMB_USER" 2>/dev/null || true
echo "  已设置 $SMB_USER 的 Samba 密码"
S pdbedit -L 2>/dev/null | sed 's/^/    /'

echo
echo "=================== 4) 启动 smbd ==================="
S systemctl enable --now smbd
S systemctl restart smbd
sleep 3
echo -n "  smbd 状态: "; systemctl is-active smbd
echo -n "  445 监听:  "; ss -tln 2>/dev/null | grep -c ':445'
ss -tln 2>/dev/null | grep -E ':(445|139)' | sed 's/^/    /'

echo
echo "=================== 5) cron 自动索引（每 10 分钟）==================="
mkdir -p "$STACK_DIR/data/state"
cat > /tmp/ragcron <<EOF
*/10 * * * * cd $STACK_DIR && /usr/local/bin/docker-compose --profile tools run --rm indexer >> $STACK_DIR/data/state/cron.log 2>&1
EOF
( crontab -l 2>/dev/null | grep -v "rag-stack" ; cat /tmp/ragcron ) | crontab -
rm -f /tmp/ragcron
echo "  当前 crontab:"
crontab -l | sed 's/^/    /'

echo
echo "=================== 6) 验收 ==================="
echo -n "  smbd:          "; systemctl is-active smbd
echo -n "  445 端口:      "; ss -tln 2>/dev/null | grep -c ':445'
echo -n "  共享目录:      "; ls -ld "$SHARE_DIR"
echo -n "  目录内文件数:  "; find "$SHARE_DIR" -type f | wc -l
echo
echo "  Windows 访问:  \\\\<OLD_SERVER_IP>\\ragdocs"
echo "  用户名: $SMB_USER   密码: $SMB_PASS"