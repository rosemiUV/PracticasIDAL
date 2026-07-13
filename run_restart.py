import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('diepry3.uv.es', port=40, username='rosemi', password='IDALR2BPI')

script_content = """#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

cd ~/BPI
git pull origin main

cd frontend
npm install
cd ..

pkill -f uvicorn
pkill -f vite

source .venv/bin/activate
export LD_LIBRARY_PATH=/home/rosemi/BPI/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
nohup uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

cd frontend
nohup npm run dev -- --host 0.0.0.0 --port 5173 > web.log 2>&1 &
"""

sftp = client.open_sftp()
with sftp.open('restart_bpi.sh', 'w') as f:
    f.write(script_content)
sftp.close()

stdin, stdout, stderr = client.exec_command('bash restart_bpi.sh')
out = stdout.read()
err = stderr.read()

with open('restart_out_final.txt', 'wb') as f:
    f.write(out + err)

print("Done")
