# Comandos del Servidor (diepry3.uv.es)

Esta guía contiene los comandos necesarios para desplegar, levantar y cerrar la aplicación (Backend y Frontend) en el servidor de la universidad.

## 1. Conexión al servidor
Abre una terminal y conéctate mediante SSH:
```bash
ssh -p40 rosemi@diepry3.uv.es
```

---

## 2. Levantar la aplicación (Segundo plano)

Si cierras la terminal del servidor, los programas normales se cierran. Por eso usamos `nohup` y `&` para dejarlos corriendo "de fondo" y poder cerrar la terminal tranquilamente.

### Arrancar la API (Backend)
```bash
cd ~/BPI
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/rosemi/BPI/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
nohup uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

### Arrancar la Web (Frontend)
```bash
cd ~/BPI/frontend
# Asegurarse de que el .env del servidor apunte a la IP correcta
echo "VITE_API_URL=http://diepry3.uv.es:8000" > .env
nohup npm run dev -- --host 0.0.0.0 --port 5173 > web.log 2>&1 &
```

---

## 3. Comprobar que están funcionando (Logs)

Para ver qué está pasando o si ha habido algún error, puedes leer los archivos de registro (logs) que hemos creado:

**Para ver los logs de la API:**
```bash
cd ~/BPI
tail -f api.log
```

**Para ver los logs del Frontend:**
```bash
cd ~/BPI/frontend
tail -f web.log
```
*(Para salir de la vista de logs, pulsa `Ctrl+C`)*

---

## 4. Cerrar / Apagar la aplicación

Si necesitas reiniciar la aplicación o quieres apagarla por completo, tienes que "matar" los procesos que dejamos corriendo en segundo plano.

### Apagar el Backend (API)
```bash
pkill -f uvicorn
```

### Apagar el Frontend (Web)
```bash
pkill -f "vite"
```
*(Opcionalmente, si sigue corriendo, puedes probar con `pkill -f node`)*
