# Deploy GPU Compare on Ubuntu 24.04 (port 9000)

These steps set up a systemd service so the app runs on port 9000 and starts automatically after every boot.

## 1. Copy app files to the server

On your Ubuntu 24.04 server, create the app directory and copy the required files:

```bash
sudo mkdir -p /opt/gpu-compare
sudo cp gpu_compare.html gpu_specs.csv /opt/gpu-compare/
sudo chown -R ubuntu:ubuntu /opt/gpu-compare
```

(Replace `ubuntu` with your server user if different.)

## 2. Install the systemd service

```bash
sudo cp gpu-compare.service /etc/systemd/system/
```

Edit the service if your paths or user differ:

```bash
sudo nano /etc/systemd/system/gpu-compare.service
```

- **User/Group**: Change `User=ubuntu` and `Group=ubuntu` if you use another user.
- **WorkingDirectory**: Change `/opt/gpu-compare` if you put the files somewhere else.

## 3. Enable and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable gpu-compare
sudo systemctl start gpu-compare
```

Check status:

```bash
sudo systemctl status gpu-compare
```

You should see `active (running)` and the server listening on port 9000.

## 4. Open in browser

From another machine:

- **http://YOUR_SERVER_IP:9000/gpu_compare.html**

From the server itself:

- **http://localhost:9000/gpu_compare.html**

## 5. Use a hostname instead of the IP (DNS)

You can use a name in the browser so users don’t have to remember the server IP.

### Option A: Public domain (e.g. gpucompare.example.com)

If you own a domain (e.g. from Cloudflare, Namecheap, Route53):

1. In your DNS provider, add an **A record**:
   - **Name**: `gpucompare` (or any subdomain you want)
   - **Value**: your server’s **public IP**
   - **TTL**: 300 or default

2. After DNS propagates (a few minutes), open:
   - **http://gpucompare.example.com:9000/gpu_compare.html**

To avoid typing `:9000`, use Option C (reverse proxy) below.

### Option B: Local network only (no domain)

On a home or office LAN:

- **Router hostname**: Some routers (e.g. when DHCP is used) assign a hostname like `ubuntu-server` or `hostname.local`. Check your router’s “DHCP client list” or “Connected devices” for the hostname of the Ubuntu machine, then try **http://hostname:9000/gpu_compare.html**.
- **mDNS (e.g. .local)**: If Avahi is installed on the server (`sudo apt install avahi-daemon`), other devices on the LAN can use **http://your-server-name.local:9000/gpu_compare.html** (replace `your-server-name` with the server’s hostname).
- **/etc/hosts on each client**: On each computer that will open the app, add a line (e.g. on Mac/Linux edit `/etc/hosts`, on Windows `C:\Windows\System32\drivers\etc\hosts`):
  ```text
  SERVER_IP   gpucompare
  ```
  Then open **http://gpucompare:9000/gpu_compare.html** on that machine.

### Option C: Friendly URL without :9000 (reverse proxy)

To use **http://gpucompare.example.com/gpu_compare.html** (no port), put Nginx (or Caddy) in front and proxy to port 9000:

**Nginx example** (after `sudo apt install nginx`):

```bash
sudo nano /etc/nginx/sites-available/gpu-compare
```

Add (replace `gpucompare.example.com` with your hostname or server IP):

```nginx
server {
    listen 80;
    server_name gpucompare.example.com;   # or your server IP for testing
    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/gpu-compare /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

Then open **http://gpucompare.example.com/gpu_compare.html** (no port). If you use a firewall, allow port 80: `sudo ufw allow 80/tcp && sudo ufw reload`.

## 6. Firewall (if UFW is enabled)

Allow port 9000:

```bash
sudo ufw allow 9000/tcp
sudo ufw reload
```

## Useful commands

| Command | Purpose |
|--------|---------|
| `sudo systemctl status gpu-compare` | Check if the service is running |
| `sudo systemctl restart gpu-compare` | Restart after changing HTML/CSV |
| `sudo systemctl stop gpu-compare` | Stop the service |
| `sudo journalctl -u gpu-compare -f` | Follow service logs |

## After updating files

When you update `gpu_compare.html` or `gpu_specs.csv` in `/opt/gpu-compare`, no need to restart the service—reload the page in the browser. Restart only if you change the service file:

```bash
sudo systemctl daemon-reload
sudo systemctl restart gpu-compare
```
