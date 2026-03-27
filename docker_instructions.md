# Running Serial Audio Inference with Docker Compose

I have integrated the `audio-streamer` into your `docker-compose.yml`. Here is how to ensure it runs correctly on Windows.

## 1. Map COM Port to WSL2 (Windows Only)
Docker Desktop on Windows runs inside WSL2. To give a Docker container access to a USB serial port (like `COM4`), you must first "attach" the device to WSL2.

1.  **Install usbipd-win**: Download from [github.com/dorssel/usbipd-win](https://github.com/dorssel/usbipd-win).
2.  **Identify the Bus ID**: Open PowerShell as Admin and run:
    ```powershell
    usbipd list
    ```
    Find your Arduino (e.g., `2-1`).
3.  **Bind and Attach**:
    ```powershell
    usbipd bind --busid 2-1
    usbipd attach --wsl --busid 2-1
    ```
4.  **Confirm in WSL2**: The device should now appear as `/dev/ttyACM0` (or similar) inside your WSL2 terminal.

## 2. Configure Docker Compose
Update your `.env` file or set the following environment variables:
```yaml
SERIAL_PORT=/dev/ttyACM0
NEXTJS_WEBHOOK_URL=http://host.docker.internal:3000/api/audio/webhook
```
*(Note: `host.docker.internal` allows the container to talk to your Next.js app running on the host machine).*

## 3. Run
```bash
docker-compose build
docker-compose up
```

## Troubleshooting
- **Permission Denied**: If the container can't access `/dev/ttyACM0`, you might need to add `user: root` to the `audio-streamer` service in `docker-compose.yml` or chmod the device in the container.
- **Model Loading**: The first run will download YAMNet (~30MB) and cached in the container.
