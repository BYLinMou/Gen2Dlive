# Gen2Dlive

CPU-only “2D live” loop generator for watercolor/ink style images.

It generates a short seamless-loop MP4 by applying subtle, masked warps (hair/sleeves) and optional floating particles.

## Quickstart (Windows / PowerShell)

```powershell
cd C:\Users\User\Desktop\Script\Gen2Dlive
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

If `ffmpeg` is available in `PATH`, the service uses it for higher-quality H.264 output. Otherwise it falls back to OpenCV’s `mp4v`.

## API

- `GET /healthz` -> `{"ok": true}`
- `POST /animate` (multipart/form-data)
  - `image` (file): input image (png/jpg/webp)
  - `duration_sec` (float, default `4.0`)
  - `fps` (int, default `24`)
  - `size` (int, default `768`) output square size
  - `strength` (float, default `1.0`) overall motion strength
  - `particles` (int, default `18`) number of floating particles (set `0` to disable)

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8010/animate" `
  -F "image=@example.png" `
  -F "duration_sec=4" -F "fps=24" -F "size=768" -F "strength=1.0" -F "particles=18" `
  --output out.mp4
```

## Notes

- This is a lightweight, non-generative approach. It does **not** inpaint new background content.
- For best results, keep `strength` subtle (`0.6`–`1.2`).
