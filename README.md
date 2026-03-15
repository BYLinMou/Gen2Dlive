# Gen2Dlive

CPU-only 2D-live loop generator for watercolor/ink style images.

It generates a short seamless-loop MP4 by applying subtle, masked warps (hair/sleeves) and optional floating particles.

## Quickstart (Windows / PowerShell)

```powershell
cd C:\Users\User\Desktop\Script\Gen2Dlive
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

If `ffmpeg` is available in `PATH`, the service uses it for higher-quality H.264 output. Otherwise it falls back to OpenCV `mp4v`.

## API

- `GET /healthz` -> `{"ok": true}`
- `POST /animate` (multipart/form-data)
  - input image: `image` (file) or `image_base64` (string)
  - provide only one of them
  - `duration_sec` (float, default `4.0`)
  - `fps` (int, default `30`)
  - `width` (int, optional) output width, must be used with `height`
  - `height` (int, optional) output height, must be used with `width`
  - `size` (int, optional, legacy) output square size
  - priority: `width+height` > `size` > original image size
  - `strength` (float, default `1.0`) overall motion strength
  - `particles` (int, default `18`) number of floating particles (set `0` to disable)
- `POST /animate-json` (`application/json`)
  - `image_base64` (string, supports plain base64 or Data URI)
  - `duration_sec` (float, default `4.0`)
  - `fps` (int, default `30`)
  - `width` (int, optional) output width, must be used with `height`
  - `height` (int, optional) output height, must be used with `width`
  - `size` (int, optional, legacy) output square size
  - priority: `width+height` > `size` > original image size
  - `strength` (float, default `1.0`) overall motion strength
  - `particles` (int, default `18`) number of floating particles (set `0` to disable)

## Auth

- If `GEN2DLIVE_API_KEY` is empty or not set, API is open.
- If `GEN2DLIVE_API_KEY` is set (system env, `.env.local`, or `.env`), `POST /animate` and `POST /animate-json` require auth.
- Supported headers:
  - `Authorization: Bearer <GEN2DLIVE_API_KEY>`
  - `X-API-Key: <GEN2DLIVE_API_KEY>`

Multipart file example:

```powershell
curl.exe -X POST "http://127.0.0.1:8010/animate" `
  -F "image=@example.png" `
  -F "duration_sec=4" -F "fps=30" -F "width=1280" -F "height=720" -F "strength=1.0" -F "particles=18" `
  --output out.mp4
```

JSON base64 example:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("example.png"))
curl.exe -X POST "http://127.0.0.1:8010/animate-json" `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your_key_here" `
  -d "{""image_base64"":""$b64"",""duration_sec"":4,""fps"":30,""width"":1280,""height"":720}" `
  --output out.mp4
```

## Notes

- This is a lightweight, non-generative approach. It does **not** inpaint new background content.
- For best results, keep `strength` subtle (`0.6`-`1.2`).
