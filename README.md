# Gen2Dlive

CPU-only 2D-live loop generator for watercolor/ink style images.

It generates a single seamless-loop MP4 by combining anime foreground parsing, thin motion layers (hair / upper cloth / lower cloth), and optional floating particles.

## Quickstart (Windows / PowerShell)

```powershell
cd C:\Users\User\Desktop\Script\Gen2Dlive
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

If `ffmpeg` is available in `PATH`, the service uses it for higher-quality H.264 output. Otherwise it falls back to OpenCV `mp4v`.

The first request with `GEN2DLIVE_SEGMENTATION_BACKEND=rembg` may download the configured ONNX model to `~/.u2net`.

## API

- `GET /healthz` -> `{"ok": true}`
- `POST /animate`
  - synchronous multipart API
  - returns `video/mp4` directly
- `POST /animate` (multipart/form-data)
  - input image: `image` (file) or `image_base64` (string)
  - provide only one of them
  - `duration_sec` (float, default `3.0`) single cycle length in seconds
  - `fps` (int, default `30`)
  - `width` (int, optional) output width, must be used with `height`
  - `height` (int, optional) output height, must be used with `width`
  - `size` (int, optional, legacy) output square size
  - priority: `width+height` > `size` > original image size
  - `strength` (float, default `2.0`) overall motion strength
  - `particles` (int, default `0`) number of floating particles (set `0` to disable)
- `POST /animate-json`
  - synchronous JSON API
  - returns `video/mp4` directly
- `POST /animate-json` (`application/json`)
  - `image_base64` (string, supports plain base64 or Data URI)
  - `duration_sec` (float, default `3.0`) single cycle length in seconds
  - `fps` (int, default `30`)
  - `width` (int, optional) output width, must be used with `height`
  - `height` (int, optional) output height, must be used with `width`
  - `size` (int, optional, legacy) output square size
  - priority: `width+height` > `size` > original image size
  - `strength` (float, default `2.0`) overall motion strength
  - `particles` (int, default `0`) number of floating particles (set `0` to disable)
- `POST /jobs/animate`
  - asynchronous multipart API
  - returns `202` JSON with `job_id`
- `POST /jobs/animate-json`
  - asynchronous JSON API
  - returns `202` JSON with `job_id`
- `GET /jobs/{job_id}`
  - returns job status JSON
  - when finished, includes `result_url`
- `GET /jobs/{job_id}/result`
  - downloads the generated `video/mp4`
  - returns `409` if the job is still running or failed
  - result is one-time download in the current in-memory implementation

## Auth

- If `GEN2DLIVE_API_KEY` is empty or not set, API is open.
- If `GEN2DLIVE_API_KEY` is set (system env, `.env.local`, or `.env`), all non-`/healthz` endpoints require auth.
- Supported headers:
  - `Authorization: Bearer <GEN2DLIVE_API_KEY>`
  - `X-API-Key: <GEN2DLIVE_API_KEY>`

Multipart file example:

```powershell
curl.exe -X POST "http://127.0.0.1:8010/animate" `
  -F "image=@example.png" `
  -F "duration_sec=3.0" -F "fps=30" -F "width=1280" -F "height=720" -F "strength=1.0" -F "particles=0" `
  --output out.mp4
```

JSON base64 example:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("example.png"))
curl.exe -X POST "http://127.0.0.1:8010/animate-json" `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your_key_here" `
  -d "{""image_base64"":""$b64"",""duration_sec"":3.0,""fps"":30,""width"":1280,""height"":720}" `
  --output out.mp4
```

Async JSON job example:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("example.png"))
$job = curl.exe -X POST "http://127.0.0.1:8010/jobs/animate-json" `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your_key_here" `
  -d "{""image_base64"":""$b64"",""duration_sec"":3.0,""fps"":30}" | ConvertFrom-Json

curl.exe -H "Authorization: Bearer your_key_here" "http://127.0.0.1:8010/jobs/$($job.job_id)"
curl.exe -H "Authorization: Bearer your_key_here" "http://127.0.0.1:8010/jobs/$($job.job_id)/result" --output out.mp4
```

## Notes

- This is a lightweight, non-generative approach. It uses thin inpainted holdout regions for moved layers, but it does **not** generate missing background content.
- `duration_sec` now controls the length of one exported cycle instead of repeating the motion for an arbitrary duration.
- Default foreground parsing is `isnet-anime` through `rembg`; set `GEN2DLIVE_SEGMENTATION_MODEL` if you need `u2net_human_seg` or `u2net`.
- For best results, keep `strength` subtle (`0.7`-`1.4`) on complex scenes.
- The async job queue is currently in-process and in-memory. If the process restarts, queued/running job state is lost.
- Completed async results are currently consumed on first successful `/jobs/{job_id}/result` download.
