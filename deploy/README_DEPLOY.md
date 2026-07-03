# Deployment Notes

## Backend

Deploy the backend as a Python FastAPI service.

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

If `PORT` is not provided by the platform, the app falls back to `8000` when started locally.

Set backend environment variables as needed:

- `MODEL_ROOT`
- `DATA_ROOT`
- `QWEN_MODEL_DIR`
- `RELEASE_LSTM_MODEL_DIR`
- `PRODUCT_LSTM_MODEL_DIR`
- `RELEASE_NOTES_DATA_DIR`
- `PRODUCT_DOCS_DATA_DIR`
- `RELEASE_LSTM_DATA_DIR`
- `PRODUCT_LSTM_DATA_DIR`
- `OLLAMA_BASE_URL`
- `ALLOWED_ORIGINS`
- `PORT`

Notes:

- The backend uses environment variables only when they are present.
- If variables are missing, the repo keeps its local default paths and behavior.
- Data and model files must be placed on the deployment server separately.
- Do not push datasets or model files to GitHub.

## Frontend

Deploy the frontend separately from the backend.

- Configure the frontend API URL before deployment.
- The frontend reads a runtime API base URL and falls back to same-origin requests when nothing is configured.
- Rebuild or redeploy the frontend after setting the API URL value.
- For static hosting, set the backend URL through a runtime global or the `api-base-url` meta tag.

Frontend API configuration example:

- `API_BASE_URL`

## Exclusions

The following are intentionally excluded from GitHub:

- data directories
- model directories and checkpoints
- outputs
- logs
- cache folders
- environment files
- generated test artifacts
