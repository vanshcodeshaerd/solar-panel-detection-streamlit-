# How to Run This Project

## Fix: "Roboflow model not loaded"

If you see that message in the app:

1. **Get your Roboflow API key:** Go to [https://app.roboflow.com](https://app.roboflow.com) → **Settings** → **Roboflow API** → copy the key.
2. **Add it in one of these places:**
   - **Option A — `.env`** (in this project folder): open `.env` and set  
     `ROBOFLOW_API_KEY=paste_your_key_here`  
     (replace `paste_your_key_here` with your actual key; leave no spaces around `=`).
   - **Option B — Streamlit secrets:** open `.streamlit/secrets.toml` and replace `"your_roboflow_api_key_here"` with your actual key in quotes.
3. **Restart the app:** Stop the running Streamlit app (Ctrl+C in the terminal), then run again:  
   `python -m streamlit run streamlit_roboflow_7.py`

After that, detection should work.

---

## Google Maps — Aerial / satellite view

To use **Latitude/Longitude** in the app (aerial satellite view of an area), you need a **Google Maps API key** and the **Maps Static API** enabled.

### Step 1: Google Cloud project

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Sign in with your Google account.
3. Create a project (or pick an existing one):
   - Click the project dropdown at the top → **New Project**.
   - Name it (e.g. `solar-panel-app`) → **Create**.

### Step 2: Enable Maps Static API

1. In the Cloud Console, open **APIs & Services** → **Library** (or go to [API Library](https://console.cloud.google.com/apis/library)).
2. Search for **Maps Static API**.
3. Open **Maps Static API** → click **Enable**.

### Step 3: Create an API key

1. Go to **APIs & Services** → **Credentials** (or [Credentials](https://console.cloud.google.com/apis/credentials)).
2. Click **+ Create Credentials** → **API key**.
3. Copy the new key. (Optional: click **Edit API key** → under **API restrictions** choose **Restrict key** → select only **Maps Static API** → Save.)

### Step 4: Add the key to the app

Use **one** of these:

- **Option A — `.env`**  
  In the project folder, open `.env` and set:
  ```env
  GOOGLE_API_KEY=your_actual_google_api_key_here
  ```
  (no quotes, no spaces around `=`).

- **Option B — Streamlit secrets**  
  Open `.streamlit/secrets.toml` and set:
  ```toml
  GOOGLE_API_KEY = "your_actual_google_api_key_here"
  ```
  (use quotes in TOML).

### Step 5: Restart the app

Stop the app (Ctrl+C), then run again:

```powershell
python -m streamlit run streamlit_roboflow_7.py
```

In the app, choose **Latitude/Longitude**, enter coordinates, and click **Fetch & Detect** to load the aerial view and run solar panel detection.

**Note:** Maps Static API has a [free tier](https://developers.google.com/maps/billing-and-pricing); usage beyond that is billed by Google.

---

## Prerequisites

- Python 3.11 (see `.python-version`)
- pip

## One-Time Setup

1. **Create a virtual environment (recommended):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - A `.env` file already exists with default structure.
   - **Roboflow** (required for detection): add `ROBOFLOW_API_KEY` — see *Fix: Roboflow model not loaded* above.
   - **Google Maps** (for aerial/satellite view by lat/long): add `GOOGLE_API_KEY` — see *Google Maps — Aerial / satellite view* below.
   - Default `ROBOFLOW_PROJECT=solarpv-india-lczsp` is set; change if using another project.

   **Alternative:** use `.streamlit/secrets.toml` instead of `.env` and fill in the same keys in TOML format.

## Run the App

**Backend + Frontend (single command — this is a Streamlit app):**

**Always use `python` first** (so `-m` is passed to Python, not run as a separate command):

```powershell
python -m streamlit run streamlit_roboflow_7.py
```

Alternatively, if `streamlit` is on your PATH:

```powershell
streamlit run streamlit_roboflow_7.py
```

- App URL: http://localhost:8501 (opens in browser).
- If you did not set `ROBOFLOW_API_KEY`, the app still starts; the sidebar shows a warning and detection will prompt you to set the key.
- **Upload Image**: works as soon as `ROBOFLOW_API_KEY` is set.
- **Latitude/Longitude**: also requires `GOOGLE_API_KEY` in `.env` or `secrets.toml`.

## Other Scripts (optional)

- `yolo8_solar.py` — Excel input (uses `.env`): `python yolo8_solar.py`
- `yolo8_solar_csv.py` — CSV input: `python yolo8_solar_csv.py`
- `yolo8_solar_zoom21.py` — Zoom 21 version: `python yolo8_solar_zoom21.py`

## APIs Used

| API            | Purpose                          | Env variable       |
|----------------|-----------------------------------|--------------------|
| Roboflow       | Solar panel detection model      | `ROBOFLOW_API_KEY` |
| Google Maps    | Satellite imagery for lat/long   | `GOOGLE_API_KEY`   |

All endpoints are consumed inside the Streamlit app (no separate backend server). Configuration is read from `.env` (via `python-dotenv`) or `.streamlit/secrets.toml`.

## Notes

- No database or migrations; no auth flow.
- Main app entry: `streamlit_roboflow_7.py` (per DEPLOYMENT.md).
- If Roboflow project or version differs, update `ROBOFLOW_PROJECT` and the `project.version(6)` call in `streamlit_roboflow_7.py` if needed.
