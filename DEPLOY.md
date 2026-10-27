# Deployment guide

## 0. Before you deploy
```bash
python scripts/build_db.py --data data/genres_original --out db/fingerprints.npz
ls -lh db/fingerprints.npz        # must be < 100 MB for plain GitHub
streamlit run app.py              # check it works locally first
```

## (a) Local run
```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q && streamlit run app.py
```

## (b) Push to GitHub
```bash
git init
git add .
git status                        # data/ and venv/ must NOT appear; db/fingerprints.npz must
git commit -m "Mini Shazam: audio fingerprinting app"
git branch -M main
git remote add origin https://github.com/<user>/audio-fingerprinting.git
git push -u origin main
```
**DB bigger than 100 MB?** Pick one:
- *Git LFS:* `git lfs install && git lfs track "db/*.npz" && git add .gitattributes db/fingerprints.npz`. Note: Streamlit Cloud pulls LFS files, but LFS bandwidth on free GitHub is limited.
- *Host it elsewhere (recommended):* upload `fingerprints.npz` to a Hugging Face model/dataset repo or a GitHub Release, copy the direct download URL, and set it as env var / secret `FP_DB_URL`. The app downloads it once at startup (`ensure_db` in `src/utils.py`) and `@st.cache_resource` keeps it in memory.
- *Or* shrink it: fingerprint only the first 20 s of each song (`--max-seconds 20`), or lower `FAN_OUT` to 6.

## (c) Streamlit Community Cloud (primary)
1. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
2. Repository: `<user>/audio-fingerprinting`, branch `main`, **Main file path: `app.py`**.
3. **Advanced settings -> Python version: 3.11**.
4. Secrets (only if you host the DB externally): `FP_DB_URL = "https://..."`. In Streamlit secrets, root keys are also exposed as environment variables.
5. Deploy. First build takes 3-6 minutes (installs ffmpeg from `packages.txt` + librosa/numba).
6. Copy the app URL into README and your resume.

## (d) Hugging Face Spaces (Docker, secondary)
1. https://huggingface.co/new-space -> SDK: **Docker** -> blank template -> Free CPU.
2. The Space's `README.md` must start with this header (put it at the very top of the README in the Space repo only):
```yaml
---
title: Mini Shazam
emoji: 🎵
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```
3. Push the same code:
```bash
git remote add space https://huggingface.co/spaces/<hf-user>/mini-shazam
git push space main
```
Use an access token with write permission as the password. Large binary files on HF need `git lfs track "*.npz"` (HF requires LFS for files > 10 MB).
4. The Dockerfile already installs ffmpeg, runs as user 1000, listens on port 7860 and disables XSRF/CORS checks, which Spaces needs because the app runs inside an iframe.

## Troubleshooting
| Problem | Fix |
|---|---|
| `NoBackendError` / m4a or mp3 won't load | `packages.txt` must contain `ffmpeg` and `libsndfile1` (one per line, repo root). Reboot the app. |
| Memory exceeded / app restarts | Shrink the DB (`--max-seconds 20`, lower `FAN_OUT`). Keep `plt.close(fig)` calls. Don't load audio at startup. |
| `numba` / `librosa` import error or cache permission error | Set env `NUMBA_CACHE_DIR=/tmp/numba_cache`. Make sure Python is 3.11 and versions match `requirements.txt`. |
| `Fingerprint database not found` | `db/fingerprints.npz` not committed (check `.gitignore`) or `FP_DB_URL` not set. |
| `st.audio_input` missing | Streamlit older than 1.39. The app still works without the mic widget; keep `streamlit==1.40.2`. |
| Microphone not working | Browsers need HTTPS; deployed apps have it, `localhost` works too. Allow mic permission. |
| App "went to sleep" | Free Streamlit apps sleep after ~12 h of no traffic; click "Wake up". Open the link once before an interview. HF Spaces also sleeps after inactivity. |
| Wrong match / "no confident match" for good clips | Tune `MIN_SCORE`, `MIN_RATIO` in `src/match.py`, and `AMP_MIN_DB`, `NEIGHBORHOOD` in `src/fingerprint.py`. Rebuild the DB after changing DSP parameters. |
