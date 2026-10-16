# Free deployment: Streamlit Community Cloud (about 15 minutes)

Hugging Face Docker Spaces now require a paid plan, so we use Streamlit Community Cloud (free, public
GitHub repo). Its memory limit is about 1 to 2.7 GB, so the cloud copy runs in LITE mode: no QA model,
the best evidence sentence is extracted instead. The local demo runs the full model.

1. Run build_corpus.py and evaluate.py locally FIRST (the cloud uses the saved chunks and results).
2. github.com -> New repository -> public -> name "esahulat-ai" -> Create.
3. "uploading an existing file" -> drag in: app.py, engine.py, normalize.py, requirements.txt,
   README.md, an empty file named LITE_MODE, and the folders data/, outputs/, figures/.
   (Do NOT upload .venv or hf_cache.)  Commit.
4. share.streamlit.io -> sign in with GitHub -> Create app -> pick the repo, branch main, file app.py.
   Advanced settings -> Python 3.12. Deploy.
5. First start downloads e5-small (~470 MB): wait 3 to 6 minutes, then ask one question to warm it up.
6. Put the URL (and a QR code) on your report and demo screen.
If it shows "over its resource limits": reboot from "Manage app". If it still fails, demo locally;
the local app is the primary deliverable.
