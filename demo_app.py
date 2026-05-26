from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def check_required_files() -> tuple[bool, list[str]]:
    """Check if all required artifact files exist. Returns (all_exist, missing_files)."""
    required_files = [
        "artifacts/classifier/shot_classifier.pt",
        "artifacts/text_classifier/text_classifier.pt",
        "artifacts/dataset/dataset_manifest.json",
    ]
    missing = [f for f in required_files if not Path(f).exists()]
    return len(missing) == 0, missing


def run_analysis(video_path: Path, overlay_mode: str) -> Path:
    output_dir = Path("artifacts/demo_streamlit")
    cmd = [
        sys.executable,
        "scripts/analyze_video.py",
        "--video",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--overlay-mode",
        overlay_mode,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Analysis failed with return code {result.returncode}.\nStderr:\n{result.stderr}")
    return output_dir / video_path.stem.split("_", 1)[0]


def main() -> None:
    import streamlit as st

    st.title("Shortform Shot Guide Demo")

    # Check required files
    all_exist, missing_files = check_required_files()
    if not all_exist:
        st.error(f"Required files are missing:\n" + "\n".join(f"- {f}" for f in missing_files))
        st.info("Please run the full pipeline first: `python scripts/run_full_pipeline.py`")
        return

    uploaded = st.file_uploader("Upload mp4", type=["mp4", "mov", "m4v"])
    overlay_mode = st.selectbox("Overlay mode", ["compact", "detailed"], index=0)
    if uploaded is None:
        st.info("Upload a video to run the demo.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = Path(tmp.name)

    if st.button("Analyze"):
        with st.spinner("Analyzing video..."):
            try:
                demo_dir = run_analysis(video_path, overlay_mode=overlay_mode)
            except RuntimeError as exc:
                st.error(str(exc))
                return
        overlay_path = demo_dir / "output_overlay.mp4"
        summary_path = demo_dir / "guide_summary.txt"
        shot_log_path = demo_dir / "shot_log.csv"
        if overlay_path.exists():
            st.video(str(overlay_path))
        if summary_path.exists():
            st.subheader("Guide Summary")
            st.text(summary_path.read_text(encoding="utf-8"))
        if shot_log_path.exists():
            st.subheader("Shot Log")
            st.dataframe(pd.read_csv(shot_log_path))


if __name__ == "__main__":
    main()
