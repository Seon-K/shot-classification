from __future__ import annotations

import subprocess
import sys
import tempfile
import urllib.request
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


def repair_video(video_path: Path) -> Path:
    """Repair corrupted/incomplete video using av."""
    try:
        import av
        
        # First, try to verify the video
        try:
            container = av.open(str(video_path))
            stream = next(s for s in container.streams.video)
            # If we can read frames, it's ok
            for _ in container.decode(stream):
                break
            return video_path  # Video is fine
        except Exception:
            pass
        
        # Try to repair
        repaired_path = video_path.parent / f"{video_path.stem}_repaired.mp4"
        
        container_in = av.open(str(video_path))
        container_out = av.open(str(repaired_path), 'w')
        
        # Copy video streams
        for stream in container_in.streams.video:
            out_stream = container_out.add_stream("h264", rate=stream.rate)
            out_stream.width = stream.width
            out_stream.height = stream.height
            
            for frame in container_in.decode(stream):
                for packet in out_stream.encode(frame):
                    container_out.mux(packet)
            
            # Flush encoder
            for packet in out_stream.encode():
                container_out.mux(packet)
        
        container_out.close()
        container_in.close()
        
        if repaired_path.exists() and repaired_path.stat().st_size > 0:
            video_path.unlink()
            return repaired_path
    except Exception as e:
        print(f"Video repair failed: {e}")
    
    return video_path


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

    input_mode = st.radio("Input method", ["File upload", "URL"], index=0)
    overlay_mode = st.selectbox("Overlay mode", ["compact", "detailed"], index=0)

    video_path = None

    if input_mode == "File upload":
        uploaded = st.file_uploader("Upload mp4", type=["mp4", "mov", "m4v"])
        if uploaded is None:
            st.info("Upload a video to run the demo.")
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            video_path = Path(tmp.name)
    else:  # URL mode
        video_url = st.text_input("Enter video URL (mp4, mov, m4v)")
        if not video_url:
            st.info("Enter a video URL to run the demo.")
            return
        try:
            with st.spinner("Downloading video..."):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_path = tmp.name
                tmp.close()
                
                # Download with timeout
                try:
                    urllib.request.urlretrieve(video_url, tmp_path, timeout=60)
                except Exception as e:
                    st.error(f"Download failed: {e}")
                    return
                
                # Check file size
                file_size = Path(tmp_path).stat().st_size
                if file_size < 1024 * 100:  # Less than 100KB
                    st.error(f"Downloaded file too small ({file_size} bytes). URL might not be a valid video.")
                    return
                
                video_path = Path(tmp_path)
            st.success(f"Video downloaded successfully! ({file_size / (1024*1024):.1f} MB)")
        except Exception as e:
            st.error(f"Failed to download video: {e}")
            return

    if st.button("Analyze"):
        with st.spinner("Analyzing video..."):
            try:
                # Repair video if needed before analysis
                video_path_to_use = repair_video(video_path)
                demo_dir = run_analysis(video_path_to_use, overlay_mode=overlay_mode)
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
