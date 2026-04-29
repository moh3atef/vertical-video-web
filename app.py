import streamlit as st
import os
from main import VerticalNewsConverter


st.set_page_config(
    page_title="Vertical News Converter",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Vertical News Converter")
st.write("Batch convert 16:9 broadcast videos into clean 9:16 social videos.")

uploaded_files = st.file_uploader(
    "Upload one or more landscape news videos",
    type=["mp4", "mov", "m4v"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Videos uploaded: {len(uploaded_files)}")

    if st.button("Start Batch Processing"):
        converter = VerticalNewsConverter()

        progress_bar = st.progress(0)
        status_text = st.empty()

        output_files = []

        os.makedirs("batch_inputs", exist_ok=True)
        os.makedirs("batch_outputs", exist_ok=True)

        for index, uploaded_file in enumerate(uploaded_files):
            input_path = os.path.join("batch_inputs", uploaded_file.name)

            base_name = os.path.splitext(uploaded_file.name)[0]
            output_name = f"{base_name}_vertical.mp4"
            output_path = os.path.join("batch_outputs", output_name)

            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())

            status_text.write(f"Processing {index + 1} of {len(uploaded_files)}: {uploaded_file.name}")

            try:
                converter.process_video(input_path, output_path)
                output_files.append(output_path)
                st.success(f"Done: {output_name}")

            except Exception as e:
                st.error(f"Failed: {uploaded_file.name}")
                st.exception(e)

            progress_bar.progress((index + 1) / len(uploaded_files))

        status_text.write("Batch processing completed.")

        st.subheader("Download processed videos")

        for output_path in output_files:
            with open(output_path, "rb") as file:
                st.download_button(
                    label=f"Download {os.path.basename(output_path)}",
                    data=file,
                    file_name=os.path.basename(output_path),
                    mime="video/mp4"
                )

if st.button("Clear Batch Files"):
    for folder in ["batch_inputs", "batch_outputs"]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))
    st.success("Batch files cleared.")
