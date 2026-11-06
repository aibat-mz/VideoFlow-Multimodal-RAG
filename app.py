import streamlit as st
import os
import base64
import mimetypes
from urllib.parse import parse_qs, urlparse
from moviepy.editor import VideoFileClip
from pathlib import Path
import speech_recognition as sr
from PIL import Image
import lancedb
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.schema import ImageNode
import openai
from dotenv import load_dotenv
import json

load_dotenv()

# Streamlit app setup with a custom layout and title
st.set_page_config(page_title="YouTube Video Q&A with LLM", layout="centered", page_icon="🎬")

# Custom CSS styling to enhance UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        margin-top: 10px;
        font-size: 16px;
    }
    .stTextInput > div > input {
        background-color: #e9e9f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 YouTube Video Q&A with LLM")
st.subheader("🚀 Process and ask questions about YouTube videos.")

st.write("Input a YouTube video link, process it, and ask questions about its content. Enjoy exploring! 🎉")
st.markdown("<hr style='border:2px solid gray'>", unsafe_allow_html=True)

# OpenAI API key input
env_api_key = os.getenv("OPENAI_API_KEY")
if env_api_key:
    openai.api_key = env_api_key
    video_step_number = 1
    question_step_number = 2
else:
    st.write("**Step 1: Enter your OpenAI API Key**")
    api_key = st.text_input("OpenAI API Key:", type="password", placeholder="sk-xxxxxxxxxxxxxxxx", help="Enter your OpenAI API key here.")
    openai.api_key = api_key
    video_step_number = 2
    question_step_number = 3

# Path configurations
output_video_path = "./video_data/"
output_folder = "./mixed_data/"
output_audio_path = "./mixed_data/output_audio.wav"
filepath = output_video_path + "input_vid.mp4"
Path(output_video_path).mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

def get_youtube_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ("youtu.be", "www.youtu.be"):
        return parsed_url.path.lstrip("/")
    if parsed_url.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        query_id = parse_qs(parsed_url.query).get("v")
        if query_id:
            return query_id[0]
        path_parts = [part for part in parsed_url.path.split("/") if part]
        if path_parts and path_parts[0] in ("embed", "shorts", "live") and len(path_parts) > 1:
            return path_parts[1]
    return None

def get_cache_paths(video_id):
    video_dir = os.path.join(output_video_path, video_id)
    data_dir = os.path.join(output_folder, video_id)
    return {
        "video_dir": video_dir,
        "data_dir": data_dir,
        "video_path": os.path.join(video_dir, "input_vid.mp4"),
        "audio_path": os.path.join(data_dir, "output_audio.wav"),
        "text_path": os.path.join(data_dir, "output_text.txt"),
        "metadata_path": os.path.join(video_dir, "metadata.json"),
        "text_table": f"text_collection_{video_id}",
        "image_table": f"image_collection_{video_id}",
    }

def get_frame_paths(data_dir):
    return sorted(str(path) for path in Path(data_dir).glob("frame*.png"))

def cached_index_exists(text_table, image_table):
    db = lancedb.connect("lancedb")
    table_names = db.table_names()
    return text_table in table_names and image_table in table_names

def create_retriever(video_id, cached_index=False):
    cache_paths = get_cache_paths(video_id)
    mode = "append" if cached_index else "overwrite"
    text_store = LanceDBVectorStore(uri="lancedb", table_name=cache_paths["text_table"], mode=mode)
    image_store = LanceDBVectorStore(uri="lancedb", table_name=cache_paths["image_table"], mode=mode)
    if cached_index:
        index = MultiModalVectorStoreIndex.from_vector_store(
            text_store,
            image_vector_store=image_store,
            image_embed_model="clip:ViT-B/32",
        )
    else:
        input_files = [cache_paths["text_path"]] + get_frame_paths(cache_paths["data_dir"])
        storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)
        documents = SimpleDirectoryReader(input_files=input_files).load_data()
        index = MultiModalVectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index.as_retriever(similarity_top_k=3, image_similarity_top_k=3)

def image_path_to_data_url(image_path):
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"

# Function to download video from YouTube
def download_video(url, output_path):
    import logging

    import imageio_ffmpeg
    import yt_dlp

    logger = logging.getLogger(__name__)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path) / "input_vid.mp4"
    output_existed = output_file.exists()
    completed = False

    try:
        logger.info("YouTube download: retrieving metadata and formats with yt-dlp")
        options = {
            "noplaylist": True,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(Path(output_path) / "input_vid.%(ext)s"),
            "merge_output_format": "mp4",
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
            "socket_timeout": 60,
            "retries": 10,
            "fragment_retries": 10,
            "file_access_retries": 3,
            "concurrent_fragment_downloads": 1,
            "logger": logger,
            "quiet": True,
            "no_warnings": False,
        }

        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)

        metadata = {
            "Author": info.get("uploader") or info.get("channel") or "Unknown",
            "Title": info.get("title") or "Unknown",
            "Views": info.get("view_count", 0),
        }

        if not output_file.is_file() or output_file.stat().st_size == 0:
            raise RuntimeError("YouTube download did not create a valid input_vid.mp4 file.")

        completed = True
        logger.info("YouTube download: completed successfully")
        return metadata
    except Exception as exc:
        message = str(exc)
        lowered_message = message.lower()
        if "bot" in lowered_message or "detected" in lowered_message or "po token" in lowered_message:
            raise ValueError(
                "YouTube rejected the download because of automated-traffic or PO Token protection. "
                "No video was downloaded."
            ) from exc
        if isinstance(exc, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"YouTube download failed: {message}") from exc
    finally:
        if not completed and not output_existed and output_file.exists():
            try:
                output_file.unlink()
            except OSError:
                logger.warning("Could not remove incomplete output file: %s", output_file)

# Function to extract frames from a video and save as images
def video_to_images(video_path, output_folder):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(video_path)
    try:
        clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=0.5)
    finally:
        clip.close()
    if not get_frame_paths(output_folder):
        raise ValueError("Frame extraction completed but no frames were created.")

# Function to extract audio from a video
def video_to_audio(video_path, output_audio_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    clip = VideoFileClip(video_path)
    try:
        audio = clip.audio
        if audio is None:
            return False
        audio.write_audiofile(output_audio_path)
        return True
    finally:
        clip.close()

# Function to convert audio to text
def audio_to_text(audio_path):
    if not os.path.isfile(audio_path):
        return ""
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)
    with audio as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            text = "Audio not recognized"
        except sr.RequestError as e:
            text = f"Error: {e}"
    return text

# Function to plot images
def plot_images(image_paths):
    st.write("Frames used in the response:")
    images_shown = 0
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            st.image(image, caption=f"Frame {images_shown + 1}", width="content")
            images_shown += 1
            if images_shown >= 7:
                break

# Retrieve query results
def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)
    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            file_path = res_node.node.metadata.get("file_path")
            if file_path:
                retrieved_image.append(file_path)
        else:
            retrieved_text.append(res_node.text)
    return retrieved_image, retrieved_text

# Initial state management
if "retriever_engine" not in st.session_state:
    st.session_state.retriever_engine = None
    st.session_state.metadata_vid = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 2: Input YouTube URL
st.markdown(f"### 🎥 Step {video_step_number}: Enter the YouTube Video URL")
video_url = st.text_input("Enter the YouTube video link:", key="video_input", placeholder="https://www.youtube.com/watch?v=example")

# Process the video on button click
if video_url and st.session_state.retriever_engine is None:
    if st.button("🚀 Process Video"):
        try:
            with st.spinner("Processing video... This might take a while :( "):
                video_id = get_youtube_video_id(video_url)
                if not video_id:
                    raise ValueError("Could not identify the YouTube video ID from the URL.")

                cache_paths = get_cache_paths(video_id)
                Path(cache_paths["video_dir"]).mkdir(parents=True, exist_ok=True)
                Path(cache_paths["data_dir"]).mkdir(parents=True, exist_ok=True)

                transcript_cached = os.path.isfile(cache_paths["text_path"])
                frames_cached = bool(get_frame_paths(cache_paths["data_dir"]))
                index_cached = (
                    transcript_cached
                    and frames_cached
                    and cached_index_exists(cache_paths["text_table"], cache_paths["image_table"])
                )

                if os.path.isfile(cache_paths["metadata_path"]):
                    with open(cache_paths["metadata_path"], "r") as file:
                        st.session_state.metadata_vid = json.load(file)
                elif index_cached:
                    st.session_state.metadata_vid = {"Video ID": video_id}
                else:
                    st.session_state.metadata_vid = download_video(video_url, cache_paths["video_dir"])
                    with open(cache_paths["metadata_path"], "w") as file:
                        json.dump(st.session_state.metadata_vid, file)

                if (not transcript_cached or not frames_cached) and not os.path.isfile(cache_paths["video_path"]):
                    st.session_state.metadata_vid = download_video(video_url, cache_paths["video_dir"])
                    with open(cache_paths["metadata_path"], "w") as file:
                        json.dump(st.session_state.metadata_vid, file)

                if not frames_cached:
                    video_to_images(cache_paths["video_path"], cache_paths["data_dir"])

                if not transcript_cached:
                    audio_created = video_to_audio(cache_paths["video_path"], cache_paths["audio_path"])
                    text_data = audio_to_text(cache_paths["audio_path"]) if audio_created else "No audio track found."
                    with open(cache_paths["text_path"], "w") as file:
                        file.write(text_data)
                    if os.path.isfile(cache_paths["audio_path"]):
                        os.remove(cache_paths["audio_path"])

                st.session_state.retriever_engine = create_retriever(video_id, cached_index=index_cached)

                st.success("✅ Video processing completed! You can now ask questions.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Step 3: Ask questions
if st.session_state.retriever_engine:
    st.markdown(f"### 💬 Step {question_step_number}: Ask a Question About the Video")
    user_query = st.text_input("Ask a question about the video:", key="question_input")
    for chat_item in st.session_state.chat_history:
        st.write("### 🙋 Question")
        st.write(chat_item["question"])
        st.write("### 📝 Extracted Text Context")
        st.text(chat_item["context"])
        plot_images(chat_item["images"])
        st.write("### 🤖 LLM Response")
        st.write(chat_item["answer"])
    
    if st.button("🔍 Submit Query") and user_query:
        try:
            if not openai.api_key:
                raise ValueError("OpenAI API key is missing. Add OPENAI_API_KEY to .env or enter it in the app.")

            img, txt = retrieve(retriever_engine=st.session_state.retriever_engine, query_str=user_query)
            context_str = "".join(txt)
            valid_images = [img_path for img_path in img if os.path.isfile(img_path)][:3]
            if not context_str and not valid_images:
                raise ValueError("No relevant text or image context was retrieved for this question.")

            # Display metadata and context
            st.write("### 📄 Video Metadata:")
            st.json(st.session_state.metadata_vid)

            st.write("### 📝 Extracted Text Context:")
            st.text(context_str)

            # Display the frames used in response
            plot_images(img)

            # Create the LLM prompt
            qa_tmpl_str = (
                "Given the provided information, including relevant images and retrieved context from the video, "
                "accurately and precisely answer the query without any additional prior knowledge.\n"
                "---------------------\n"
                "Context: {context_str}\n"
                "Metadata for video: {metadata_str}\n"
                "---------------------\n"
                "Query: {query_str}\n"
                "Answer: "
            )

            prompt = qa_tmpl_str.format(
                context_str=context_str, query_str=user_query, metadata_str=json.dumps(st.session_state.metadata_vid)
            )
            content = [{"type": "input_text", "text": prompt}]
            for img_path in valid_images:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": image_path_to_data_url(img_path),
                    }
                )

            # Interact with LLM
            client = openai.OpenAI(api_key=openai.api_key)
            response_1 = client.responses.create(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": content}],
                max_output_tokens=1500,
            )

            # Display the response
            st.markdown("### 🤖 LLM Response:")
            st.write(response_1.output_text)
            st.session_state.chat_history.append(
                {
                    "question": user_query,
                    "context": context_str,
                    "images": valid_images,
                    "answer": response_1.output_text,
                }
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Step 4: Process new video
if st.button("🔄 Process New Video"):
    # Reset session state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.write("App reset. Please enter a new video link.")
