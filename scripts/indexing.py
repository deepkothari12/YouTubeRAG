from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# video_id = "Gfr50f6ZBvo"

def indexing_chunks(video_id, languages=["en"] , chunk_size = 2000 , chunk_overlap = 500 ):
    try:
        
        transcript_raw = YouTubeTranscriptApi().fetch(video_id=video_id, languages = languages)
        transcript_text = " ".join([item.text for item in transcript_raw])

    except TranscriptsDisabled:
        raise RuntimeError("No captions available for the video")
        
    except NoTranscriptFound:
        raise RuntimeError("No transcripts found")
        
    except VideoUnavailable:
        raise RuntimeError("Video unavailable")
        
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([transcript_text])

    logging.info(f"Transcript split into {len(chunks)} chunks for video {video_id}")
    return chunks
