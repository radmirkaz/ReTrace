import asyncio
import time
import threading
from typing import AsyncGenerator, Tuple
import cv2
import numpy as np
import supervision as sv
from loguru import logger

class VideoProcessor:
    """Handles video processing with frame streaming and timeout management."""
    
    def __init__(
        self,
        device: str = "cpu",
        cuda_timeout: float = 900.0,  # 15 minutes for CUDA
        mps_timeout: float = 1800.0,   # 30 minutes for MPS
        cpu_timeout: float = 10800.0,  # 3 hours for CPU
    ):
        self.device = device
        # Set timeout based on device type
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:  # cpu or any other device
            self.processing_timeout = cpu_timeout
            
        logger.info(
            f"Video processor initialized with {device} device, timeout: {self.processing_timeout:.1f}s"
        )
    
    async def stream_frames(
        self,
        video_path: str
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Stream video frames asynchronously with timeout protection.
        All frames are processed regardless of the compute device.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            Tuple[int, np.ndarray]: Frame number and frame data
        """
        start_time = time.time()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        loop = asyncio.get_running_loop()
        frame_queue: asyncio.Queue = asyncio.Queue()  # Unbounded queue
        
        def read_frames() -> None:
            frame_count = 0
            try:
                while True:
                    if time.time() - start_time > self.processing_timeout:
                        logger.warning(
                            f"Video processing timeout reached after {time.time() - start_time:.1f}s "
                            f"on {self.device} device ({frame_count} frames processed)"
                        )
                        break
                    ret, frame = cap.read()
                    if not ret:
                        logger.info(
                            f"Completed processing {frame_count} frames in {time.time() - start_time:.1f}s on {self.device} device"
                        )
                        break
                    # Safely schedule putting the frame onto the asyncio queue
                    loop.call_soon_threadsafe(frame_queue.put_nowait, (frame_count, frame))
                    frame_count += 1
            except Exception as e:
                logger.error(f"Error in frame reading thread: {e}")
            finally:
                cap.release()
                # Put a sentinel value to indicate completion
                loop.call_soon_threadsafe(frame_queue.put_nowait, None)
        
        # Launch the blocking frame reader in a background thread
        threading.Thread(target=read_frames, daemon=True).start()
        
        # Yield frames from the queue as they become available
        while True:
            item = await frame_queue.get()
            if item is None:
                break
            yield item

    @staticmethod
    def get_video_info(video_path: str) -> sv.VideoInfo:
        """Get video information using supervision."""
        return sv.VideoInfo.from_video_path(video_path)
    
    @staticmethod
    async def ensure_video_readable(video_path: str, timeout: float = 5.0) -> bool:
        """
        Check if a video is readable within a timeout period.
        
        Args:
            video_path: Path to the video file.
            timeout: Maximum time to wait for the check.
            
        Returns:
            bool: True if the video is readable, False otherwise.
        """
        try:
            # Run the synchronous check in a separate thread without blocking the event loop
            return await asyncio.wait_for(
                asyncio.to_thread(VideoProcessor._check_video, video_path),
                timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout while checking video readability: {video_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking video readability: {e}")
            return False

    @staticmethod
    def _check_video(video_path: str) -> bool:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return ret