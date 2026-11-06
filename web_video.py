"""Encode annotated BGR frames directly to H.264, retaining source audio."""
import os
import shutil
import subprocess


class VideoWriter:
    def __init__(self, source, target, info):
        executable = os.getenv('TENNIS_FFMPEG') or shutil.which('ffmpeg')
        if not executable: raise RuntimeError('FFmpeg required; configure TENNIS_FFMPEG')
        self.process = subprocess.Popen([executable, '-nostdin', '-v', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f"{info['width']}x{info['height']}",
            '-r', str(info['fps']), '-i', 'pipe:0', '-i', str(source),
            '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ar', '48000', '-ac', '2',
            '-shortest', '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

    def isOpened(self): return self.process.poll() is None

    def write(self, frame): self.process.stdin.write(frame.tobytes())

    def release(self):
        try: self.process.stdin.close()
        except BrokenPipeError: pass
        if self.process.wait() != 0: raise RuntimeError('H.264 video encoding failed')
