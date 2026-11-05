"""Bounded video IO: decode/preprocess ahead without dropping frames."""
from contextlib import contextmanager
import queue
import threading
import cv2
import numpy as np


def metadata(path):
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {path}')
        width,height=int(cap.get(3)),int(cap.get(4)); fps=float(cap.get(5)); count=int(cap.get(7))
        if min(width,height)<=0 or not np.isfinite(fps) or fps<=0:
            raise ValueError(f'Invalid video geometry/FPS: {path}')
        return dict(width=width,height=height,fps=fps,declared_frames=count)
    finally: cap.release()


def frames(path,limit=0,selected=None):
    cap=cv2.VideoCapture(str(path)); index=0
    if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {path}')
    selected_set=set(selected) if selected is not None else None
    last=max(selected_set) if selected_set else None
    try:
        while not limit or index<limit:
            if selected_set is not None and index not in selected_set:
                ok=cap.grab(); image=None
            else:
                ok,image=cap.read()
            if not ok: break
            if image is not None: yield index,image
            index+=1
            if last is not None and index>last: break
    finally: cap.release()


@contextmanager
def prefetch(iterable,capacity=2):
    if capacity<1: raise ValueError('Prefetch capacity must be positive.')
    tasks=queue.Queue(maxsize=capacity); stopped=threading.Event()
    def put(item):
        while not stopped.is_set():
            try: tasks.put(item,timeout=.1); return True
            except queue.Full: pass
        return False
    def produce():
        try:
            for value in iterable:
                if not put(('data',value)): return
            put(('end',None))
        except BaseException as error:
            put(('error',error))
        finally:
            close=getattr(iterable,'close',None)
            if close is not None: close()
    thread=threading.Thread(target=produce,name='rally-decode',daemon=True)
    thread.start()
    def consume():
        while True:
            kind,value=tasks.get()
            if kind=='end': return
            if kind=='error': raise value
            yield value
    try: yield consume()
    finally:
        stopped.set(); thread.join(timeout=5)
