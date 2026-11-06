"""Explicit cloud smoke test: real Firestore, R2 and CUDA, with test-data cleanup.

Run: runtime/python.exe tools/verify_api_cloud.py
Creates a synthetic 16-frame clip, submits it through the API, decodes the
uploaded result, then removes only the test's own documents and R2 objects.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run', ROOT / 'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

import cv2
import numpy as np
from fastapi.testclient import TestClient
from rally_analysis.api import Settings, Service, create_app


def main():
    settings = Settings.from_env()
    service = Service(settings)
    test_id = 'api_smoke_' + uuid4().hex
    source_key = f'api-smoke/{test_id}/input.mp4'
    job_id = None
    headers = {'Authorization': 'Bearer ' + settings.tokens[0]} if settings.tokens else {}
    if settings.require_user_auth and not headers:
        raise RuntimeError('Cloud verification requires a configured backend API token.')
    try:
        with tempfile.TemporaryDirectory(prefix='rally-api-smoke-') as temp:
            video = Path(temp) / 'input.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
            if not writer.isOpened():
                raise RuntimeError('Cannot create smoke-test video.')
            try:
                for _ in range(16):
                    image = np.full((480, 640, 3), (90, 95, 45), dtype=np.uint8)
                    cv2.polylines(image, [np.array([[220, 150], [420, 150], [580, 430], [60, 430]])],
                                  True, (255, 255, 255), 2)
                    writer.write(image)
            finally:
                writer.release()
            service.storage.upload(video, source_key)
            with TestClient(create_app(service=service)) as client:
                response = client.put('/courts/' + test_id, headers=headers, json={
                    'image_width': 640, 'image_height': 480,
                    'corners_px': [[220, 150], [420, 150], [580, 430], [60, 430]]})
                response.raise_for_status()
                print('Court saved to configured Firestore database.', flush=True)
                response = client.post('/analyses', headers=headers, json={
                    'court_id': test_id,
                    'file_url': settings.endpoint + '/' + settings.bucket + '/' + source_key})
                response.raise_for_status()
                job_id = response.json()['job_id']
                previous = None
                while True:
                    response = client.get('/analyses/' + job_id, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    if result['state'] != previous:
                        print('Analysis state:', result['state'], flush=True)
                        previous = result['state']
                    if result['state'] in ('complete', 'failed'):
                        break
                    time.sleep(1)
                if result['state'] != 'complete':
                    raise RuntimeError(result.get('error', 'Analysis failed'))
                target = Path(temp) / 'downloaded-analysis.mp4'
                service.storage.download(result['output_key'], target)
                cap = cv2.VideoCapture(str(target))
                count = 0
                try:
                    while cap.read()[0]:
                        count += 1
                finally:
                    cap.release()
                if count != 16:
                    raise RuntimeError(f'Expected 16 output frames, got {count}.')
                print(json.dumps({'result': 'passed', 'decoded_frames': count,
                                  'firestore_database': settings.database}), flush=True)
    finally:
        service.close()
        service.storage.client.delete_object(Bucket=settings.bucket, Key=source_key)
        service.store.ref('courts', test_id).delete()
        if job_id:
            service.storage.client.delete_object(Bucket=settings.bucket,
                                                 Key=f'analyzed/{job_id}/analysis.mp4')
            service.store.ref('analyses', job_id).delete()
        print('Removed cloud smoke-test artifacts.', flush=True)


if __name__ == '__main__':
    main()
