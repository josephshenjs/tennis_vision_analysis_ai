import copy
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from rally_analysis.api import Settings, Service, create_app, object_key


class MemoryStore:
    def __init__(self):
        self.records = {}

    def get(self, kind, key):
        return copy.deepcopy(self.records.get((kind, key)))

    def put(self, kind, key, data):
        self.records[kind, key] = copy.deepcopy(data)

    def update(self, kind, key, data):
        self.records[kind, key].update(data)


class Storage:
    def __init__(self):
        self.uploads = []
        self.downloads = []

    def download(self, key, target):
        self.downloads.append(key)
        target.write_bytes(b'input video')

    def upload(self, source, key):
        self.uploads.append((source.read_bytes(), key))

    def url(self, key):
        return 'https://download.example/' + key


def analyze(video, calibration, output):
    assert video.read_bytes() == b'input video'
    data = json.loads(calibration.read_text())
    assert data['corner_order'] == ['far_left', 'far_right', 'near_right', 'near_left']
    output.mkdir()
    target = output / 'analysis.mp4'
    target.write_bytes(b'analyzed video')
    return target


class APITests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings('project', 'test-db', 'test', 'videos',
            'https://account.r2.cloudflarestorage.com', 'key', 'secret', ('test-token',))
        self.store = MemoryStore()
        self.storage = Storage()
        self.service = Service(self.settings, self.store, self.storage, analyze)
        self.client = TestClient(create_app(service=self.service))
        self.client.__enter__()
        self.headers = {'Authorization': 'Bearer test-token'}
        self.court = {'image_width': 1920, 'image_height': 1080,
                      'corners_px': [[700, 300], [1200, 300], [1700, 900], [200, 900]]}
        self.url = self.settings.endpoint + '/videos/clips/point%2001.mp4?signature=ignored'

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def save(self, headers=None):
        return self.client.put('/courts/camera1', json=self.court,
                               headers=headers or self.headers)

    def wait_for_job(self, job_id):
        for _ in range(200):
            result = self.client.get('/analyses/' + job_id, headers=self.headers)
            if result.json()['state'] in ('complete', 'failed'):
                return result.json()
            time.sleep(.01)
        self.fail('Job did not finish')

    def test_auth_and_corner_validation(self):
        self.assertEqual(self.client.put('/courts/camera1', json=self.court).status_code, 401)
        for corners in ([[0, 0]], [[0, 0]] * 4,
                        [[700, 300], [1200, 300], [1920, 900], [200, 900]]):
            body = dict(self.court, corners_px=corners)
            self.assertEqual(self.client.put('/courts/camera1', json=body,
                                            headers=self.headers).status_code, 422)
        self.assertEqual(self.save().status_code, 200)
        saved = self.store.get('courts', 'camera1')
        self.assertIsInstance(saved['corners'][0], dict)
        self.assertIsInstance(saved['calibration_json'], str)
        self.assertEqual(self.client.get('/courts/camera1', headers=self.headers).json()
                         ['calibration']['corners_px'], self.court['corners_px'])

    def test_download_analysis_upload_and_result(self):
        self.save()
        response = self.client.post('/analyses', headers=self.headers,
            json={'court_id': 'camera1', 'file_url': self.url})
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()['job_id']
        result = self.wait_for_job(job_id)
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(self.storage.downloads, ['clips/point 01.mp4'])
        self.assertEqual(self.storage.uploads, [(b'analyzed video', f'analyzed/{job_id}/analysis.mp4')])
        self.assertTrue(result['output_url'].startswith('https://download.example/'))
        self.assertNotIn('owner', result)

    def test_analysis_failure_is_persisted_without_upload(self):
        self.save()
        def fail(*args):
            raise RuntimeError('private credentials')
        self.service.analyzer = fail
        with self.assertLogs('rally_analysis.api', level='ERROR'):
            response = self.client.post('/analyses', headers=self.headers,
                json={'court_id': 'camera1', 'file_url': self.url})
            result = self.wait_for_job(response.json()['job_id'])
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['failed_stage'], 'analyzing')
        self.assertNotIn('private credentials', json.dumps(result))
        self.assertEqual(self.storage.uploads, [])

    def test_upload_failure_is_not_reported_as_complete(self):
        self.save()
        with patch.object(self.storage, 'upload', side_effect=RuntimeError('upload failed')):
            with self.assertLogs('rally_analysis.api', level='ERROR'):
                response = self.client.post('/analyses', headers=self.headers,
                    json={'court_id': 'camera1', 'file_url': self.url})
                result = self.wait_for_job(response.json()['job_id'])
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['failed_stage'], 'uploading')
        self.assertNotIn('output_url', result)

    def test_rejects_foreign_urls_and_missing_calibration(self):
        for url in ('http://localhost/a.mp4', 'https://evil.example/video.mp4',
                    self.settings.endpoint + '/another-bucket/a.mp4',
                    self.settings.endpoint + '/videos/%2e%2e/a.mp4'):
            with self.assertRaises(ValueError):
                object_key(url, self.settings)
        result = self.client.post('/analyses', headers=self.headers,
            json={'court_id': 'missing', 'file_url': self.url})
        self.assertEqual(result.status_code, 404)

    def test_queue_limit_and_gpu_serialization(self):
        self.save()
        entered = threading.Event()
        release = threading.Event()
        def waiting(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test timeout')
            return analyze(*args)
        self.service.analyzer = waiting
        self.service.slots = threading.BoundedSemaphore(1)
        try:
            first = self.client.post('/analyses', headers=self.headers,
                json={'court_id': 'camera1', 'file_url': self.url})
            self.assertTrue(entered.wait(2))
            second = self.client.post('/analyses', headers=self.headers,
                json={'court_id': 'camera1', 'file_url': self.url})
            self.assertEqual(second.status_code, 429)
        finally:
            release.set()
        self.assertEqual(self.wait_for_job(first.json()['job_id'])['state'], 'complete')

    def test_firebase_users_have_separate_courts_and_jobs(self):
        with patch('firebase_admin.auth.verify_id_token', return_value={'uid': 'alice'}), \
             patch('firebase_admin.get_app', return_value=object()):
            alice = {'Authorization': 'Bearer firebase-alice'}
            self.assertEqual(self.save(alice).status_code, 200)
            response = self.client.post('/analyses', headers=alice,
                json={'court_id': 'camera1', 'file_url': self.url})
            self.assertEqual(response.status_code, 202)
            job_id = response.json()['job_id']
        with patch('firebase_admin.auth.verify_id_token', return_value={'uid': 'bob'}), \
             patch('firebase_admin.get_app', return_value=object()):
            bob = {'Authorization': 'Bearer firebase-bob'}
            self.assertEqual(self.client.get('/courts/camera1', headers=bob).status_code, 404)
            self.assertEqual(self.client.get('/analyses/' + job_id, headers=bob).status_code, 404)

    def test_firestore_outage_returns_service_unavailable(self):
        with patch.object(self.store, 'put', side_effect=RuntimeError('offline')):
            with self.assertLogs('rally_analysis.api', level='ERROR'):
                self.assertEqual(self.save().status_code, 503)


if __name__ == '__main__':
    unittest.main()
