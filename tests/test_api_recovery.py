import json
import unittest
import test_api as base
from test_api import MemoryStore, Storage, analyze
from rally_analysis.api import Service


class RecoveryTests(unittest.TestCase):
    setUp=base.APITests.setUp
    tearDown=base.APITests.tearDown
    save=base.APITests.save
    wait_for_job=base.APITests.wait_for_job
    def test_request_id_reuses_job_and_rejects_changed_input(self):
        self.save()
        body={'court_id':'camera1','file_url':self.url,'request_id':'a'*64}
        first=self.client.post('/analyses',headers=self.headers,json=body)
        second=self.client.post('/analyses',headers=self.headers,json=body)
        self.assertEqual(first.json()['job_id'],second.json()['job_id'])
        self.wait_for_job(first.json()['job_id'])
        self.assertEqual(len(self.storage.downloads),1)
        body['file_url']=self.settings.endpoint+'/videos/other.mp4'
        self.assertEqual(self.client.post('/analyses',headers=self.headers,json=body).status_code,409)

    def test_restart_resumes_saved_calibration_snapshot(self):
        self.save()
        calibration=self.store.get('courts','camera1')['calibration_json']
        class Durable(MemoryStore):
            def pending(self):return [self.get('analyses','b'*32)]
        store=Durable()
        store.put('analyses','b'*32,{'job_id':'b'*32,'input_key':'input.mp4',
                  'state':'analyzing','calibration_json':calibration})
        storage=Storage()
        service=Service(self.settings,store,storage,analyze)
        service.close()
        self.assertEqual(store.get('analyses','b'*32)['state'],'complete')
        self.assertEqual(len(storage.downloads),1)
