"""FastAPI adapter for court calibration and authenticated R2 analysis jobs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .paths import ROOT

log = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Settings:
    project: str
    database: str
    prefix: str
    bucket: str
    endpoint: str
    access_key: str
    secret_key: str
    tokens: tuple
    require_user_auth: bool = True
    public_base: str = ''
    max_pending: int = 8
    worker_namespace: str = 'local'

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / '.env.local', override=False)

        def required(name):
            value = os.getenv(name, '').strip()
            if not value:
                raise ValueError(f'Missing configuration: {name}')
            return value

        return cls(
            project=required('RALLYCLIP_FIREBASE_PROJECT'),
            database=os.getenv('RALLYCLIP_FIRESTORE_DATABASE', '(default)'),
            prefix=os.getenv('RALLYCLIP_FIRESTORE_PREFIX', 'rallyclip'),
            bucket=required('RALLYCLIP_ARTIFACT_BUCKET'),
            endpoint=required('RALLYCLIP_R2_ENDPOINT').rstrip('/'),
            access_key=required('RALLYCLIP_R2_ACCESS_KEY_ID'),
            secret_key=required('RALLYCLIP_R2_SECRET_ACCESS_KEY'),
            tokens=tuple(v for v in (os.getenv('RALLYCLIP_ANALYSIS_API_TOKEN'),
                                    os.getenv('RALLYCLIP_API_TOKEN')) if v),
            require_user_auth=os.getenv('RALLYCLIP_REQUIRE_USER_AUTH', '1').lower() in ('1', 'true', 'yes'),
            public_base=os.getenv('RALLYCLIP_R2_PUBLIC_BASE_URL', '').rstrip('/'),
            worker_namespace=os.getenv('RALLYCLIP_ANALYSIS_WORKER_NAMESPACE', 'local'))


class CourtInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    image_width: int = Field(gt=0, le=32768)
    image_height: int = Field(gt=0, le=32768)
    corners_px: list[tuple[float, float]] = Field(min_length=4, max_length=4,
        description='Original pixels: far-left, far-right, near-right, near-left doubles baseline corners.')


class AnalysisInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    court_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,128}$')
    file_url: str = Field(min_length=1, max_length=8192,
        description='HTTPS object URL under RALLYCLIP_R2_ENDPOINT/<bucket>/, or configured public base URL.')


class Store:
    def __init__(self, settings):
        from google.cloud import firestore
        self.client = firestore.Client(project=settings.project, database=settings.database)
        self.prefix = settings.prefix
        self.worker_namespace = settings.worker_namespace

    def ref(self, kind, identifier):
        return self.client.collection(f'{self.prefix}_{kind}').document(identifier)

    def get(self, kind, identifier):
        return self.ref(kind, identifier).get().to_dict()

    def pending(self):
        from google.cloud.firestore_v1.base_query import FieldFilter
        return [s.to_dict() for s in self.client.collection(f'{self.prefix}_analyses').where(
            filter=FieldFilter('state', 'in', ['queued','downloading','analyzing','uploading'])).stream()
                if s.to_dict().get('worker_namespace', 'local') == self.worker_namespace]

    def put(self, kind, identifier, data):
        self.ref(kind, identifier).set(data)

    def update(self, kind, identifier, data):
        self.ref(kind, identifier).update(data)


def object_key(file_url, settings):
    """Resolve configured R2 URLs; never fetch arbitrary URLs or redirects."""
    url = urlsplit(file_url)
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.fragment:
        raise ValueError('file_url must be an HTTPS R2 object URL.')
    bases = [settings.endpoint + '/' + settings.bucket]
    if settings.public_base:
        bases.append(settings.public_base)
    for base in bases:
        allowed = urlsplit(base)
        prefix = allowed.path.rstrip('/') + '/'
        if url.netloc.lower() == allowed.netloc.lower() and url.path.startswith(prefix):
            key = unquote(url.path[len(prefix):])
            if key and not any(part in ('.', '..') for part in key.split('/')) and '\x00' not in key:
                if Path(key).suffix.lower() not in ('.mp4', '.mov', '.mkv', '.avi', '.webm'):
                    raise ValueError('Unsupported video extension.')
                return key
    raise ValueError('file_url must reference the configured R2 bucket or public base URL.')


class R2:
    def __init__(self, settings):
        import boto3
        from botocore.config import Config
        self.settings = settings
        self.client = boto3.client('s3', endpoint_url=settings.endpoint,
            aws_access_key_id=settings.access_key, aws_secret_access_key=settings.secret_key,
            region_name='auto', config=Config(signature_version='s3v4',
                connect_timeout=15, read_timeout=120, retries={'max_attempts': 3},
                s3={'addressing_style': 'path'}))

    def download(self, key, target):
        self.client.download_file(self.settings.bucket, key, str(target))

    def upload(self, source, key):
        self.client.upload_file(str(source), self.settings.bucket, key,
                                ExtraArgs={'ContentType': 'video/mp4'})

    def url(self, key):
        return self.client.generate_presigned_url('get_object',
            Params={'Bucket': self.settings.bucket, 'Key': key}, ExpiresIn=3600)


class ResidentAnalyzer:
    """Models and CUDA executor survive across HTTP jobs; only one CUDA owner."""
    def __init__(self):
        self.init_lock = threading.Lock()
        self.resources = {'stop': threading.Event(), 'gpu': ThreadPoolExecutor(max_workers=1, thread_name_prefix='resident-cuda'),
                          'cpu': ThreadPoolExecutor(max_workers=2, thread_name_prefix='resident-cpu')}

    def __call__(self, video, calibration_path, output):
        import asyncio
        from argparse import Namespace
        from .paths import BOUNCE_CHECKPOINT
        from .pipeline import analyze_videos
        with self.init_lock:
            if 'engine' not in self.resources:
                from .engine import Engine
                self.resources['engine'] = self.resources['gpu'].submit(Engine, 'cuda:0', False, 2).result()
        args = Namespace(videos=[video], output_dir=output, court_calibration=calibration_path,
            overwrite=False, device='cuda:0', amp=False, prefetch=2, cpu_workers=2, jobs=1,
            single_window=False, far_roi=None, court_margin_m=1., ball_lift=.25,
            batch_size=4, threshold=.5, median_samples=100, max_frames=0,
            no_local_recovery=False, bounce_threshold=.55, bounce_model=BOUNCE_CHECKPOINT, no_video=False)
        if asyncio.run(analyze_videos(args, self.resources)):
            raise RuntimeError('Analysis pipeline failed; see server logs.')
        target = output / video.stem / 'analysis.mp4'
        if not target.is_file() or not target.stat().st_size: raise RuntimeError('Empty analysis output')
        return target

    def close(self):
        self.resources['stop'].set()
        self.resources['gpu'].shutdown(wait=True)
        self.resources['cpu'].shutdown(wait=True)


class Service:
    def __init__(self, settings, store=None, storage=None, analyzer=None):
        self.settings = settings
        self.store = store if store is not None else Store(settings)
        self.storage = storage if storage is not None else R2(settings)
        self.analyzer = analyzer if analyzer is not None else ResidentAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=2 if isinstance(self.analyzer, ResidentAnalyzer) else 1, thread_name_prefix='rally-api')
        self.slots = threading.BoundedSemaphore(settings.max_pending)
        self.submit_lock = threading.Lock()
        if hasattr(self.store, 'pending'):
            for job in self.store.pending():
                if not job.get('calibration_json'):
                    self.store.update('analyses', job['job_id'], {'state':'failed', 'failed_stage':'recovery', 'error':'Legacy job interrupted; resubmit.'})
                    continue
                self.submit(job['job_id'], job['input_key'], json.loads(job['calibration_json']))

    def close(self):
        self.executor.shutdown(wait=True)
        if hasattr(self.analyzer, 'close'): self.analyzer.close()

    def submit(self, job_id, key, calibration):
        if not self.slots.acquire(blocking=False):
            raise HTTPException(429, 'Analysis queue is full. Retry later.')
        try:
            self.executor.submit(self.process, job_id, key, calibration)
        except Exception:
            self.slots.release()
            raise

    def process(self, job_id, key, calibration):
        stage = 'downloading'
        try:
            self.store.update('analyses', job_id, {'state': stage, 'updated_at': now()})
            with tempfile.TemporaryDirectory(prefix='rally-analysis-') as temp:
                folder = Path(temp)
                video = folder / ('input' + Path(key).suffix.lower())
                self.storage.download(key, video)
                calibration_path = folder / 'court.json'
                calibration_path.write_text(json.dumps(calibration), encoding='utf8')
                stage = 'analyzing'
                self.store.update('analyses', job_id, {'state': stage, 'updated_at': now()})
                target = self.analyzer(video, calibration_path, folder / 'output')
                stage = 'uploading'
                self.store.update('analyses', job_id, {'state': stage, 'updated_at': now()})
                with target.open('rb') as handle: output_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
                output_size = target.stat().st_size
                result_key = f'analyzed/{job_id}/analysis.mp4'
                self.storage.upload(target, result_key)
                self.store.update('analyses', job_id, {'state': 'complete',
                    'output_key': result_key, 'output_bucket': self.settings.bucket, 'output_sha256': output_sha256, 'output_size': output_size,
                    'updated_at': now(), 'completed_at': now()})
        except Exception:
            log.exception('Analysis job %s failed during %s', job_id, stage)
            try:
                self.store.update('analyses', job_id, {'state': 'failed', 'failed_stage': stage,
                    'error': f'Job failed during {stage}. See server logs.', 'updated_at': now()})
            except Exception:
                log.exception('Could not persist failed job %s', job_id)
        finally:
            self.slots.release()


def court_identifier(court_id, principal):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', court_id):
        raise HTTPException(422, 'Invalid court_id.')
    return court_id if principal == 'service' else hashlib.sha256(
        (principal + ':' + court_id).encode()).hexdigest()


def create_app(settings=None, service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service or Service(settings or Settings.from_env())
        try:
            yield
        finally:
            import asyncio
            await asyncio.to_thread(app.state.service.close)

    app = FastAPI(title='Rally Analysis API', version='1.0.0', lifespan=lifespan)

    def current_service():
        return app.state.service

    def authenticate(authorization: str | None = Header(default=None),
                     x_api_token: str | None = Header(default=None),
                     svc: Service = Depends(current_service)):
        bearer = authorization[7:] if authorization and authorization.lower().startswith('bearer ') else ''
        # Trusted backend tokens bypass user authentication. Firebase tokens are uid-scoped.
        if any(hmac.compare_digest(provided.encode(), token.encode()) for provided in (bearer, x_api_token or '')
               for token in svc.settings.tokens):
            return 'service'
        if svc.settings.require_user_auth and bearer:
            try:
                import firebase_admin
                from firebase_admin import auth
                try:
                    firebase_app = firebase_admin.get_app('rally-analysis-api')
                except ValueError:
                    firebase_app = firebase_admin.initialize_app(
                        options={'projectId': svc.settings.project}, name='rally-analysis-api')
                decoded = auth.verify_id_token(bearer, app=firebase_app, check_revoked=True)
                return 'user:' + decoded['uid']
            except Exception:
                raise HTTPException(401, 'Invalid Firebase ID token.') from None
        if not svc.settings.require_user_auth and not svc.settings.tokens:
            return 'service'
        raise HTTPException(401, 'A valid API token or Firebase ID token is required.')

    def get_record(svc, kind, identifier, principal):
        try:
            record = svc.store.get(kind, identifier)
        except Exception:
            log.exception('Firestore read failed')
            raise HTTPException(503, 'Firestore is unavailable.') from None
        if record is None or (principal != 'service' and record.get('owner') != principal):
            raise HTTPException(404, 'Record not found.')
        return record

    @app.get('/health')
    def health():
        return {'status': 'ok', 'api_version': '2', 'resident_models': True, 'durable_recovery': True}

    @app.put('/courts/{court_id}')
    def save_court(court_id: str, body: CourtInput, principal=Depends(authenticate),
                   svc=Depends(current_service)):
        identifier = court_identifier(court_id, principal)
        from .compat.court_geometry import build_calibration
        try:
            calibration = build_calibration(body.corners_px, body.image_width, body.image_height, court_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        record = {'court_id': court_id, 'owner': principal, 'updated_at': now(),
            'image_width': body.image_width, 'image_height': body.image_height,
            'corners': [{'x': x, 'y': y} for x, y in body.corners_px],
            # Firestore forbids nested arrays; preserve the pipeline format as JSON.
            'calibration_json': json.dumps(calibration)}
        try:
            svc.store.put('courts', identifier, record)
        except Exception:
            log.exception('Firestore court save failed')
            raise HTTPException(503, 'Could not save court to Firestore.') from None
        return {'court_id': court_id, 'calibration': calibration, 'updated_at': record['updated_at']}

    @app.get('/courts/{court_id}')
    def get_court(court_id: str, principal=Depends(authenticate), svc=Depends(current_service)):
        record = get_record(svc, 'courts', court_identifier(court_id, principal), principal)
        return {'court_id': court_id, 'calibration': json.loads(record['calibration_json']),
                'updated_at': record['updated_at']}

    @app.post('/analyses', status_code=202)
    def start_analysis(body: AnalysisInput, principal=Depends(authenticate), svc=Depends(current_service)):
        try:
            key = object_key(body.file_url, svc.settings)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        record = get_record(svc, 'courts', court_identifier(body.court_id, principal), principal)
        with svc.submit_lock:
            job_id = (hashlib.sha256((principal + ':' + body.request_id).encode()).hexdigest()[:32]
                      if body.request_id else uuid4().hex)
            previous = svc.store.get('analyses', job_id)
            calibration_json = record['calibration_json']
            if previous:
                if previous['input_key'] != key or previous['court_id'] != body.court_id or previous.get('calibration_json') != calibration_json:
                    raise HTTPException(409, 'Idempotency key was used for different input or calibration.')
                return {'job_id': job_id, 'state': previous['state'], 'status_url': f'/analyses/{job_id}'}
            job = {'job_id': job_id, 'court_id': body.court_id, 'owner': principal,
                   'input_key': key, 'calibration_json': calibration_json, 'worker_namespace': svc.settings.worker_namespace,
                   'state': 'queued', 'created_at': now(), 'updated_at': now()}
            # Check capacity before storing the durable job. Serial submission prevents races.
            if not svc.slots.acquire(blocking=False): raise HTTPException(429, 'Analysis queue is full. Retry later.')
            try:
                svc.store.put('analyses', job_id, job)
                svc.executor.submit(svc.process, job_id, key, json.loads(calibration_json))
            except Exception:
                svc.slots.release()
                log.exception('Could not submit analysis job')
                raise HTTPException(503, 'Could not submit analysis job.') from None
            return {'job_id': job_id, 'state': 'queued', 'status_url': f'/analyses/{job_id}'}

    @app.get('/analyses/{job_id}')
    def get_analysis(job_id: str, principal=Depends(authenticate), svc=Depends(current_service)):
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise HTTPException(404, 'Record not found.')
        record = dict(get_record(svc, 'analyses', job_id, principal))
        record.pop('owner', None)
        record.pop('calibration_json', None)
        if record['state'] == 'complete':
            record['output_url'] = svc.storage.url(record['output_key'])
            record['output_url_expires_in'] = 3600
        return record

    return app


app = create_app()
