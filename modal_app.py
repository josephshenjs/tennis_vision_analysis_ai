"""New court/R2 analysis on the existing raccoon-vision/api Modal address."""
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parent
app=modal.App('raccoon-vision')
secret=modal.Secret.from_name('rally-analysis-v2')
base=(modal.Image.debian_slim(python_version='3.11')
      .pip_install_from_requirements(str(ROOT/'requirements-api.txt'))
      .pip_install('numpy==1.26.4','opencv-python-headless==4.10.0.84')
      .env({'PYTHONPATH':'/app','PYTHONUNBUFFERED':'1','TENNIS_FFMPEG':'/usr/bin/ffmpeg'}))
def sources(image):
    for file in sorted(ROOT.glob('*.py')):
        if file.name == 'modal_app.py': continue
        image=image.add_local_file(file,'/app/rally_analysis/'+file.name,copy=True)
    return image.add_local_dir(ROOT/'compat','/app/rally_analysis/compat',copy=True,
                              ignore=['**/__pycache__/**','**/*.pyc'])

gpu_image=(base.apt_install('ffmpeg','libgl1','libglib2.0-0','libgomp1')
    .pip_install('numpy==1.26.4','setuptools<81','wheel')
    .pip_install('torch==2.1.0','torchvision==0.16.0',index_url='https://download.pytorch.org/whl/cu121')
    .pip_install('mmcv==2.1.0',find_links='https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html')
    .pip_install('chumpy==0.70',extra_options='--no-build-isolation')
    .pip_install('numpy==1.26.4','opencv-python==4.10.0.84','mmengine==0.10.7','mmdet==3.3.0','mmpose==1.3.2',
                 'pandas==2.2.0','pillow==10.2.0','catboost==1.2.8','opencv-python-headless==4.10.0.84')
)


base=sources(base)
gpu_image=(sources(gpu_image)
    .add_local_dir(ROOT/'RacketVision','/app/rally_analysis/RacketVision',ignore=['**/__pycache__/**','**/*.pyc'])
    .add_local_dir(ROOT/'models','/app/rally_analysis/models'))

def settings():
    import os
    from rally_analysis.api import Settings
    value=os.environ.get('RALLYCLIP_GOOGLE_CREDENTIALS_JSON')
    if value:
        target=Path('/tmp/analysis-google.json');target.write_text(value);target.chmod(0o600)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']=str(target)
    return Settings.from_env()


@app.cls(image=gpu_image,gpu='A10',cpu=4,memory=16384,max_containers=1,
         min_containers=0,scaledown_window=120,timeout=7200,secrets=[secret])
@modal.concurrent(max_inputs=2)
class GPU:
    @modal.enter()
    def load(self):
        import threading
        from rally_analysis.api import Service,Store,R2,ResidentAnalyzer
        config=settings()
        self.service=Service.__new__(Service)
        self.service.settings=config
        self.service.store=Store(config)
        self.service.storage=R2(config)
        self.service.analyzer=ResidentAnalyzer()
        self.service.slots=threading.BoundedSemaphore(config.max_pending)

    @modal.method()
    def process(self,job_id,key,calibration):
        self.service.slots.acquire()
        self.service.process(job_id,key,calibration)

    @modal.method()
    def check(self):
        import torch
        import mmcv.ops
        from rally_analysis.engine import Engine
        analyzer=self.service.analyzer
        def initialize():
            engine=Engine('cuda:0',False,2)
            engine.ball_model();engine.racket_models()
            return engine
        with analyzer.init_lock:
            if 'engine' not in analyzer.resources:
                analyzer.resources['engine']=analyzer.resources['gpu'].submit(initialize).result()
        return {'cuda':torch.cuda.get_device_name(0),'model_load_counts':analyzer.resources['engine'].load_counts}

    @modal.exit()
    def close(self):
        self.service.analyzer.close()


@app.function(image=base,cpu=.25,memory=1024,min_containers=1,max_containers=1,
              timeout=3600,secrets=[secret])
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def api():
    from rally_analysis.api import Service,create_app
    class RemoteService(Service):
        def process(self,job_id,key,calibration):
            try: self.worker.process.remote(job_id,key,calibration)
            except Exception:
                self.store.update('analyses',job_id,{'state':'failed','failed_stage':'gpu_dispatch',
                    'error':'GPU dispatch failed; retry the clip.'})
            finally: self.slots.release()
    service=RemoteService.__new__(RemoteService)
    service.worker=GPU()
    Service.__init__(service,settings())
    result=create_app(service=service)
    # Existing frontend connection indicators can keep their health endpoint.
    result.add_api_route('/api/health',lambda:{'status':'ok','api_version':'2','automatic_analysis':True,
                                             'max_upload_bytes':100_000_000},methods=['GET'])
    return result


@app.local_entrypoint()
def check_gpu():
    print(GPU().check.remote())
