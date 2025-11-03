from neurosurfer.config import config

# Socket Path
bind = f"{config.app.host_ip}:{config.app.host_port}"

# Worker Options
# import multiprocessing
# workers = multiprocessing.cpu_count() * 2 + 1
workers = config.app.workers
worker_class = 'uvicorn.workers.UvicornWorker'
preload_app = False  # Avoid re-running lifespan unnecessarily

# Workers silent for more than this many seconds are killed and restarted
timeout = config.app.worker_timeout

# The maximum number of requests a worker will process before restarting.
# max_requests = 100

# After receiving a restart signal, workers have this much time to finish serving requests.
# graceful_timeout = 30

# Logging Options
# loglevel = config.app.logs_level
# accesslog = '/home/demo/fastapi_demo/access_log'
# errorlog =  '/home/demo/fastapi_demo/error_log'
