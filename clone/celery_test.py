from celery import Celery, task
import time
app = Celery('test')
app.conf.update(
    BROKER_URL='redis://localhost',
    CELERY_TASK_SERIALIZER='json',
    CELERY_ACCEPT_CONTENT=['json'],
    CELERYBEAT_SCHEDULE={
        'test':{
            'task':'celery_test.test'
        }
    }
)
 
@app.task(name='celery_test.test')
def test():
	for i in range(10):
		print(i)
		time.sleep(1)