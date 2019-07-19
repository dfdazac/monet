import os
from sacred.observers import MongoObserver


def add_observers(experiment):
    uri = os.environ.get('MLAB_URI')
    database = os.environ.get('MLAB_DB')
    if all([uri, database]):
        experiment.observers.append(MongoObserver.create(uri, database))
