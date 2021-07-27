import os


__all__ = ['automl', 'dataset', 'ml_algo', 'pipelines', 'image',
           'reader', 'transformers', 'validation', 'text', 'tasks',
           'utils', 'addons', 'report']

if os.getenv('DOCUMENTATION_ENV') is None:
    try:
        import importlib.metadata as importlib_metadata
    except ModuleNotFoundError:
        import importlib_metadata

    __version__ = importlib_metadata.version(__name__)
