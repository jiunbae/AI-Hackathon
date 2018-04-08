#nsml: floydhub/pytorch:0.3.0-gpu.cuda8cudnn6-py3.17

from distutils.core import setup
setup(
    name='naver2vec movie review',
    version='0.1',
    description='',
    install_requires=[
        'JPype1',
        'konlpy'
    ]
)
