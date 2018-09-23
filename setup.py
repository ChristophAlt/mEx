from package_settings import NAME, VERSION, PACKAGES, DESCRIPTION
from setuptools import setup
from pathlib import Path
import json
import urllib.request
from functools import lru_cache


@lru_cache(maxsize=50)
def _get_github_sha(github_install_url: str):
    """From the github_install_url get the hash of the latest commit"""
    repository = Path(github_install_url).stem.split('#egg', 1)[0]
    organisation = Path(github_install_url).parent.stem
    with urllib.request.urlopen(f'https://api.github.com/repos/{organisation}/{repository}/commits/master') as response:
        return json.loads(response.read())['sha']


setup(
    name=NAME,
    version=VERSION,
    long_description=DESCRIPTION,
    author='Christoph Alt',
    author_email='christoph.alt@dfki.de',
    packages=PACKAGES,
    include_package_data=True,
    install_requires=[
        'spacy==2.0.12',
        'scikit-learn==0.19.2',
        'pytest==3.8.1',
        'fire==0.1.3'
    ],
    dependency_links=[
        'git+git://github.com/ChristophAlt/flair.git#egg=flair-' + _get_github_sha(
            'git+git://github.com/ChristophAlt/flair.git#egg=flair')
    ],
    package_data={
        '': ['*.*'],
    },
)