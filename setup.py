from setuptools import setup, find_packages

setup(
    name='orangedetect',
    version='0.2',
    packages=find_packages(where='src'),
    package_dir={'src': 'src'}

)
