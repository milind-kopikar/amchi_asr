from setuptools import setup, find_packages

setup(
    name='amchi_asr_local_scripts',
    version='0.0.1',
    packages=['scripts'],
    package_dir={'scripts': 'scripts'},
    include_package_data=True,
)
