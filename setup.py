from distutils.core import setup


setup(
    name='lsdo_utils',
    version='1',
    packages=[
        'lsdo_utils',
    ],
    install_requires=[
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
    ],
)