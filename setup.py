from setuptools import setup, find_packages

setup(
    name='btstn',
    version='1.0',
    description='Bidirectional Time-series State Transfer Network (BTSTN)',
    author='Shaohua Xu, Ting Xu, Yuping Yang, Xin Chen',
    author_email='12218022@zju.edu.cn',
    packages=find_packages(),
    install_requires=[
        'torch==2.0.0',
        'numpy>=1.23.5',
        'pandas>=1.5.2',
        'pypots==0.5'
    ]
)
