from setuptools import setup, find_packages

setup(
    name='toc_btstn',
    version='1.0',
    description='Bidirectional Time-series State Transfer Network for Target-directed Adptive Control of Metabolic Processes',
    author='Shaohua Xu, Xin Chen',
    author_email='12218022@zju.edu.cn',
    packages=find_packages(),
    install_requires=[
        'torch==2.7.0',
        'numpy==2.0.2',
        'pandas==2.2.3',
        'scikit-learn==1.6.1'
    ]
)
