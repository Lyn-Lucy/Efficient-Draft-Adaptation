from setuptools import setup, find_packages

setup(
    name='eda',
    version='1.0.0',
    description='Efficient Domain Adaptation for Speculative Decoding via Shared-Private Mixture-of-Experts',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.26.0",
        "deepspeed>=0.12.0",
        "datasets>=2.14.0",
        "safetensors",
        "fschat==0.2.31",
        "shortuuid",
        "tqdm",
    ],
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
)
