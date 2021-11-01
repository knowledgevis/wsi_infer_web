#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'girder>=3.0.0a1'
]

setup(
    author="Curtis Lisle",
    author_email='clisle@knowledgevis.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    description='Image Inference API as a girder3 plugin',
    install_requires=requirements,
    license='Proprietary/NA',
    long_description=readme,
    include_package_data=True,
    keywords='girder-plugin, inference',
    name='inference',
    packages=find_packages(exclude=['test', 'test.*']),
    url='https://github.com/knowledgevis/web_infer_web',
    version='0.1.0',
    zip_safe=False,
    entry_points={
        'girder.plugin': [
            'inference = inference:GirderPlugin'
        ]
    }
)
