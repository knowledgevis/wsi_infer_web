#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()


setup(
    author="Curtis Lisle",
    author_email='curtislisle@knowledgevis.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3'
    ],
    description='Easy Mode Apps hosted by Girder 3',
    license='Apache Software License 2.0',
    long_description=readme,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='girder-plugin, web_apps',
    name='Web Apps',
    packages=find_packages(exclude=['test', 'test.*']),
    url='https://github.com/knowledgevis/wsi_infer_web',
    version='0.1.0',
    zip_safe=False,
    install_requires=[
        'girder>=3.0.0a1',
        'girder-jobs>=3.0.0a1',
        'girder-worker-utils'
    ],
    entry_points={
        'girder.plugin': [
            'arbor_nova = arbor_nova:ArborNovaGirderPlugin'
        ]
    }
)
