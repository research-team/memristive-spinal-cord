from setuptools import setup

setup(
    name='memristive-spinal-cord',
    version='0.0.1',
    description='a putative spinal cord neural network',
    url='https://github.com/research-team/memristive-spinal-cord/',
    license='MIT',
    package_data={'memristive_spinal_cord': ['layer1/moraud/afferents/data']},
    install_requires=[
        "neucogar"
    ]
)
