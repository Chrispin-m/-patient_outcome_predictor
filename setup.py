from setuptools import setup, find_packages

setup(
    name='patient_outcome_predictor',
    version='0.1.7',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy',
        'streamlit',
    ],
    test_suite='tests',
    description='A package that uses machine learning models to predict patient outcomes based on medical data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='iChrispin',
    author_email='chrismwangicw@gmail.com',
    url='https://github.com/Chrispin-m/-patient_outcome_predictor',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
