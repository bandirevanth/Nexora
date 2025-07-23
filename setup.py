from setuptools import setup, find_packages

setup(
    name='nexora',
    version='0.14.7',
    packages=find_packages(),
    install_requires=[
        'requests',
        'docstring_parser',
        'litellm',
        'colorlog',
        'prompt_toolkit==3.0.43',
        'anthropic>=0.25.6',
        'openai>=1.0',
        'click>=8.1.7',
        'getch',
        'markdown2'
    ],
    python_requires='>=3.6',
    author='Bandi Revanth',
    description='Nexora: CLI Utility for Integrating LLMs into the Shell',
    license='MIT',
    keywords='languagemodel tools shell',
    url='https://github.com/bandirevanth/nexora',
    entry_points={
        'console_scripts': [
            'nexora = nexora.__main__:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
        "Operating System :: POSIX :: Linux",
    ]
)
