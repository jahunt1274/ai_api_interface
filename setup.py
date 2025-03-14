from setuptools import setup, find_packages

setup(
    name="ai_service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "tenacity>=8.0.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
    },
    python_requires=">=3.8",
    author="jahunt1274",
    author_email="jahunt1274@gmail.com",
    description="A service for interacting with AI APIs",
    keywords="AI, API, service",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)