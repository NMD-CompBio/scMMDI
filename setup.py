import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scMMDI",# Replace with your own username
    version="0.0.1",
    author="Mengdi Nan",
    author_email="2139698547@qq.com",
    description="A deep learning framework for single cell multimode data mosaic integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NMD-CompBio/scMMDI",
    packages=setuptools.find_packages(),
    py_modules=['scMMDI.model', 'scMMDI.scMMDI', 'scMMDI.constant', 'scMMDI.dataset', 'scMMDI.decoders',
                'scMMDI.discriminator', 'scMMDI.encoders', 'scMMDI.estimator', 'scMMDI.genomic', 'scMMDI.network',
                'scMMDI.plotting', 'scMMDI.preprocess', 'scMMDI.trainer', 'scMMDI.typehint', 'scMMDI.utils',
                'scMMDI.utils_dev'],  # 添加多个模块
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

