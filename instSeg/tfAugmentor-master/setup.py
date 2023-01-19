import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tfAugmentor",
    version="1.3",
    author="Long Chen",
    author_email="looooong.chen@gmail.com",
    description="An image augmentation library for tensorflow. "
                "Easy use with tf.data.Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/looooongChen/tfAugmentor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tensorflow>=2.0.0',
        'numpy',
    ],
)
