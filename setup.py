from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    # TODO: This is a hack to get around the fact that we can't read the requirements.txt file, we should fix this.
    # def _read_reqs(relpath):
    #     fullpath = os.path.join(Path(__file__).parent, relpath)
    #     with open(fullpath) as f:
    #         return [
    #             s.strip()
    #             for s in f.readlines()
    #             if (s.strip() and not s.startswith("#"))
    #         ]

    REQUIREMENTS = [
        # ===== 纯工具依赖（安全）=====
        "einops",
        "einops-exts",
        "pillow",
        "more-itertools",
        "datasets",
        "braceexpand",
        "webdataset",
        "nltk",
        "scipy",
        "inflection",

        # ===== logging / misc =====
        "wandb",

        # ===== CLIP utils（eval 用）=====
        "open_clip_torch>=2.16.0",
    ]


    setup(
        name="icl-qwen-vqa",
        packages=find_packages(),
        include_package_data=True,
        version="1.0.0",
        license="MIT",
        description="In-Context Learning for Visual Question Answering using Qwen2.5-VL",
        long_description=long_description,
        long_description_content_type="text/markdown",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning"],
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.11",
        ],
    )
