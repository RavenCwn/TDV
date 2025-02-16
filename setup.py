from setuptools import setup, find_packages

setup(
    name="coordiff",  # 项目名称
    version="1.0.0",          # 版本号
    author="pgp",       # 作者
    author_email="pgp@pgp.com",  # 作者邮箱
    long_description=open("README.md").read(),  # 详细描述（从README.md读取）
    
    packages=find_packages(),  # 自动发现所有包
    
    # 项目依赖
    install_requires=[
        # 在这里列出你的依赖包
        # 例如: "requests>=2.25.1",
        # "pandas>=1.2.0",
    ],

)
