set -e
# install
eval "MAX_JOBS=48 python setup.py install"

# # build whl
# eval "MAX_JOBS=48 python setup.py bdist_wheel"
# pip install dist/flash_attn-0.2.6.post2-cp38-cp38-linux_x86_64.whl

