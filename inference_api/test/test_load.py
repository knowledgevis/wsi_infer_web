import pytest

from girder.plugin import loadedPlugins


@pytest.mark.plugin('vxapi')
def test_import(server):
    assert 'vxapi' in loadedPlugins()
