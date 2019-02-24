class TestPyBertUtil(object):

    def test_safe_log10(self):
        """Simple test case to verify pytest and tox is up and working."""
        from pybert.pybert_util import safe_log10
        assert safe_log10(0) == -20
