from flash.data.filters import FilterWrapper, KeyFilter, KeyFilterConfig


class TestKeyFilter:
    def test_key_filter(self):
        filter = KeyFilter(KeyFilterConfig(keys=["a", "b"]))
        assert filter({"a": 1, "b": 2, "c": 3})
        assert not filter({"a": 1})
        assert not filter({"b": 2})
        assert not filter({"c": 3})

    def test_key_filter_single_key(self):
        filter = KeyFilter(KeyFilterConfig(keys="a"))
        assert filter({"a": 1, "b": 2})
        assert not filter({"b": 2})


class TestFilterWrapper:
    def test_filter_wrapper(self):
        filter = FilterWrapper(
            [
                KeyFilter(KeyFilterConfig(keys=["a", "b"])),
                KeyFilter(KeyFilterConfig(keys="c")),
            ]
        )
        assert not filter({"a": 1, "b": 2, "c": 3})
        assert not filter({"a": 1})
        assert not filter({"b": 2})
        assert not filter({"c": 3})

    def test_filter_wrapper(self):
        filter = FilterWrapper(
            [
                KeyFilter(KeyFilterConfig(keys=["a", "b"])),
                KeyFilter(KeyFilterConfig(keys=["a", "c"])),
            ]
        )
        assert filter({"a": 1, "b": 2, "c": 3})
        assert not filter({"a": 1})
        assert not filter({"b": 2})
        assert not filter({"c": 3})
