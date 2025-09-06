from src.utils.text_cleaning import normalize_text

def test_normalize_basic():
    s = "Visit https://example.com or EMAIL me@foo.com! <b>Great</b> 😄"
    out = normalize_text(s)
    assert "<url>" in out or "<URL>" in out
    assert "<email>" in out or "<EMAIL>" in out
    assert "great" in out
