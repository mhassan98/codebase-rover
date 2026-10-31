from app.indexing.chunker import chunk_source


SAMPLE = '''"""module"""

def alpha():
    return 1


def beta(x):
    return x
'''


def test_python_chunks_match_source_and_cover_code() -> None:
    chunks = chunk_source("pkg/mod.py", SAMPLE)
    lines = SAMPLE.splitlines()
    covered: set[int] = set()
    for chunk in chunks:
        assert chunk.content == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.end_line >= chunk.start_line
        covered.update(range(chunk.start_line, chunk.end_line + 1))
    for number, line in enumerate(lines, start=1):
        if line.strip():
            assert number in covered


def test_tree_sitter_splits_functions() -> None:
    chunks = chunk_source("pkg/mod.py", SAMPLE)
    alpha = [chunk for chunk in chunks if chunk.symbol_name == "alpha"]
    beta = [chunk for chunk in chunks if chunk.symbol_name == "beta"]
    assert alpha and beta
    assert "return 1" in alpha[0].content
    assert "def beta" not in alpha[0].content
    assert "return x" in beta[0].content
    assert "return 1" not in beta[0].content


def test_decorator_stays_with_function() -> None:
    source = "@traced\ndef alpha():\n    return 1\n"
    chunks = chunk_source("pkg/mod.py", source)
    alpha = [chunk for chunk in chunks if chunk.symbol_name == "alpha"]
    assert alpha
    assert "@traced" in alpha[0].content


def test_plain_text_still_gets_a_window() -> None:
    source = "alpha ships crates\nbeta scans tickets\n"
    chunks = chunk_source("notes.txt", source)
    assert chunks
    assert chunks[0].chunk_kind == "window"
    assert "alpha ships crates" in chunks[0].content
