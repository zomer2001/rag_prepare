from rag_graph_demo.config import Settings


def test_default_query_mode_is_mix() -> None:
    settings = Settings(_env_file=None)

    assert settings.lightrag_query_mode == "mix"


def test_required_data_directories_are_scoped_to_data_dir(tmp_path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)

    assert all(path.parent == tmp_path for path in settings.required_data_dirs)

