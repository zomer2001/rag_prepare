from rag_graph_demo.config import Settings


def test_default_query_mode_is_mix() -> None:
    settings = Settings(_env_file=None)

    assert settings.lightrag_query_mode == "mix"


def test_local_embedding_and_controlled_entity_are_safe_defaults() -> None:
    fields = Settings.model_fields

    assert fields["embedding_provider"].default == "local"
    assert fields["embedding_batch_size"].default == 8
    assert fields["embedding_max_concurrency"].default == 2
    assert fields["bailian_embedding_model"].default == "text-embedding-v4"
    assert fields["zhipu_embedding_model"].default == "embedding-3"
    assert fields["llm_max_concurrency"].default == 4
    assert fields["light_entity_max_gleaning"].default == 0
    assert fields["light_entity_max_entities"].default == 4
    assert fields["light_entity_max_relationships"].default == 4
    assert fields["light_entity_max_records"].default == 8
    assert fields["graph_index_concurrency"].default == 2
    assert fields["event_extraction_max_events"].default == 12
    assert fields["event_extraction_max_events_per_part"].default == 5
    assert fields["event_index_concurrency"].default == 3

    local_settings = Settings(_env_file=None, embedding_provider="local")
    api_settings = Settings(_env_file=None, embedding_provider="zhipu")
    assert local_settings.effective_embedding_concurrency == 1
    assert api_settings.effective_embedding_concurrency == 2


def test_required_data_directories_are_scoped_to_data_dir(tmp_path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)

    assert all(path.parent == tmp_path for path in settings.required_data_dirs)
