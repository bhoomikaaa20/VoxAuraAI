from voxaura.config import load_settings


def test_env_file_edits_are_picked_up_without_a_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-old\nCHUNK_SIZE=500\n")
    assert load_settings(env_file).openai_api_key == "sk-old"

    env_file.write_text("OPENAI_API_KEY= sk-new \n")  # the user fixes the key while the app runs
    settings = load_settings(env_file)
    assert settings.openai_api_key == "sk-new"
    assert settings.chunk_size == 800  # removed from the file -> back to the default


def test_real_environment_variables_win_over_the_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-from-env")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_CHAT_MODEL=gpt-from-file\n")
    assert load_settings(env_file).chat_model == "gpt-from-env"


def test_missing_keys_are_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    assert load_settings(tmp_path / "missing.env").missing_keys() == ["ASSEMBLYAI_API_KEY", "OPENAI_API_KEY"]
