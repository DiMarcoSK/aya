from config import DEFAULT_CONFIG_PATH, load_yaml_config


def test_default_config_file_exists():
    assert DEFAULT_CONFIG_PATH.exists()


def test_load_yaml_config_returns_expected_keys():
    config = load_yaml_config()
    expected_keys = {
        "model_name", "max_length", "batch_size", "gradient_accumulation_steps",
        "learning_rate", "num_epochs", "lora_r", "lora_alpha", "lora_dropout",
        "eval_split", "gradient_checkpointing", "seed",
    }
    assert expected_keys.issubset(config.keys())
