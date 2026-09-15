"""Metadata-only upgrades for explicitly supported financial training protocols."""

from __future__ import annotations

from copy import deepcopy
import json

from .denoising import encoder_denoising_contract
from .reference import reference_identity, validate_reference_config
from .rewards import LEGACY_TRACKING_REWARD_CONTRACT, TRACKING_REWARD_CONTRACT, validate_reward_contract


def _metadata_json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _validate_protocol(payload, schema):
    if type(schema) is not int or schema not in (1, 2, 3, 4):
        raise ValueError("Unsupported training checkpoint schema")
    source_reward = LEGACY_TRACKING_REWARD_CONTRACT if schema <= 2 else TRACKING_REWARD_CONTRACT
    reward = payload.get("reward_contract", source_reward if schema <= 2 else None)
    validate_reward_contract(reward, allow_legacy=True)
    if reward != source_reward:
        raise ValueError("Checkpoint reward contract does not match its source schema")
    env_config = payload.get("env_config")
    if not isinstance(env_config, dict):
        raise ValueError("Checkpoint encoder denoising requires an environment configuration")
    if schema <= 3:
        if ("encoder_denoising_contract" in payload
                or env_config.get("encoder_denoising", False) is not False):
            raise ValueError("Legacy checkpoint schema has unexpected encoder denoising metadata")
    else:
        try:
            actual, expected = payload.get("encoder_denoising_contract"), encoder_denoising_contract()
            compatible = actual == expected and _metadata_json(actual) == _metadata_json(expected)
        except (TypeError, ValueError):
            compatible = False
        if not compatible or env_config.get("encoder_denoising") is not True:
            raise ValueError("Checkpoint encoder denoising contract is incompatible")
    return source_reward


def _validate_reference(reference, source_reward):
    validate_reference_config(reference)
    if reference["reward_contract"] != source_reward:
        raise ValueError("Checkpoint reference reward contract differs from its original source protocol")


def prepare_resume_checkpoint(checkpoint, *, reference_config=None):
    """Validate the original protocol, then copy only metadata that must change.

    Model and optimizer state remain shared with the input checkpoint. Legacy
    resumes restart episodes under Reward V1 with encoder denoising; they do not
    reproduce the old training objective or in-flight rollout state.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("Training checkpoint must be a dictionary")
    schema = checkpoint.get("schema_version")
    source_reward = _validate_protocol(checkpoint, schema)
    saved_reference = checkpoint.get("reference_config")
    if reference_config is not None and saved_reference is not None:
        raise ValueError("Explicit reference configuration cannot override a saved reference")
    reference = saved_reference if saved_reference is not None else reference_config
    if reference is not None:
        _validate_reference(reference, source_reward)

    history = checkpoint.get("training_migrations", [])
    if not isinstance(history, list):
        raise ValueError("Checkpoint training migrations must be a JSON-serializable list")
    try:
        json.dumps(history, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint training migrations must be a JSON-serializable list") from error

    upgraded = dict(checkpoint)
    upgraded["training_migrations"] = deepcopy(history)
    if reference_config is not None:
        upgraded["reference_config"] = deepcopy(reference_config)
        upgraded["reference_provenance"] = "legacy_reference_unverified"
    if schema == 4:
        return upgraded

    upgraded.update(
        schema_version=4,
        reward_contract=deepcopy(TRACKING_REWARD_CONTRACT),
        encoder_denoising_contract=encoder_denoising_contract(),
        env_config=deepcopy(checkpoint["env_config"]),
    )
    upgraded["env_config"]["encoder_denoising"] = True
    if reference is not None:
        upgraded["reference_config"] = deepcopy(reference)
        upgraded["reference_config"]["reward_contract"] = deepcopy(TRACKING_REWARD_CONTRACT)
    upgraded["training_migrations"].append({
        "source_schema": schema, "target_schema": 4, "source_iteration": checkpoint["iteration"],
        "source_reward_contract": deepcopy(source_reward),
        "target_reward_contract": deepcopy(TRACKING_REWARD_CONTRACT),
        "source_encoder_denoising_contract": None,
        "target_encoder_denoising_contract": encoder_denoising_contract(),
    })
    return upgraded


def legacy_run_recipe_matches(existing, resolved, migrations):
    """Match an old run recipe only through a recorded, known protocol upgrade."""
    if (not isinstance(existing, dict) or not isinstance(resolved, dict)
            or not isinstance(migrations, list) or not migrations):
        return False
    try:
        _validate_protocol(resolved, 4)
        _validate_reference(resolved.get("reference_config"), TRACKING_REWARD_CONTRACT)
        json.dumps(migrations, allow_nan=False)
        for migration in reversed(migrations):
            if not isinstance(migration, dict):
                continue
            schema = migration.get("source_schema")
            if type(schema) is not int or schema not in (1, 2, 3):
                continue
            source_reward = _validate_protocol(existing, schema)
            expected_migration = {
                "source_schema": schema, "target_schema": 4,
                "source_iteration": migration.get("source_iteration"),
                "source_reward_contract": source_reward,
                "target_reward_contract": TRACKING_REWARD_CONTRACT,
                "source_encoder_denoising_contract": None,
                "target_encoder_denoising_contract": encoder_denoising_contract(),
            }
            if (type(migration.get("source_iteration")) is not int
                    or _metadata_json(migration) != _metadata_json(expected_migration)):
                continue
            _validate_reference(existing.get("reference_config"), source_reward)
            adapted = deepcopy(existing)
            adapted["reward_contract"] = deepcopy(TRACKING_REWARD_CONTRACT)
            adapted["encoder_denoising_contract"] = encoder_denoising_contract()
            adapted["env_config"]["encoder_denoising"] = True
            adapted["reference_config"]["reward_contract"] = deepcopy(TRACKING_REWARD_CONTRACT)
            adapted["reference_config"] = reference_identity(adapted["reference_config"])
            comparable = {**resolved, "reference_config": reference_identity(resolved["reference_config"])}
            return _metadata_json(adapted) == _metadata_json(comparable)
    except (KeyError, TypeError, ValueError):
        return False
    return False
