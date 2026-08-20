from models.rdt_runner import RDTRunner
import torch


def test_cosine_scheduler_uses_lambda_clip() -> None:
    config = {
        "lang_adaptor": "linear",
        "img_adaptor": "linear",
        "state_adaptor": "linear",
        "rdt": {"hidden_size": 16, "depth": 1, "num_heads": 1, "cond_pos_embed_type": "multimodal"},
        "noise_scheduler": {
            "num_train_timesteps": 1000,
            "num_inference_timesteps": 5,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
            "clip_sample": False,
            "lambda_min_clipped": -5.1,
        },
    }
    runner = RDTRunner(
        action_dim=4,
        pred_horizon=2,
        config=config,
        lang_token_dim=4,
        img_token_dim=4,
        state_token_dim=4,
        max_lang_cond_len=2,
        img_cond_len=4,
        lang_pos_embed_config=[("lang", -2)],
        img_pos_embed_config=[("image", (1, 2, -2))],
    )
    assert runner.noise_scheduler_sample.config.lambda_min_clipped == -5.1
    assert runner.noise_scheduler_sample.config.thresholding is False


def test_velocity_prediction_target_is_supported() -> None:
    config = {
        "lang_adaptor": "linear",
        "img_adaptor": "linear",
        "state_adaptor": "linear",
        "rdt": {"hidden_size": 16, "depth": 1, "num_heads": 1, "cond_pos_embed_type": "multimodal"},
        "noise_scheduler": {
            "num_train_timesteps": 1000,
            "num_inference_timesteps": 5,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "v_prediction",
            "clip_sample": False,
        },
    }
    runner = RDTRunner(
        action_dim=4,
        pred_horizon=2,
        config=config,
        lang_token_dim=4,
        img_token_dim=4,
        state_token_dim=4,
        max_lang_cond_len=2,
        img_cond_len=4,
        lang_pos_embed_config=[("lang", -2)],
        img_pos_embed_config=[("image", (1, 2, -2))],
        dtype=torch.float32,
    )
    loss = runner.compute_loss(
        lang_tokens=torch.zeros(1, 2, 4),
        lang_attn_mask=torch.ones(1, 2, dtype=torch.bool),
        img_tokens=torch.zeros(1, 4, 4),
        state_tokens=torch.zeros(1, 1, 4),
        action_gt=torch.zeros(1, 2, 4),
        action_mask=torch.ones(1, 1, 4),
        ctrl_freqs=torch.ones(1),
        noise=torch.ones(1, 2, 4),
        timesteps=torch.tensor([999]),
    )
    assert torch.isfinite(loss)
