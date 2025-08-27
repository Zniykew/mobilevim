# 新增config.py文件（与文档2完全一致）
class Config:
    # 数据集
    data_root = "/workspace/datasets/NYU_Depth_V2"
    input_size = (480, 640)
    depth_scale = 1000.0
    max_depth = 10.0

    # 模型
    encoder_params = {
        "model_type": "xx_small",
        "embed_dims": [192, 384, 448],
        "global_ratio": [0.8, 0.7, 0.6],
        "local_ratio": [0.2, 0.2, 0.3],
        "kernels": [7, 5, 3],
        "drop_path": 0,
        "ssm_ratio": 2
    }
    decoder_params = {
        "hidden_dim": 256,
        "depths": 3,
        "num_heads": 4,
        "anchor_points": 16,
        "expansion": 4,
        "afp_latent_dim": 128,
        "num_latents": 64,
        "isd_depths": 2
    }

    # 系统
    num_workers = 2
    amp = True