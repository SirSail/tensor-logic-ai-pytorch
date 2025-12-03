import torch

from tensorlogic.nn import MLPConfig, MLPProgram


def main() -> None:
    cfg = MLPConfig(input_dim=4, hidden_dims=[8, 4], output_dim=1)
    model = MLPProgram(cfg)

    x = torch.randn(2, 4)
    y = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)
    print("Output sample:\n", y)


if __name__ == "__main__":
    main()
