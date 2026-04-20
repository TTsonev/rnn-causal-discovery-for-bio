import torch
from model import RNNModel


def test_rnn_model_forward_and_loss():
    # Setup dummy configurations
    dim_in = 2
    dim_out = 1
    dim_latent = 32
    num_rnn_layers = 2
    batch_size = 4
    seq_len = 30  # Since we divide by 3 in forward pass, this should be divisible by 3

    model = RNNModel(
        dim_in=dim_in,
        dim_out=dim_out,
        dim_latent=dim_latent,
        num_rnn_layers=num_rnn_layers,
        learning_rate=0.001,
        loss_function_str="bce",
    )

    # Creating dummy batch. The real batch is a tuple from the dataloader:
    # x, y, l, m, m_targets, meta, y_cont, freq
    x = torch.zeros(batch_size, seq_len, 2, dtype=torch.float32)
    # the first element of feature axis is used as embedding indicy. 0..3
    x[:, :, 0] = torch.randint(0, 4, (batch_size, seq_len)).float()

    y = torch.randint(0, 2, (batch_size, 1)).float()
    l = torch.tensor([seq_len] * batch_size, dtype=torch.long)
    m = torch.ones(batch_size, seq_len, dtype=torch.float32)
    m_targets = torch.ones(batch_size, 1, dtype=torch.float32)
    meta = torch.zeros(batch_size, dtype=torch.long)
    y_cont = torch.rand(batch_size, 1, dtype=torch.float32)
    freq = torch.rand(batch_size, 64)

    batch = (x, y, l, m, m_targets, meta, y_cont, freq)

    # Test forward pass
    rnn_output, y_pred, y_score = model.forward(batch)

    # Check shapes
    assert rnn_output.shape == (batch_size, dim_latent)
    assert y_pred.shape == (batch_size, dim_out)
    assert y_score.shape == (batch_size, dim_out)

    # Test loss computation
    loss = model.loss(batch, return_y_pred=False)
    assert not torch.isnan(loss)
    assert loss.dim() == 0, "Loss should be a scalar"
