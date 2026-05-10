import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction

def hello_rnn_lstm_captioning():
    print("Hello from rnn_lstm_captioning.py!")


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, verbose: bool = True):
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")

        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, images: torch.Tensor):
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        features = self.backbone(images)["c5"]
        return features

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    #forward pass for a single timestep of a vanilla RNN that uses a tanh activation function.
    next_h, cache = None, None
    a = x @ Wx + prev_h @ Wh + b
    next_h = torch.tanh(a)
    cache = (x, prev_h, Wx, Wh, a, next_h)
    return next_h, cache

def rnn_step_backward(dnext_h, cache):
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    x, prev_h, Wx, Wh, a, next_h = cache
    # gradient through tanh: d(tanh) = 1 - tanh^2
    da = dnext_h * (1 - next_h ** 2)
    dx = da @ Wx.T
    dprev_h = da @ Wh.T
    dWx = x.T @ da
    dWh = prev_h.T @ da
    db = da.sum(dim=0)
    return dx, dprev_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = None, None
    N, T, D = x.shape
    H = h0.shape[1]
    h = torch.zeros(N, T, H, dtype=x.dtype, device=x.device)
    cache = []
    prev_h = h0
    for t in range(T):
        next_h, step_cache = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = next_h
        cache.append(step_cache)
        prev_h = next_h
    return h, cache

def rnn_backward(dh, cache):
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dx = torch.zeros(N, T, D, dtype=dh.dtype, device=dh.device)
    dWx = torch.zeros(D, H, dtype=dh.dtype, device=dh.device)
    dWh = torch.zeros(H, H, dtype=dh.dtype, device=dh.device)
    db = torch.zeros(H, dtype=dh.dtype, device=dh.device)
    dprev_h = torch.zeros(N, H, dtype=dh.dtype, device=dh.device)
    for t in reversed(range(T)):
        dnext_h = dh[:, t, :] + dprev_h
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db

class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h

class WordEmbedding(nn.Module):
    #simplified version of torch.nn.Embedding.
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):
        out = None
        out = self.W_embed[x]
        return out

def temporal_softmax_loss(x, y, ignore_index=None):
    loss = None
    loss = F.cross_entropy(x.reshape(-1, x.shape[2]), y.reshape(-1), ignore_index=ignore_index, reduction='sum') / x.shape[0]
    return loss

class CaptioningRNN(nn.Module):
    # CaptioningRNN produces captions from images using a recurrent neural network.
    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = "rnn",
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index

        self.image_encoder = ImageEncoder(pretrained=image_encoder_pretrained, verbose=False)
        self.word_embed = WordEmbedding(vocab_size, wordvec_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        if cell_type == "rnn":
            self.rnn = RNN(wordvec_dim, hidden_dim)
            self.feature_proj = nn.Linear(input_dim, hidden_dim)
        elif cell_type == "lstm":
            self.rnn = LSTM(wordvec_dim, hidden_dim)
            self.feature_proj = nn.Linear(input_dim, hidden_dim)
        elif cell_type == "attn":
            self.rnn = AttentionLSTM(wordvec_dim, hidden_dim)
            self.feature_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

    def forward(self, images, captions):
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        features = self.image_encoder(images)  # (N, C, H/32, W/32)

        if self.cell_type in ("rnn", "lstm"):
            # average pool spatial features -> (N, C)
            h0 = self.feature_proj(features.mean(dim=(2, 3)))  # (N, H)
        else:  # attn
            A = self.feature_proj(features)  # (N, H, 4, 4)

        word_vecs = self.word_embed(captions_in)  # (N, T, W)

        if self.cell_type == "rnn":
            hidden = self.rnn(word_vecs, h0)          # (N, T, H)
        elif self.cell_type == "lstm":
            hidden = self.rnn(word_vecs, h0)          # (N, T, H)
        else:  # attn
            hidden = self.rnn(word_vecs, A)           # (N, T, H)

        scores = self.output_proj(hidden)             # (N, T, V)
        loss = temporal_softmax_loss(scores, captions_out, self.ignore_index)

        return loss

    def sample(self, images, max_length=15):
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()
        features = self.image_encoder(images)

        if self.cell_type in ("rnn", "lstm"):
            h = self.feature_proj(features.mean(dim=(2, 3)))  # (N, H)
            c = torch.zeros_like(h)
        else:  # attn
            A = self.feature_proj(features)              # (N, H, 4, 4)
            h = A.mean(dim=(2, 3))
            c = h.clone()

        word = torch.full((N,), self._start, dtype=torch.long, device=images.device)

        for t in range(max_length):
            word_vec = self.word_embed(word)             # (N, W)
            if self.cell_type == "rnn":
                h = self.rnn.step_forward(word_vec, h)
            elif self.cell_type == "lstm":
                h, c = self.rnn.step_forward(word_vec, h, c)
            else:  # attn
                attn, attn_weights = dot_product_attention(h, A)
                h, c = self.rnn.step_forward(word_vec, h, c, attn)
                attn_weights_all[:, t, :, :] = attn_weights

            scores = self.output_proj(h)                 # (N, V)
            word = scores.argmax(dim=1)                  # (N,)
            captions[:, t] = word
        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions

class LSTM(nn.Module):
    #single-layer, uni-directional LSTM module.
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        next_h, next_c = None, None
        # Replace "pass" statement with your code
        H = prev_h.shape[1]
        a = x @ self.Wx + prev_h @ self.Wh + self.b  # (N, 4H)
        ai, af, ao, ag = a[:, :H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:]
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        c0 = torch.zeros_like(
            h0
        )
        hn = None
        N, T, D = x.shape
        H = h0.shape[1]
        hn = torch.zeros(N, T, H, dtype=x.dtype, device=x.device)
        h, c = h0, c0
        for t in range(T):
            h, c = self.step_forward(x[:, t, :], h, c)
            hn[:, t, :] = h
        return hn

def dot_product_attention(prev_h, A):
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None
    # A: (N, H, 4, 4) -> flatten spatial: (N, H, 16)
    A_flat = A.view(N, H, -1)                        # (N, H, 16)
    # scores: prev_h (N, H) x A_flat (N, H, 16) -> (N, 16)
    scores = torch.bmm(prev_h.unsqueeze(1), A_flat).squeeze(1)  # (N, 16)
    scores = scores / math.sqrt(H)
    attn_weights = F.softmax(scores, dim=1).view(N, D_a, D_a)   # (N, 4, 4)
    # attn: A_flat (N, H, 16) x weights (N, 16, 1) -> (N, H)
    attn = torch.bmm(A_flat, attn_weights.view(N, -1, 1)).squeeze(2)  # (N, H)
    return attn, attn_weights

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_h, next_c = None, None
        H = prev_h.shape[1]
        a = x @ self.Wx + prev_h @ self.Wh + attn @ self.Wattn + self.b  # (N, 4H)
        ai, af, ao, ag = a[:, :H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:]
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
        c0 = h0  # Initial cell state, of shape (N, H)
        hn = None
        N, T, D = x.shape
        H = h0.shape[1]
        hn = torch.zeros(N, T, H, dtype=x.dtype, device=x.device)
        h, c = h0, c0
        for t in range(T):
            attn, _ = dot_product_attention(h, A)
            h, c = self.step_forward(x[:, t, :], h, c, attn)
            hn[:, t, :] = h
        return hn

