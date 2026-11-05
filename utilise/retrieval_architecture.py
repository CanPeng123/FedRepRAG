import re
import torch
import torch.nn as nn


class PrefixProjector(nn.Module):
    """
    RepRAG-style projector adapted to your injection pipeline.

    RepRAG behavior:
      projector_type = "mlp{d}x_gelu"
      - depth d MLP
      - first layer: input_size -> output_size (RepRAG uses retriever_hidden_size -> LLM hidden_size)
      - for each extra depth: GELU + Linear(output_size -> output_size)

    Your improvements kept:
      - LayerNorm input (stability)
      - Dropout
      - Output LayerNorm (optional but helpful for VLM embedding space)
      - Learned gate alpha (sigmoid)
      - Zero-init last linear so it starts as near no-op
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        projector_type: str = "mlp2x_gelu",
        p_drop: float = 0.1,
        use_ln_in: bool = True,
        use_ln_out: bool = True,
        init_alpha: float = -2.0,   # sigmoid ~0.12
        zero_init_last: bool = True,
    ):
        super().__init__()

        self.use_ln_in = use_ln_in
        self.use_ln_out = use_ln_out

        self.ln_in = nn.LayerNorm(input_size) if use_ln_in else nn.Identity()
        self.drop = nn.Dropout(p_drop)

        # --- RepRAG-style projector_type parsing ---
        m = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if not m:
            raise ValueError(
                f"Unsupported projector_type: {projector_type}. "
                f"Expected format like 'mlp2x_gelu', 'mlp3x_gelu'."
            )
        depth = int(m.group(1))
        if depth < 1:
            raise ValueError(f"MLP depth must be >=1, got {depth}")

        modules = []
        # first layer maps retriever embedding dim -> model hidden dim
        modules.append(nn.Linear(input_size, output_size))
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_size, output_size))

        self.projector = nn.Sequential(*modules)

        self.ln_out = nn.LayerNorm(output_size) if use_ln_out else nn.Identity()

        # global gate (your design)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        # RepRAG doesn’t do this, but your “safe start” is excellent for frozen-LLM training
        if zero_init_last:
            last_linear = None
            # last module should be Linear
            for mod in reversed(self.projector):
                if isinstance(mod, nn.Linear):
                    last_linear = mod
                    break
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,H] or [B,K,H]  (both work with nn.Sequential)

        return: same shape, projected into output_size, gated by sigmoid(alpha)
        """
        x = self.ln_in(x)
        y = self.projector(x)
        y = self.drop(y)
        y = self.ln_out(y)
        return torch.sigmoid(self.alpha) * y
