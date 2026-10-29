"""Bounded inference-only mask reuse for the inspected Transformers CLAP."""
import weakref
import torch


class AttentionMaskCache:
    def __init__(self, layer, byte_limit=2 * 1024**2):
        self.layer = weakref.ref(layer)
        self.original = type(layer).get_attn_mask
        self.byte_limit = byte_limit
        self.key = self.value = None

    def __call__(self, height, width, dtype, device):
        layer = self.layer()
        if (layer.training or torch.is_grad_enabled() or torch.jit.is_tracing()
                or not all(isinstance(v, int) for v in (height, width, layer.window_size, layer.shift_size))):
            return self.original(layer, height, width, dtype, device)
        key = (height, width, dtype, torch.device(device), layer.window_size, layer.shift_size)
        if key == self.key:
            return self.value
        value = self.original(layer, height, width, dtype, device)
        self.key = self.value = None
        if value is None or value.numel() * value.element_size() <= self.byte_limit:
            self.key, self.value = key, value
        return value


def cache_attention_masks(model):
    # Only this inspected implementation creates a pure shape/shift mask and
    # consumes it without mutation. Other versions retain their original path.
    import transformers
    if getattr(transformers, '__version__', None) != '5.3.0' or not hasattr(model, 'modules'):
        return
    from transformers.models.clap.modeling_clap import ClapAudioLayer
    for layer in model.modules():
        if type(layer) is ClapAudioLayer and not isinstance(layer.get_attn_mask, AttentionMaskCache):
            layer.get_attn_mask = AttentionMaskCache(layer)
