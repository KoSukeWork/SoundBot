from pathlib import Path
import sys
import unittest
import torch
from transformers import ClapAudioConfig
from transformers.models.clap.modeling_clap import ClapAudioLayer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from core.clap_masks import AttentionMaskCache, cache_attention_masks


class AttentionMaskTests(unittest.TestCase):
    def layer(self):
        config=ClapAudioConfig(window_size=8)
        return ClapAudioLayer(config,dim=16,input_resolution=(16,16),num_heads=2,shift_size=4).eval()

    def test_cached_mask_and_forward_are_bitwise_identical(self):
        torch.manual_seed(42);layer=self.layer();data=torch.randn(2,256,16)
        original=layer.get_attn_mask
        with torch.inference_mode():
            mask=original(16,16,torch.float32,'cpu')
            expected=layer(data,(16,16))[0]
            cache_attention_masks(layer)
            cached=layer.get_attn_mask(16,16,torch.float32,'cpu')
            self.assertTrue(torch.equal(mask,cached))
            self.assertIs(cached,layer.get_attn_mask(16,16,torch.float32,'cpu'))
            actual=layer(data,(16,16))[0]
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(torch.equal(mask,cached))

    def test_key_covers_shape_dtype_and_mutable_shift_and_respects_budget(self):
        layer=self.layer();original=layer.get_attn_mask;cache=AttentionMaskCache(layer)
        with torch.inference_mode():
            for height,width,dtype,shift in ((16,16,torch.float32,4),(24,16,torch.float32,4),(16,16,torch.float64,4),(16,16,torch.float32,0),(16,16,torch.float32,4)):
                layer.shift_size=shift
                expected=original(height,width,dtype,'cpu');actual=cache(height,width,dtype,'cpu')
                self.assertTrue(actual is None if expected is None else torch.equal(expected,actual))
            bounded=AttentionMaskCache(layer,byte_limit=1)
            bounded(16,16,torch.float32,'cpu')
            self.assertIsNone(bounded.value)
            layer.train();cache.key=cache.value=None
            cache(16,16,torch.float32,'cpu');self.assertIsNone(cache.value)
