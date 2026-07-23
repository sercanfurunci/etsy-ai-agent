"""
Tests for agent/providers/comfyui_workflows.py — pure functions, no network.
"""
import json
import pytest

from agent.providers.comfyui_workflows import (
    build_sdxl_workflow,
    build_flux_schnell_workflow,
    SDXL_OUTPUT_NODE_ID,
    FLUX_OUTPUT_NODE_ID,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _all_class_types(workflow: dict) -> set[str]:
    """Collect all class_type values across all nodes."""
    return {node["class_type"] for node in workflow.values()}


def _all_input_values(workflow: dict) -> list:
    """Flatten all input values from all nodes (for quick membership checks)."""
    vals = []
    for node in workflow.values():
        for v in node.get("inputs", {}).values():
            vals.append(v)
    return vals


def _is_json_serializable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


# ── SDXL workflow tests ────────────────────────────────────────────────────────

class TestBuildSdxlWorkflow:

    def _make(self, **kwargs):
        defaults = dict(
            prompt="ukiyo-e crane over misty river",
            negative_prompt="blurry, lowres",
            checkpoint_name="sdxl_base.safetensors",
            seed=42,
        )
        defaults.update(kwargs)
        return build_sdxl_workflow(**defaults)

    def test_required_node_class_names_present(self):
        wf = self._make()
        types = _all_class_types(wf)
        assert "CheckpointLoaderSimple" in types
        assert "CLIPTextEncode" in types
        assert "EmptyLatentImage" in types
        assert "KSampler" in types
        assert "VAEDecode" in types
        assert "SaveImage" in types

    def test_output_node_id_present(self):
        wf = self._make()
        assert SDXL_OUTPUT_NODE_ID in wf
        assert wf[SDXL_OUTPUT_NODE_ID]["class_type"] == "SaveImage"

    def test_prompt_preserved_exactly(self):
        prompt = "ukiyo-e crane over misty river"
        wf = self._make(prompt=prompt)
        # Find CLIPTextEncode node with positive prompt
        clip_texts = [
            node["inputs"]["text"]
            for node in wf.values()
            if node["class_type"] == "CLIPTextEncode"
        ]
        assert prompt in clip_texts

    def test_negative_prompt_preserved_exactly(self):
        neg = "blurry, lowres, watermark"
        wf = self._make(negative_prompt=neg)
        clip_texts = [
            node["inputs"]["text"]
            for node in wf.values()
            if node["class_type"] == "CLIPTextEncode"
        ]
        assert neg in clip_texts

    def test_checkpoint_name_in_workflow(self):
        wf = self._make(checkpoint_name="my_sdxl.safetensors")
        node1 = wf["1"]
        assert node1["class_type"] == "CheckpointLoaderSimple"
        assert node1["inputs"]["ckpt_name"] == "my_sdxl.safetensors"

    def test_vae_name_none_uses_checkpoint_vae(self):
        wf = self._make(vae_name=None)
        # No VAELoader node
        types = _all_class_types(wf)
        assert "VAELoader" not in types
        # VAEDecode should reference node 1 (checkpoint), port 2
        node6 = wf["6"]
        assert node6["class_type"] == "VAEDecode"
        assert node6["inputs"]["vae"] == ["1", 2]

    def test_vae_name_set_uses_vae_loader(self):
        wf = self._make(vae_name="my_vae.safetensors")
        types = _all_class_types(wf)
        assert "VAELoader" in types
        # Find VAELoader node (should be "7")
        node7 = wf["7"]
        assert node7["class_type"] == "VAELoader"
        assert node7["inputs"]["vae_name"] == "my_vae.safetensors"
        # VAEDecode should reference node 7
        node6 = wf["6"]
        assert node6["inputs"]["vae"] == ["7", 0]

    def test_width_height_in_empty_latent(self):
        wf = self._make(width=832, height=1216)
        node4 = wf["4"]
        assert node4["class_type"] == "EmptyLatentImage"
        assert node4["inputs"]["width"] == 832
        assert node4["inputs"]["height"] == 1216

    def test_seed_in_ksampler(self):
        wf = self._make(seed=12345)
        node5 = wf["5"]
        assert node5["class_type"] == "KSampler"
        assert node5["inputs"]["seed"] == 12345

    def test_steps_cfg_sampler_scheduler_in_ksampler(self):
        wf = self._make(steps=25, cfg=6.5, sampler_name="dpmpp_2m", scheduler="karras")
        node5 = wf["5"]
        assert node5["inputs"]["steps"] == 25
        assert node5["inputs"]["cfg"] == 6.5
        assert node5["inputs"]["sampler_name"] == "dpmpp_2m"
        assert node5["inputs"]["scheduler"] == "karras"

    def test_result_is_json_serializable(self):
        wf = self._make()
        assert _is_json_serializable(wf)

    def test_batch_size_1_works(self):
        wf = self._make(batch_size=1)
        node4 = wf["4"]
        assert node4["inputs"]["batch_size"] == 1

    def test_filename_prefix_in_save_image(self):
        wf = self._make(filename_prefix="my_prefix")
        save_node = wf[SDXL_OUTPUT_NODE_ID]
        assert save_node["inputs"]["filename_prefix"] == "my_prefix"


# ── FLUX workflow tests ────────────────────────────────────────────────────────

class TestBuildFluxSchnellWorkflow:

    def _make(self, **kwargs):
        defaults = dict(
            prompt="ukiyo-e crane at sunrise",
            unet_name="flux1-schnell.safetensors",
            clip_l_name="clip_l.safetensors",
            t5xxl_name="t5xxl_fp16.safetensors",
            vae_name="ae.safetensors",
            seed=99,
        )
        defaults.update(kwargs)
        return build_flux_schnell_workflow(**defaults)

    def test_required_node_class_names_present(self):
        wf = self._make()
        types = _all_class_types(wf)
        assert "UNETLoader" in types
        assert "DualCLIPLoader" in types
        assert "VAELoader" in types
        assert "CLIPTextEncode" in types
        assert "ModelSamplingFlux" in types
        assert "RandomNoise" in types
        assert "BasicGuider" in types
        assert "KSamplerSelect" in types
        assert "BasicScheduler" in types
        assert "SamplerCustomAdvanced" in types
        assert "VAEDecode" in types
        assert "SaveImage" in types

    def test_output_node_id_present(self):
        wf = self._make()
        assert FLUX_OUTPUT_NODE_ID in wf
        assert wf[FLUX_OUTPUT_NODE_ID]["class_type"] == "SaveImage"

    def test_prompt_preserved_exactly(self):
        prompt = "ukiyo-e crane at sunrise with golden light"
        wf = self._make(prompt=prompt)
        clip_texts = [
            node["inputs"].get("text")
            for node in wf.values()
            if node["class_type"] == "CLIPTextEncode"
        ]
        assert prompt in clip_texts

    def test_negative_prompt_not_wired_to_conditioning(self):
        """FLUX Schnell does not use negative prompts."""
        wf = self._make()
        clip_encode_nodes = [
            node for node in wf.values()
            if node["class_type"] == "CLIPTextEncode"
        ]
        # There should be exactly 1 CLIPTextEncode (positive only)
        assert len(clip_encode_nodes) == 1

    def test_unet_name_in_workflow(self):
        wf = self._make(unet_name="flux1-schnell.safetensors")
        unet_node = next(n for n in wf.values() if n["class_type"] == "UNETLoader")
        assert unet_node["inputs"]["unet_name"] == "flux1-schnell.safetensors"

    def test_clip_names_in_dual_clip_loader(self):
        wf = self._make(clip_l_name="clip_l.safetensors", t5xxl_name="t5xxl_fp16.safetensors")
        clip_node = next(n for n in wf.values() if n["class_type"] == "DualCLIPLoader")
        assert clip_node["inputs"]["clip_name1"] == "clip_l.safetensors"
        assert clip_node["inputs"]["clip_name2"] == "t5xxl_fp16.safetensors"
        assert clip_node["inputs"]["type"] == "flux"

    def test_vae_name_in_vae_loader(self):
        wf = self._make(vae_name="ae.safetensors")
        vae_node = next(n for n in wf.values() if n["class_type"] == "VAELoader")
        assert vae_node["inputs"]["vae_name"] == "ae.safetensors"

    def test_width_height_in_latent_and_model_sampling(self):
        wf = self._make(width=1024, height=1536)
        latent_node = next(n for n in wf.values() if "LatentImage" in n["class_type"])
        assert latent_node["inputs"]["width"] == 1024
        assert latent_node["inputs"]["height"] == 1536
        flux_sampler_node = next(n for n in wf.values() if n["class_type"] == "ModelSamplingFlux")
        assert flux_sampler_node["inputs"]["width"] == 1024
        assert flux_sampler_node["inputs"]["height"] == 1536

    def test_seed_in_random_noise(self):
        wf = self._make(seed=777)
        noise_node = next(n for n in wf.values() if n["class_type"] == "RandomNoise")
        assert noise_node["inputs"]["noise_seed"] == 777

    def test_steps_in_basic_scheduler(self):
        wf = self._make(steps=4)
        sched_node = next(n for n in wf.values() if n["class_type"] == "BasicScheduler")
        assert sched_node["inputs"]["steps"] == 4

    def test_result_is_json_serializable(self):
        wf = self._make()
        assert _is_json_serializable(wf)


# ── Validation error tests ─────────────────────────────────────────────────────

class TestValidationErrors:

    def _sdxl(self, **kwargs):
        defaults = dict(
            prompt="test",
            negative_prompt="",
            checkpoint_name="sdxl.safetensors",
            seed=0,
        )
        defaults.update(kwargs)
        return build_sdxl_workflow(**defaults)

    def _flux(self, **kwargs):
        defaults = dict(
            prompt="test",
            unet_name="flux.safetensors",
            clip_l_name="clip_l.safetensors",
            t5xxl_name="t5xxl.safetensors",
            vae_name="ae.safetensors",
            seed=0,
        )
        defaults.update(kwargs)
        return build_flux_schnell_workflow(**defaults)

    def test_sdxl_width_not_divisible_by_8(self):
        with pytest.raises(ValueError, match="divisible by 8"):
            self._sdxl(width=1025)

    def test_sdxl_height_zero(self):
        with pytest.raises(ValueError):
            self._sdxl(height=0)

    def test_sdxl_height_negative(self):
        with pytest.raises(ValueError):
            self._sdxl(height=-8)

    def test_sdxl_width_exceeds_8192(self):
        with pytest.raises(ValueError, match="8192"):
            self._sdxl(width=8200)

    def test_sdxl_steps_zero(self):
        with pytest.raises(ValueError, match="steps"):
            self._sdxl(steps=0)

    def test_sdxl_steps_over_150(self):
        with pytest.raises(ValueError, match="steps"):
            self._sdxl(steps=151)

    def test_sdxl_batch_size_2(self):
        with pytest.raises(ValueError, match="batch_size"):
            self._sdxl(batch_size=2)

    def test_sdxl_empty_prompt(self):
        with pytest.raises(ValueError, match="prompt"):
            self._sdxl(prompt="")

    def test_sdxl_whitespace_only_prompt(self):
        with pytest.raises(ValueError, match="prompt"):
            self._sdxl(prompt="   ")

    def test_sdxl_empty_checkpoint_name(self):
        with pytest.raises(ValueError, match="checkpoint_name"):
            self._sdxl(checkpoint_name="")

    def test_sdxl_seed_negative(self):
        with pytest.raises(ValueError, match="seed"):
            self._sdxl(seed=-1)

    def test_sdxl_seed_too_large(self):
        with pytest.raises(ValueError, match="seed"):
            self._sdxl(seed=18446744073709551616)

    def test_sdxl_seed_zero_valid(self):
        wf = self._sdxl(seed=0)
        assert wf["5"]["inputs"]["seed"] == 0

    def test_sdxl_seed_max_valid(self):
        wf = self._sdxl(seed=18446744073709551615)
        assert wf["5"]["inputs"]["seed"] == 18446744073709551615

    def test_flux_width_not_divisible_by_8(self):
        with pytest.raises(ValueError, match="divisible by 8"):
            self._flux(width=1023)

    def test_flux_height_zero(self):
        with pytest.raises(ValueError):
            self._flux(height=0)

    def test_flux_steps_zero(self):
        with pytest.raises(ValueError, match="steps"):
            self._flux(steps=0)

    def test_flux_batch_size_2(self):
        with pytest.raises(ValueError, match="batch_size"):
            self._flux(batch_size=2)

    def test_flux_empty_prompt(self):
        with pytest.raises(ValueError, match="prompt"):
            self._flux(prompt="")

    def test_flux_empty_unet_name(self):
        with pytest.raises(ValueError, match="unet_name"):
            self._flux(unet_name="")

    def test_flux_empty_clip_l_name(self):
        with pytest.raises(ValueError, match="clip_l_name"):
            self._flux(clip_l_name="")

    def test_flux_empty_t5xxl_name(self):
        with pytest.raises(ValueError, match="t5xxl_name"):
            self._flux(t5xxl_name="")

    def test_flux_empty_vae_name(self):
        with pytest.raises(ValueError, match="vae_name"):
            self._flux(vae_name="")

    def test_flux_seed_negative(self):
        with pytest.raises(ValueError, match="seed"):
            self._flux(seed=-1)

    def test_flux_seed_too_large(self):
        with pytest.raises(ValueError, match="seed"):
            self._flux(seed=18446744073709551616)
