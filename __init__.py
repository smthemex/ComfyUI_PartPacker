# ComfyUI_PartPacker - native refactor.
# The old wrapper files (PartPacker_node.py / PartPacker/ / node_utils.py) are
# kept on disk for reference but are NOT imported here, so the plugin no longer
# depends on rembg / kiui / fpsample / meshiki / transformers.

from .partpacker_native.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
