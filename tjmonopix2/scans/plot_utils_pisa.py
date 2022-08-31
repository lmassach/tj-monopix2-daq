"""Utilities for plotting scripts."""
import tables as tb

__all__ = ['get_config_dict']


def get_config_dict(h5_file):
    """Returns the configuration stored in an h5 as a dictionary of strings.

    Example usage:
        f = tb.open_file("path/to/file.h5")
        cfg = get_config_dict(f)
        chip_serial_number = cfg["configuration_in.chip.settings.chip_sn"]
    """
    if isinstance(h5_file, str):  # Also accepts a path to the file
        with tb.open_file(h5_file) as f:
            return get_config_dict(f)
    res = {}
    for cfg_path in ['configuration_in', 'configuration_out']:
        for node in h5_file.walk_nodes(f"/{cfg_path}"):
            if isinstance(node, tb.Table):
                directory = node._v_pathname.strip("/").replace("/", ".")
                try:
                    for a, b in node[:]:
                        res[f"{directory}.{str(a, encoding='utf8')}"] = str(b, encoding='utf8')
                except Exception:
                    pass  # print("Could not read node", node._v_pathname)
    return res
